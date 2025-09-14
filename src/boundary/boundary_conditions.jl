struct BoundaryCondition <: AbstractBoundaryCondition
    type::BoundaryType
    value::Union{Float64, Function, Nothing}
    direction::Symbol  # :x, :y, :z
    location::Symbol   # :left, :right, :bottom, :top, :front, :back
end


struct BoundaryConditions
    conditions::Dict{Tuple{Symbol, Symbol}, BoundaryCondition}
end

function BoundaryConditions()
    BoundaryConditions(Dict{Tuple{Symbol, Symbol}, BoundaryCondition}())
end

function add_boundary!(bc::BoundaryConditions, condition::BoundaryCondition)
    key = (condition.direction, condition.location)
    bc.conditions[key] = condition
end

function get_boundary(bc::BoundaryConditions, direction::Symbol, location::Symbol)
    key = (direction, location)
    return get(bc.conditions, key, nothing)
end

function apply_boundary_conditions!(grid::StaggeredGrid, state::SolutionState, 
                                  bc::BoundaryConditions, t::Float64)
    if grid.grid_type == TwoDimensional
        apply_2d_boundaries!(grid, state, bc, t)  # Now uses XZ plane
    else
        apply_3d_boundaries!(grid, state, bc, t)
    end
end

function apply_2d_boundaries!(grid::StaggeredGrid, state::SolutionState,
                             bc::BoundaryConditions, t::Float64)
    nx, nz = grid.nx, grid.nz  # XZ plane
    
    # Left boundary (x=0)
    if haskey(bc.conditions, (:x, :left))
        condition = bc.conditions[(:x, :left)]
        apply_u_boundary!(state.u, condition, 1, :, t, :left)
        # For inlet in x-direction, optionally allow a small transverse profile via env var
        if condition.type == Inlet
            A = try
                parse(Float64, get(ENV, "BIOFLOWS_W_INLET_AMP", "0.0"))
            catch
                0.0
            end
            if A != 0.0 && size(state.w, 1) >= 1
                # Sinusoidal profile across z with zero net flux
                for j = 1:size(state.w, 2)
                    zf = grid.zw[j]
                    state.w[1, j] = A * sin(pi * (zf / grid.Lz))
                end
            else
                state.w[1, :] .= 0.0
            end
            # Enforce a small zero-gradient buffer for u at inlet to avoid artificial divergence spikes
            local nfaces = min(8, size(state.u, 1))
            for i = 2:nfaces
                state.u[i, :] .= state.u[i-1, :]
            end
        else
            apply_w_boundary!(state.w, condition, 1, :, t, :left)
        end
    end
    
    # Right boundary (x=Lx)
    if haskey(bc.conditions, (:x, :right))
        condition = bc.conditions[(:x, :right)]
        apply_u_boundary!(state.u, condition, nx+1, :, t, :right)
        # Zero-gradient for transverse component at outlet
        state.w[nx, :] .= state.w[nx-1, :]
    end
    
    # Bottom boundary (z=0)
    if haskey(bc.conditions, (:z, :bottom))
        condition = bc.conditions[(:z, :bottom)]
        apply_u_boundary!(state.u, condition, :, 1, t, :bottom)
        apply_w_boundary!(state.w, condition, :, 1, t, :bottom)
    end
    
    # Top boundary (z=Lz)
    if haskey(bc.conditions, (:z, :top))
        condition = bc.conditions[(:z, :top)]
        apply_u_boundary!(state.u, condition, :, nz, t, :top)
        apply_w_boundary!(state.w, condition, :, nz+1, t, :top)
    end
end


function apply_3d_boundaries!(grid::StaggeredGrid, state::SolutionState,
                             bc::BoundaryConditions, t::Float64)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    # X-direction boundaries (left/right)
    if haskey(bc.conditions, (:x, :left))
        cond = bc.conditions[(:x, :left)]
        apply_u_boundary!(state.u, cond, 1, :, :, t, :left)
        apply_v_boundary!(state.v, cond, 1, :, :, t, :left)
        apply_w_boundary!(state.w, cond, 1, :, :, t, :left)
    end
    if haskey(bc.conditions, (:x, :right))
        cond = bc.conditions[(:x, :right)]
        apply_u_boundary!(state.u, cond, nx+1, :, :, t, :right)
        apply_v_boundary!(state.v, cond, nx, :, :, t, :right)
        apply_w_boundary!(state.w, cond, nx, :, :, t, :right)
    end
    # Y-direction boundaries (bottom/top)
    if haskey(bc.conditions, (:y, :bottom))
        cond = bc.conditions[(:y, :bottom)]
        apply_u_boundary!(state.u, cond, :, 1, :, t, :bottom)
        apply_v_boundary!(state.v, cond, :, 1, :, t, :bottom)
        apply_w_boundary!(state.w, cond, :, 1, :, t, :bottom)
    end
    if haskey(bc.conditions, (:y, :top))
        cond = bc.conditions[(:y, :top)]
        apply_u_boundary!(state.u, cond, :, ny, :, t, :top)
        apply_v_boundary!(state.v, cond, :, ny+1, :, t, :top)
        apply_w_boundary!(state.w, cond, :, ny, :, t, :top)
    end
    # Z-direction boundaries (front/back)
    if haskey(bc.conditions, (:z, :front))
        cond = bc.conditions[(:z, :front)]
        apply_u_boundary!(state.u, cond, :, :, 1, t, :front)
        apply_v_boundary!(state.v, cond, :, :, 1, t, :front)
        apply_w_boundary!(state.w, cond, :, :, 1, t, :front)
    end
    if haskey(bc.conditions, (:z, :back))
        cond = bc.conditions[(:z, :back)]
        apply_u_boundary!(state.u, cond, :, :, nz, t, :back)
        apply_v_boundary!(state.v, cond, :, :, nz, t, :back)
        apply_w_boundary!(state.w, cond, :, :, nz+1, t, :back)
    end
end

function apply_u_boundary!(u::Array, condition::BoundaryCondition, 
                          idx1, idx2, t::Float64, location::Symbol)
    if condition.type == NoSlip
        u[idx1, idx2] .= 0.0
    elseif condition.type == FreeSlip
        # Free slip: du/dn = 0 (no penetration but allow slip)
        if location in [:left, :right]
            u[idx1, idx2] .= u[idx1 == 1 ? 2 : idx1-1, idx2]
        else
            # No normal component
            u[idx1, idx2] .= 0.0
        end
    elseif condition.type == Inlet
        if condition.value isa Function
            u[idx1, idx2] .= condition.value(t)
        else
            u[idx1, idx2] .= condition.value
        end
    elseif condition.type == Outlet
        # Neumann condition: du/dx = 0
        if location == :right
            u[idx1, idx2] .= u[idx1-1, idx2]
        end
    elseif condition.type == Periodic
        # Handled separately in periodic boundary update
    end
end

function apply_v_boundary!(v::Array, condition::BoundaryCondition,
                          idx1, idx2, t::Float64, location::Symbol)
    if condition.type == NoSlip
        v[idx1, idx2] .= 0.0
    elseif condition.type == FreeSlip
        if location in [:bottom, :top]
            v[idx1, idx2] .= v[idx1, idx2 == 1 ? 2 : idx2-1]
        else
            v[idx1, idx2] .= 0.0
        end
    elseif condition.type == Inlet
        if condition.value isa Function
            v[idx1, idx2] .= condition.value(t)
        else
            v[idx1, idx2] .= condition.value !== nothing ? condition.value : 0.0
        end
    elseif condition.type == Outlet
        if location == :top
            v[idx1, idx2] .= v[idx1, idx2-1]
        end
    elseif condition.type == Periodic
        # Handled separately
    end
end

function apply_w_boundary!(w::Array, condition::BoundaryCondition, 
                          idx1, idx2, t::Float64, location::Symbol)
    # Similar to v_boundary but for z-direction
    if condition.type == NoSlip
        w[idx1, idx2] .= 0.0
    elseif condition.type == FreeSlip
        if location in [:bottom, :top]
            w[idx1, idx2] .= w[idx1, idx2 == 1 ? 2 : idx2-1]  
        else
            w[idx1, idx2] .= 0.0
        end
    elseif condition.type == Inlet
        if condition.value isa Function
            w[idx1, idx2] .= condition.value(t)
        else
            w[idx1, idx2] .= condition.value !== nothing ? condition.value : 0.0
        end
    elseif condition.type == Outlet
        if location == :top
            w[idx1, idx2] .= w[idx1, idx2-1]
        end
    end
end

# 3D variants (dispatch on 3 indices)
function apply_u_boundary!(u::Array{T,3}, condition::BoundaryCondition, 
                           i, j, k, t::Float64, location::Symbol) where T
    if condition.type == NoSlip
        u[i, j, k] = 0.0
    elseif condition.type == FreeSlip
        # du/dn = 0 → copy from interior cell face
        if location == :left
            u[1, j, k] = u[2, j, k]
        elseif location == :right
            u[end, j, k] = u[end-1, j, k]
        elseif location == :bottom
            u[:, 1, k] .= u[:, 2, k]
        elseif location == :top
            u[:, end, k] .= u[:, end-1, k]
        elseif location == :front
            u[:, j, 1] .= u[:, j, 2]
        elseif location == :back
            u[:, j, end] .= u[:, j, end-1]
        end
    elseif condition.type == Inlet
        val = condition.value isa Function ? condition.value(t) : condition.value
        u[i, j, k] = val
    elseif condition.type == Outlet
        # Zero gradient in normal direction: copy from interior
        if location == :right
            u[end, j, k] = u[end-1, j, k]
        end
    end
end

function apply_v_boundary!(v::Array{T,3}, condition::BoundaryCondition, 
                           i, j, k, t::Float64, location::Symbol) where T
    if condition.type == NoSlip
        v[i, j, k] = 0.0
    elseif condition.type == FreeSlip
        if location == :bottom
            v[i, 1, k] = v[i, 2, k]
        elseif location == :top
            v[i, end, k] = v[i, end-1, k]
        elseif location == :left
            v[1, :, k] .= v[2, :, k]
        elseif location == :right
            v[end, :, k] .= v[end-1, :, k]
        elseif location == :front
            v[i, j, 1] = v[i, j, 2]
        elseif location == :back
            v[i, j, end] = v[i, j, end-1]
        end
    elseif condition.type == Inlet
        val = condition.value isa Function ? condition.value(t) : (condition.value !== nothing ? condition.value : 0.0)
        v[i, j, k] = val
    elseif condition.type == Outlet
        if location == :top
            v[i, end, k] = v[i, end-1, k]
        end
    end
end

function apply_w_boundary!(w::Array{T,3}, condition::BoundaryCondition, 
                           i, j, k, t::Float64, location::Symbol) where T
    if condition.type == NoSlip
        w[i, j, k] = 0.0
    elseif condition.type == FreeSlip
        if location == :front
            w[i, j, 1] = w[i, j, 2]
        elseif location == :back
            w[i, j, end] = w[i, j, end-1]
        elseif location == :left
            w[1, j, k] = w[2, j, k]
        elseif location == :right
            w[end, j, k] = w[end-1, j, k]
        elseif location == :bottom
            w[i, 1, k] = w[i, 2, k]
        elseif location == :top
            w[i, end, k] = w[i, end-1, k]
        end
    elseif condition.type == Inlet
        val = condition.value isa Function ? condition.value(t) : (condition.value !== nothing ? condition.value : 0.0)
        w[i, j, k] = val
    elseif condition.type == Outlet
        if location == :back
            w[i, j, end] = w[i, j, end-1]
        end
    end
end

# Convenience constructor functions for boundary conditions
function InletBC(direction::Symbol, location::Symbol, value::Union{Float64, Function})
    BoundaryCondition(Inlet, value, direction, location)
end

# Simplified constructors for API compatibility
function InletBC(u_inlet::Float64, v_inlet::Float64=0.0)
    # For 2D: inlet in x-direction
    BoundaryCondition(Inlet, u_inlet, :x, :left)
end

function InletBC(u_inlet::Float64, v_inlet::Float64, w_inlet::Float64)
    # For 3D: inlet in x-direction  
    BoundaryCondition(Inlet, u_inlet, :x, :left)
end

function PressureOutletBC(direction::Symbol, location::Symbol, value::Float64=0.0)
    BoundaryCondition(Outlet, value, direction, location)
end

function PressureOutletBC(pressure::Float64=0.0)
    # Simplified constructor for outlet in x-direction
    BoundaryCondition(Outlet, pressure, :x, :right)
end

function VelocityOutletBC(direction::Symbol, location::Symbol, value::Union{Float64, Function}=0.0)
    BoundaryCondition(Outlet, value, direction, location)
end

function VelocityOutletBC(u_outlet::Float64, v_outlet::Float64=0.0)
    # For 2D: outlet in x-direction
    BoundaryCondition(Outlet, u_outlet, :x, :right)
end

function VelocityOutletBC(u_outlet::Float64, v_outlet::Float64, w_outlet::Float64)
    # For 3D: outlet in x-direction
    BoundaryCondition(Outlet, u_outlet, :x, :right)
end

function NoSlipBC(direction::Symbol, location::Symbol)
    BoundaryCondition(NoSlip, nothing, direction, location)
end

function NoSlipBC()
    # Generic no-slip BC
    BoundaryCondition(NoSlip, nothing, :z, :bottom)  # Default for walls
end

function FreeSlipBC(direction::Symbol, location::Symbol)
    BoundaryCondition(FreeSlip, nothing, direction, location)
end

function FreeSlipBC()
    # Generic free-slip BC
    BoundaryCondition(FreeSlip, nothing, :z, :bottom)  # Default for walls
end

function PeriodicBC(direction::Symbol, location::Symbol)
    BoundaryCondition(Periodic, nothing, direction, location)
end

function PeriodicBC()
    # Generic periodic BC  
    BoundaryCondition(Periodic, nothing, :y, :bottom)  # Default for y-direction
end

# Convenience constructors for 2D boundary conditions (XZ plane)
function BoundaryConditions2D(;
    left::Union{BoundaryCondition, Nothing}=nothing,   # x-direction left
    right::Union{BoundaryCondition, Nothing}=nothing,  # x-direction right
    bottom::Union{BoundaryCondition, Nothing}=nothing, # z-direction bottom
    top::Union{BoundaryCondition, Nothing}=nothing)    # z-direction top
    
    bc = BoundaryConditions()
    
    left !== nothing && add_boundary!(bc, left)
    right !== nothing && add_boundary!(bc, right)
    bottom !== nothing && add_boundary!(bc, bottom)
    top !== nothing && add_boundary!(bc, top)
    
    return bc
end


# Convenience constructors for 3D boundary conditions
function BoundaryConditions3D(;
    left::Union{BoundaryCondition, Nothing}=nothing,   # x-direction
    right::Union{BoundaryCondition, Nothing}=nothing,  # x-direction
    bottom::Union{BoundaryCondition, Nothing}=nothing, # y-direction
    top::Union{BoundaryCondition, Nothing}=nothing,    # y-direction
    front::Union{BoundaryCondition, Nothing}=nothing,  # z-direction
    back::Union{BoundaryCondition, Nothing}=nothing)   # z-direction
    
    bc = BoundaryConditions()
    
    left !== nothing && add_boundary!(bc, left)
    right !== nothing && add_boundary!(bc, right)
    bottom !== nothing && add_boundary!(bc, bottom)
    top !== nothing && add_boundary!(bc, top)
    front !== nothing && add_boundary!(bc, front)
    back !== nothing && add_boundary!(bc, back)
    
    return bc
end

# Simplified positional constructor for BoundaryConditions3D (API compatibility)
function BoundaryConditions3D(inlet::BoundaryCondition, outlet::BoundaryCondition, 
                             y_minus::BoundaryCondition, y_plus::BoundaryCondition, 
                             z_minus::BoundaryCondition, z_plus::BoundaryCondition)
    bc = BoundaryConditions()
    
    # Set proper direction and location for each boundary
    inlet_fixed = BoundaryCondition(inlet.type, inlet.value, :x, :left)
    outlet_fixed = BoundaryCondition(outlet.type, outlet.value, :x, :right)
    y_minus_fixed = BoundaryCondition(y_minus.type, y_minus.value, :y, :bottom)  
    y_plus_fixed = BoundaryCondition(y_plus.type, y_plus.value, :y, :top)
    z_minus_fixed = BoundaryCondition(z_minus.type, z_minus.value, :z, :front)
    z_plus_fixed = BoundaryCondition(z_plus.type, z_plus.value, :z, :back)
    
    add_boundary!(bc, inlet_fixed)
    add_boundary!(bc, outlet_fixed)
    add_boundary!(bc, y_minus_fixed)
    add_boundary!(bc, y_plus_fixed)
    add_boundary!(bc, z_minus_fixed)
    add_boundary!(bc, z_plus_fixed)
    
    return bc
end
