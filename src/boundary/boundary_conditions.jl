struct BoundaryCondition <: AbstractBoundaryCondition
    type::BoundaryType
    value::Union{Float64, Function, Nothing}
    direction::Symbol  # :x, :y, :z
    location::Symbol   # :left, :right, :bottom, :top, :front, :back
end

function BoundaryCondition(type::BoundaryType, direction::Symbol, location::Symbol; value=nothing)
    BoundaryCondition(type, value, direction, location)
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
        apply_w_boundary!(state.w, condition, 1, :, t, :left)  # w instead of v
    end
    
    # Right boundary (x=Lx)
    if haskey(bc.conditions, (:x, :right))
        condition = bc.conditions[(:x, :right)]
        apply_u_boundary!(state.u, condition, nx+1, :, t, :right)
        apply_w_boundary!(state.w, condition, nx, :, t, :right)
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
    
    # Implementation for all 6 boundaries in 3D
    # Similar pattern as 2D but with additional z-direction boundaries
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

# Convenience constructor functions for boundary conditions
function InletBC(direction::Symbol, location::Symbol, value::Union{Float64, Function})
    BoundaryCondition(Inlet, value, direction, location)
end

function PressureOutletBC(direction::Symbol, location::Symbol, value::Float64=0.0)
    BoundaryCondition(Outlet, value, direction, location)
end

function VelocityOutletBC(direction::Symbol, location::Symbol, value::Union{Float64, Function}=0.0)
    BoundaryCondition(Outlet, value, direction, location)
end

function NoSlipBC(direction::Symbol, location::Symbol)
    BoundaryCondition(NoSlip, nothing, direction, location)
end

function FreeSlipBC(direction::Symbol, location::Symbol)
    BoundaryCondition(FreeSlip, nothing, direction, location)
end

function PeriodicBC(direction::Symbol, location::Symbol)
    BoundaryCondition(Periodic, nothing, direction, location)
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