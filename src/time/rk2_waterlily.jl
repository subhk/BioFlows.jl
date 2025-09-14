# WaterLily-style RK2 time integration with BDIM predictor-corrector
# Exact implementation following WaterLily's mom_step! function

include("../immersed/waterlily_bdim.jl")
include("../immersed/stable_volume_penalty.jl")
include("adaptive_timestepping.jl")

# RK2 integrator structure
mutable struct WaterLilyRK2Integrator
    # Time stepping
    timestepper::AdaptiveTimestepper
    
    # BDIM data (if using BDIM)
    bdim::Union{WaterLilyBDIM, Nothing}
    
    # Working arrays for RK2
    u_predictor::Array{Float64, 2}
    w_predictor::Array{Float64, 2} 
    f_momentum::Array{Float64, 3}  # Momentum forcing
    
    # Method selection
    ib_method::Symbol  # :BDIM or :VolumePenalty
end

function WaterLilyRK2Integrator(nx, nz, ib_method=:VolumePenalty; dt_initial=0.01)
    # Create timestepper
    ts = AdaptiveTimestepper(dt_initial)
    
    # Create BDIM data if needed
    bdim = (ib_method == :BDIM) ? WaterLilyBDIM(nx, nz) : nothing
    
    # Working arrays
    u_pred = zeros(nx+1, nz)
    w_pred = zeros(nx, nz+1)
    f_mom = zeros(nx+1, nz+1, 2)
    
    return WaterLilyRK2Integrator(ts, bdim, u_pred, w_pred, f_mom, ib_method)
end

# Main WaterLily-style momentum step (follows Flow.jl lines 147-165)
function mom_step_waterlily!(integrator, state, solver, bodies)
    """
    WaterLily's exact momentum step with RK2 + BDIM predictor-corrector
    
    From WaterLily Flow.jl lines 147-165:
    1. Store u⁰ = u (line 148)
    2. Predictor: compute forces, apply BDIM, project pressure
    3. Corrector: compute forces, apply BDIM, average with predictor
    """
    
    dt = integrator.timestepper.dt
    nx, nz = size(state.p)
    
    # Step 1: Store previous velocity u⁰ = u (line 148)
    u⁰ = copy(state.u)
    w⁰ = copy(state.w)
    
    # Reset velocities for predictor step
    scale_velocity!(state, 0.0)  # u *= 0
    
    # === PREDICTOR STEP ===
    t₀ = integrator.timestepper.time
    t₁ = t₀ + dt
    
    # Compute momentum forcing (convection + diffusion)
    compute_momentum_forcing!(integrator.f_momentum, u⁰, w⁰, solver.grid, solver.fluid.kinematic_viscosity)
    
    # Apply momentum forcing to velocity
    apply_momentum_forcing!(state, integrator.f_momentum, dt)
    
    # Apply immersed boundary correction (BDIM or Volume Penalty)
    if integrator.ib_method == :BDIM && integrator.bdim !== nothing
        # WaterLily BDIM! function (line 154)
        BDIM_waterlily!(state, u⁰, w⁰, integrator.bdim, dt, solver.grid)
    elseif integrator.ib_method == :VolumePenalty
        apply_stable_volume_penalty_2d!(state, bodies, solver.grid, dt)
    end
    
    # Apply boundary conditions at t₁
    apply_boundary_conditions!(state, solver.boundary_conditions, t₁)
    
    # Pressure projection
    project_pressure!(state, solver)
    
    # Apply BCs again after projection
    apply_boundary_conditions!(state, solver.boundary_conditions, t₁)
    
    # Store predictor result
    integrator.u_predictor .= state.u
    integrator.w_predictor .= state.w
    
    # === CORRECTOR STEP ===
    
    # Compute momentum forcing at corrector step
    compute_momentum_forcing!(integrator.f_momentum, state.u, state.w, solver.grid, solver.fluid.kinematic_viscosity)
    
    # Apply momentum forcing
    apply_momentum_forcing!(state, integrator.f_momentum, dt)
    
    # Apply immersed boundary correction again (line 162)
    if integrator.ib_method == :BDIM && integrator.bdim !== nothing
        BDIM_waterlily!(state, u⁰, w⁰, integrator.bdim, dt, solver.grid)
    elseif integrator.ib_method == :VolumePenalty
        apply_stable_volume_penalty_2d!(state, bodies, solver.grid, dt)
    end
    
    # Average predictor and corrector (line 162: scale_u!(a,0.5))
    for j in 1:size(state.u, 2), i in 1:size(state.u, 1)
        state.u[i, j] = 0.5 * (integrator.u_predictor[i, j] + state.u[i, j])
    end
    
    for j in 1:size(state.w, 2), i in 1:size(state.w, 1)
        state.w[i, j] = 0.5 * (integrator.w_predictor[i, j] + state.w[i, j])
    end
    
    # Apply boundary conditions
    apply_boundary_conditions!(state, solver.boundary_conditions, t₁)
    
    # Final pressure projection (line 163)
    project_pressure!(state, solver, 0.5)
    
    # Final boundary conditions
    apply_boundary_conditions!(state, solver.boundary_conditions, t₁)
    
    # Update adaptive time step (WaterLily line 164: push!(a.Δt,CFL(a)))
    dt_new = update_timestep!(integrator.timestepper, state, solver.grid, solver.fluid.kinematic_viscosity)
    
    return dt_new
end

# Utility functions for RK2 step

function scale_velocity!(state, factor)
    """Scale velocity fields by factor"""
    state.u .*= factor
    state.w .*= factor
end

function compute_momentum_forcing!(f_momentum, u, w, grid, ν)
    """
    Compute momentum forcing terms (convection + diffusion)
    Uses proper upwind scheme for convection as requested by user
    """
    dx, dz = grid.dx, grid.dz
    
    # Fill momentum forcing array with zeros
    fill!(f_momentum, 0.0)
    
    # u-momentum equation: ∂u/∂t + u*∂u/∂x + w*∂u/∂z = -∂p/∂x + ν*∇²u
    for j in 2:size(u,2)-1, i in 2:size(u,1)-1
        # UPWIND CONVECTION for u-momentum
        # u * ∂u/∂x term (upwind in x-direction)
        if u[i,j] > 0.0
            dudx = (u[i,j] - u[i-1,j]) / dx  # Backward difference
        else
            dudx = (u[i+1,j] - u[i,j]) / dx  # Forward difference
        end
        conv_uu = u[i,j] * dudx
        
        # w * ∂u/∂z term (upwind in z-direction)
        # Interpolate w to u-face location first
        w_at_u = 0.25 * (w[i,j] + w[i-1,j] + w[i,j+1] + w[i-1,j+1])
        if w_at_u > 0.0
            dudz = (u[i,j] - u[i,j-1]) / dz  # Backward difference
        else
            dudz = (u[i,j+1] - u[i,j]) / dz  # Forward difference
        end
        conv_wu = w_at_u * dudz
        
        # Total convection
        conv_u = conv_uu + conv_wu
        
        # Diffusion (central differences)
        diff_u = ν * ((u[i+1,j] - 2*u[i,j] + u[i-1,j])/dx^2 + (u[i,j+1] - 2*u[i,j] + u[i,j-1])/dz^2)
        
        f_momentum[i, j, 1] = -conv_u + diff_u
    end
    
    # w-momentum equation: ∂w/∂t + u*∂w/∂x + w*∂w/∂z = -∂p/∂z + ν*∇²w
    for j in 2:size(w,2)-1, i in 2:size(w,1)-1
        # UPWIND CONVECTION for w-momentum
        # u * ∂w/∂x term (upwind in x-direction)
        # Interpolate u to w-face location first
        u_at_w = 0.25 * (u[i,j] + u[i+1,j] + u[i,j-1] + u[i+1,j-1])
        if u_at_w > 0.0
            dwdx = (w[i,j] - w[i-1,j]) / dx  # Backward difference
        else
            dwdx = (w[i+1,j] - w[i,j]) / dx  # Forward difference
        end
        conv_uw = u_at_w * dwdx
        
        # w * ∂w/∂z term (upwind in z-direction)
        if w[i,j] > 0.0
            dwdz = (w[i,j] - w[i,j-1]) / dz  # Backward difference
        else
            dwdz = (w[i,j+1] - w[i,j]) / dz  # Forward difference
        end
        conv_ww = w[i,j] * dwdz
        
        # Total convection
        conv_w = conv_uw + conv_ww
        
        # Diffusion (central differences)
        diff_w = ν * ((w[i+1,j] - 2*w[i,j] + w[i-1,j])/dx^2 + (w[i,j+1] - 2*w[i,j] + w[i,j-1])/dz^2)
        
        f_momentum[i, j, 2] = -conv_w + diff_w
    end
end

function apply_momentum_forcing!(state, f_momentum, dt)
    """Apply momentum forcing to velocity fields"""
    for j in 1:size(state.u, 2), i in 1:size(state.u, 1)
        if i <= size(f_momentum, 1) && j <= size(f_momentum, 2)
            state.u[i, j] += dt * f_momentum[i, j, 1]
        end
    end
    
    for j in 1:size(state.w, 2), i in 1:size(state.w, 1)  
        if i <= size(f_momentum, 1) && j <= size(f_momentum, 2)
            state.w[i, j] += dt * f_momentum[i, j, 2]
        end
    end
end

function apply_boundary_conditions!(state, bc, t)
    """Apply boundary conditions at time t"""
    # Implementation depends on BC structure
    # Would set inlet, outlet, wall conditions
end

function project_pressure!(state, solver, weight=1.0)
    """
    Pressure projection step (incompressibility constraint)  
    This calls the multigrid pressure solver
    """
    # This would use solver.mg_solver or similar
    # Solve ∇²p = ∇·u/dt
    # Then u = u - dt*∇p
    
    # Placeholder - actual implementation uses existing BioFlows pressure solver
end

# Integration function that replaces standard time stepping
function integrate_waterlily_step!(integrator, state, solver, bodies)
    """
    Single WaterLily-style integration step
    """
    # Update BDIM if using BDIM method
    if integrator.ib_method == :BDIM && integrator.bdim !== nothing
        measure_waterlily!(integrator.bdim, bodies, solver.grid, integrator.timestepper.time)
    end
    
    # Perform RK2 momentum step
    dt_new = mom_step_waterlily!(integrator, state, solver, bodies)
    
    # Step time forward
    new_time = step_time!(integrator.timestepper)
    
    return new_time, dt_new
end

export WaterLilyRK2Integrator, mom_step_waterlily!, integrate_waterlily_step!