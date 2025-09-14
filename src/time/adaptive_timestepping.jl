# Adaptive time stepping with CFL conditions
# Following WaterLily.jl's approach for stability

# Adaptive time stepper structure
mutable struct AdaptiveTimestepper
    dt::Float64           # Current time step
    dt_max::Float64       # Maximum allowed time step
    dt_min::Float64       # Minimum allowed time step  
    time::Float64         # Current time
    cfl_target::Float64   # Target CFL number (< 1 for stability)
    safety_factor::Float64 # Safety factor for dt adjustment
    
    # History for stability
    dt_history::Vector{Float64}
    max_history_length::Int
end

function AdaptiveTimestepper(dt_initial=0.01; dt_max=0.1, dt_min=1e-6, 
                           cfl_target=0.5, safety_factor=0.9)
    return AdaptiveTimestepper(
        dt_initial, dt_max, dt_min, 0.0, 
        cfl_target, safety_factor,
        [dt_initial], 10
    )
end

# WaterLily-style CFL calculation
function compute_cfl(state, grid, ν; method=:waterlily)
    """
    Compute CFL number following WaterLily's approach
    """
    if method == :waterlily
        return compute_cfl_waterlily(state, grid, ν)
    else
        return compute_cfl_standard(state, grid, ν)
    end
end

function compute_cfl_waterlily(state, grid, ν)
    """
    WaterLily's CFL calculation: flux_out + 5ν diffusion term
    From Flow.jl lines 168-171
    """
    dx, dz = grid.dx, grid.dz
    max_flux = 0.0
    
    # Compute maximum flux out (convective CFL)
    nx, nz = size(state.u, 1)-1, size(state.u, 2)
    
    for j = 1:nz, i = 1:nx
        # Flux out of cell (i,j) 
        flux_x = abs(state.u[i+1, j] - state.u[i, j]) / dx
        
        # Handle w-velocity bounds
        if j < size(state.w, 2)
            flux_z = abs(state.w[i, j+1] - state.w[i, j]) / dz
        else
            flux_z = 0.0
        end
        
        total_flux = flux_x + flux_z
        max_flux = max(max_flux, total_flux)
    end
    
    # Add diffusion term (WaterLily uses 5ν factor)
    diffusion_term = 5 * ν * (1/dx^2 + 1/dz^2)
    
    return max_flux + diffusion_term
end

function compute_cfl_standard(state, grid, ν)
    """
    Standard CFL calculation
    """
    dx, dz = grid.dx, grid.dz
    
    # Maximum velocities
    u_max = maximum(abs.(state.u))
    w_max = maximum(abs.(state.w))
    
    # Convective CFL
    cfl_conv = u_max/dx + w_max/dz
    
    # Viscous CFL  
    cfl_visc = 2*ν*(1/dx^2 + 1/dz^2)
    
    return cfl_conv + cfl_visc
end

# Adaptive time step update
function update_timestep!(ts::AdaptiveTimestepper, state, grid, ν)
    """
    Update time step based on CFL condition and stability
    """
    # Compute current CFL
    cfl_current = compute_cfl(state, grid, ν)
    
    # Compute suggested time step
    if cfl_current > 1e-12
        dt_suggested = ts.cfl_target / cfl_current
    else
        dt_suggested = ts.dt_max
    end
    
    # Apply safety factor
    dt_new = ts.safety_factor * dt_suggested
    
    # Enforce bounds
    dt_new = clamp(dt_new, ts.dt_min, ts.dt_max)
    
    # Smooth time step changes (avoid oscillations)
    if length(ts.dt_history) > 1
        dt_prev = ts.dt_history[end]
        # Limit change rate
        max_increase = 1.2 * dt_prev
        max_decrease = 0.8 * dt_prev
        dt_new = clamp(dt_new, max_decrease, max_increase)
    end
    
    # Update time stepper
    ts.dt = dt_new
    
    # Update history
    push!(ts.dt_history, dt_new)
    if length(ts.dt_history) > ts.max_history_length
        popfirst!(ts.dt_history)
    end
    
    return dt_new
end

# Step forward in time
function step_time!(ts::AdaptiveTimestepper)
    """
    Advance time by current time step
    """
    ts.time += ts.dt
    return ts.time
end

# Get stable time step for immersed boundaries
function stable_dt_immersed(state, grid, bodies, ν, method::Symbol)
    """
    Compute stable time step considering immersed boundary constraints
    """
    # Base CFL constraint
    cfl = compute_cfl(state, grid, ν)
    dt_cfl = 0.5 / max(cfl, 1e-12)
    
    # Method-specific constraints
    if method == :BDIM
        dt_bdim = stable_dt_bdim(bodies, grid, state)
        return min(dt_cfl, dt_bdim)
    elseif method == :VolumePenalty
        u_max = max(maximum(abs.(state.u)), maximum(abs.(state.w)))
        dt_vp = volume_penalty_stable_dt(bodies, grid, u_max, ν)
        return min(dt_cfl, dt_vp)
    else
        return dt_cfl
    end
end

function stable_dt_bdim(bodies, grid, state)
    """
    BDIM-specific time step constraint
    """
    if isempty(bodies.bodies)
        return Inf
    end
    
    dx, dz = grid.dx, grid.dz
    cell_size = min(dx, dz)
    
    # Conservative estimate for BDIM stability
    return 0.1 * cell_size
end

# RK2 time integration with adaptive stepping
function rk2_adaptive_step!(state, state_old, solver, ts::AdaptiveTimestepper)
    """
    Runge-Kutta 2nd order step with adaptive time stepping
    """
    dt = ts.dt
    
    # Store initial state
    copy_state!(state_old, state)
    
    # RK2 Predictor step (half time step)  
    # This would call the momentum equations, pressure solve, etc.
    # (Implementation depends on solver structure)
    
    # RK2 Corrector step (full time step)
    # Average of predictor and corrector
    
    # Update time
    step_time!(ts)
    
    return ts.time
end

# Utility function to copy state
function copy_state!(dest, src)
    dest.u .= src.u
    dest.w .= src.w  
    dest.p .= src.p
end

export AdaptiveTimestepper, update_timestep!, step_time!, stable_dt_immersed, 
       compute_cfl, rk2_adaptive_step!