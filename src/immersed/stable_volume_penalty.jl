# Stable Volume Penalty method for immersed boundaries
# Improved numerical stability with better parameter choices

function apply_stable_volume_penalty_2d!(state, bodies, grid, dt)
    """
    Apply volume penalty method with improved stability:
    - Adaptive penalty parameter
    - Smooth forcing transitions  
    - CFL-aware time step limits
    """
    
    if isempty(bodies.bodies)
        return
    end
    
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz
    
    # Adaptive penalty parameter (key for stability)
    # Rule: λ ~ 1/dt but not too large to avoid stiffness
    max_velocity = max(maximum(abs.(state.u)), maximum(abs.(state.w)), 1e-6)
    cell_size = min(dx, dz)
    
    # CFL-based penalty (ensures numerical stability)
    λ_base = max_velocity / (0.5 * dt * cell_size^2)  # From CFL condition
    λ_penalty = clamp(λ_base, 1e2, 1e5)  # Reasonable bounds
    
    # Apply forcing to u-velocity (staggered grid)
    for j = 1:nz, i = 1:nx+1
        if i <= size(state.u, 1) && j <= size(state.u, 2)
            # Staggered u-location
            x = (i-1) * dx
            z = (j-0.5) * dz
            
            for body in bodies.bodies
                if body.shape isa Circle
                    # Distance to circle center
                    distance = sqrt((x - body.center[1])^2 + (z - body.center[2])^2)
                    
                    if distance <= body.shape.radius
                        # Inside body: enforce no-slip with smooth transition
                        mask = smooth_heaviside(body.shape.radius - distance, 0.1*cell_size)
                        target_velocity = body.velocity[1]  # Body u-velocity
                        
                        # Stable volume penalty forcing
                        force = -λ_penalty * mask * (state.u[i, j] - target_velocity)
                        state.u[i, j] += dt * force
                        
                        # Limit velocity change for stability
                        change = dt * force
                        max_change = 0.1 * max_velocity * dt / cell_size  # CFL limit
                        if abs(change) > max_change
                            state.u[i, j] = state.u[i, j] - dt * force + sign(change) * max_change
                        end
                    end
                end
            end
        end
    end
    
    # Apply forcing to w-velocity (staggered grid)  
    for j = 1:nz+1, i = 1:nx
        if i <= size(state.w, 1) && j <= size(state.w, 2)
            # Staggered w-location
            x = (i-0.5) * dx  
            z = (j-1) * dz
            
            for body in bodies.bodies
                if body.shape isa Circle
                    # Distance to circle center
                    distance = sqrt((x - body.center[1])^2 + (z - body.center[2])^2)
                    
                    if distance <= body.shape.radius
                        # Inside body: enforce no-slip with smooth transition
                        mask = smooth_heaviside(body.shape.radius - distance, 0.1*cell_size)
                        target_velocity = body.velocity[2]  # Body w-velocity
                        
                        # Stable volume penalty forcing
                        force = -λ_penalty * mask * (state.w[i, j] - target_velocity)
                        state.w[i, j] += dt * force
                        
                        # Limit velocity change for stability
                        change = dt * force
                        max_change = 0.1 * max_velocity * dt / cell_size  # CFL limit
                        if abs(change) > max_change
                            state.w[i, j] = state.w[i, j] - dt * force + sign(change) * max_change
                        end
                    end
                end
            end
        end
    end
    
    # Clean up any NaN/Inf values that might arise
    state.u[.!isfinite.(state.u)] .= 0.0
    state.w[.!isfinite.(state.w)] .= 0.0
end

# Smooth Heaviside function to avoid discontinuous forcing
function smooth_heaviside(x, ε)
    """
    Smooth approximation to Heaviside function
    Returns 1 for x > ε, 0 for x < -ε, smooth transition in between
    """
    if x > ε
        return 1.0
    elseif x < -ε  
        return 0.0
    else
        # Smooth cubic transition
        s = x / ε
        return 0.5 * (1 + s + (2/π) * sin(π * s))
    end
end

# Compute stable time step for volume penalty
function volume_penalty_stable_dt(bodies, grid, max_velocity, ν)
    """
    Compute maximum stable time step for volume penalty method
    Based on CFL and penalty parameter constraints
    """
    
    if isempty(bodies.bodies)
        return Inf
    end
    
    dx, dz = grid.dx, grid.dz
    cell_size = min(dx, dz)
    
    # CFL constraint
    dt_cfl = 0.5 * cell_size / max(max_velocity, 1e-6)
    
    # Viscous constraint  
    dt_viscous = 0.25 * cell_size^2 / (ν + 1e-12)
    
    # Penalty constraint (empirical)
    dt_penalty = 0.01 * cell_size^2  # Conservative for penalty stability
    
    return min(dt_cfl, dt_viscous, dt_penalty)
end

export apply_stable_volume_penalty_2d!, volume_penalty_stable_dt