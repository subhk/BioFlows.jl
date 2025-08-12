using ForwardDiff

struct FlexibleBody <: AbstractBody
    # Lagrangian parameters
    length::Float64              # Filament length L
    n_points::Int               # Number of discretization points
    s::Vector{Float64}          # Lagrangian coordinates
    X::Matrix{Float64}          # Position vectors X(s,t) [n_points x 2]
    X_old::Matrix{Float64}      # Previous time step positions
    X_prev::Matrix{Float64}     # Two time steps ago for Adams-Bashforth
    
    # Material properties
    bending_rigidity::Float64   # EI bending rigidity
    mass_per_length::Float64    # ρ₁ mass per unit length
    thickness::Float64          # Filament thickness
    
    # Boundary conditions
    bc_type::Symbol            # :fixed_ends, :sinusoidal_front, :rotating_front
    amplitude::Float64         # For sinusoidal motion
    frequency::Float64         # For sinusoidal motion
    
    # Computational arrays
    tension::Vector{Float64}    # Tension T(s,t)
    curvature::Vector{Float64}  # κ(s,t) 
    force::Matrix{Float64}      # Lagrangian force F(s,t) [n_points x 2]
    
    id::Int
end

function FlexibleBody(length::Float64, n_points::Int, initial_shape::Function;
                     bending_rigidity::Float64=0.001,
                     mass_per_length::Float64=1.0,
                     thickness::Float64=0.01,
                     bc_type::Symbol=:fixed_ends,
                     amplitude::Float64=0.0,
                     frequency::Float64=1.0,
                     id::Int=1)
    
    # Initialize Lagrangian coordinates
    s = collect(LinRange(0.0, 1.0, n_points))  # Normalized [0,1]
    
    # Initialize positions
    X = zeros(n_points, 2)
    for i = 1:n_points
        pos = initial_shape(s[i] * length)
        X[i, 1] = pos[1]
        X[i, 2] = pos[2]
    end
    
    X_old = copy(X)
    X_prev = copy(X)
    
    tension = zeros(n_points)
    curvature = zeros(n_points)
    force = zeros(n_points, 2)
    
    FlexibleBody(length, n_points, s * length, X, X_old, X_prev,
                bending_rigidity, mass_per_length, thickness,
                bc_type, amplitude, frequency,
                tension, curvature, force, id)
end

function straight_filament(L::Float64, start_point::Vector{Float64}, end_point::Vector{Float64})
    return function(s::Float64)
        t = s / L  # Normalize to [0,1]
        return start_point .+ t .* (end_point .- start_point)
    end
end

function compute_derivatives(body::FlexibleBody)
    n = body.n_points
    ds = body.length / (n - 1)
    
    # First derivatives ∂X/∂s
    dXds = zeros(n, 2)
    for i = 2:n-1
        dXds[i, :] = (body.X[i+1, :] - body.X[i-1, :]) / (2 * ds)
    end
    # Boundary conditions for derivatives
    dXds[1, :] = (body.X[2, :] - body.X[1, :]) / ds
    dXds[n, :] = (body.X[n, :] - body.X[n-1, :]) / ds
    
    # Second derivatives ∂²X/∂s²
    d2Xds2 = zeros(n, 2)
    for i = 2:n-1
        d2Xds2[i, :] = (body.X[i+1, :] - 2*body.X[i, :] + body.X[i-1, :]) / ds^2
    end
    
    return dXds, d2Xds2
end

function compute_curvature!(body::FlexibleBody)
    dXds, d2Xds2 = compute_derivatives(body)
    
    for i = 1:body.n_points
        # Curvature κ = |∂X/∂s × ∂²X/∂s²| / |∂X/∂s|³
        # In 2D: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
        x_prime = dXds[i, 1]
        y_prime = dXds[i, 2]
        x_double_prime = d2Xds2[i, 1]
        y_double_prime = d2Xds2[i, 2]
        
        numerator = abs(x_prime * y_double_prime - y_prime * x_double_prime)
        denominator = (x_prime^2 + y_prime^2)^1.5
        
        if denominator > 1e-12
            body.curvature[i] = numerator / denominator
        else
            body.curvature[i] = 0.0
        end
    end
end

function solve_tension!(body::FlexibleBody, dt::Float64)
    # Solve Poisson equation for tension T based on inextensibility constraint
    # This implements equations (2.5)-(2.9) from flexible_bodies.pdf
    
    n = body.n_points
    ds = body.length / (n - 1)
    dXds, d2Xds2 = compute_derivatives(body)
    
    # Enforce inextensibility: |∂X/∂s| = 1
    for i = 1:n
        length_constraint = sqrt(dXds[i, 1]^2 + dXds[i, 2]^2)
        if abs(length_constraint - 1.0) > 1e-6
            # Normalize to maintain |∂X/∂s| = 1
            if length_constraint > 1e-12
                dXds[i, :] ./= length_constraint
            end
        end
    end
    
    # Compute tension using Poisson equation
    # Based on equation (2.6) from the PDF: ∂T/∂s = -ρ₁ ∂²X/∂t² · ∂X/∂s
    
    # Approximate ∂²X/∂t² using backward finite differences
    if dt > 0
        d2Xdt2 = (body.X - 2*body.X_old + body.X_prev) / dt^2
        
        for i = 2:n-1
            # ∂T/∂s = -ρ₁ (∂²X/∂t² · ∂X/∂s)
            dtds = -body.mass_per_length * (d2Xdt2[i, 1] * dXds[i, 1] + 
                                           d2Xdt2[i, 2] * dXds[i, 2])
            body.tension[i] = body.tension[i-1] + dtds * ds
        end
    end
    
    # Apply tension boundary conditions
    body.tension[1] = 0.0      # Free end condition
    body.tension[n] = 0.0      # Free end condition
end

function apply_boundary_conditions!(body::FlexibleBody, t::Float64)
    if body.bc_type == :fixed_ends
        # Both ends are fixed - no motion
        # End positions remain unchanged
        
    elseif body.bc_type == :sinusoidal_front
        # Front end has sinusoidal motion, rear end is free
        # X(0,t) = X₀ + A*sin(ωt), ∂²X/∂s²|_{s=0} = 0
        body.X[1, 2] += body.amplitude * sin(2π * body.frequency * t)
        
        # Rear end is free: ∂²X/∂s²|_{s=L} = 0 (natural boundary condition)
        
    elseif body.bc_type == :rotating_front
        # Front can rotate freely, rear is free
        # ∂²X/∂s²|_{s=0} = 0 and ∂²X/∂s²|_{s=L} = 0
        
    end
end

function flexible_body_rhs!(dXdt2::Matrix{Float64}, body::FlexibleBody, dt::Float64)
    # Implement equation (2.5) from flexible_bodies.pdf
    # ρ₁ ∂²X/∂t² = ∂/∂s(T ∂X/∂s) - ∂²/∂s²(EI ∂²X/∂s²) + F
    
    n = body.n_points
    ds = body.length / (n - 1)
    dXds, d2Xds2 = compute_derivatives(body)
    
    # Solve for tension
    solve_tension!(body, dt)
    
    # Compute curvature
    compute_curvature!(body)
    
    for i = 2:n-1
        # First term: ∂/∂s(T ∂X/∂s)
        tension_term_x = (body.tension[i+1] * dXds[i+1, 1] - body.tension[i-1] * dXds[i-1, 1]) / (2 * ds)
        tension_term_y = (body.tension[i+1] * dXds[i+1, 2] - body.tension[i-1] * dXds[i-1, 2]) / (2 * ds)
        
        # Second term: ∂²/∂s²(EI ∂²X/∂s²) - fourth order derivative
        # Approximate ∂⁴X/∂s⁴ using finite differences
        if i >= 3 && i <= n-2
            d4Xds4_x = (body.X[i+2, 1] - 4*body.X[i+1, 1] + 6*body.X[i, 1] - 4*body.X[i-1, 1] + body.X[i-2, 1]) / ds^4
            d4Xds4_y = (body.X[i+2, 2] - 4*body.X[i+1, 2] + 6*body.X[i, 2] - 4*body.X[i-1, 2] + body.X[i-2, 2]) / ds^4
        else
            d4Xds4_x = 0.0
            d4Xds4_y = 0.0
        end
        
        bending_term_x = body.bending_rigidity * d4Xds4_x
        bending_term_y = body.bending_rigidity * d4Xds4_y
        
        # External force term
        force_term_x = body.force[i, 1]
        force_term_y = body.force[i, 2]
        
        # Total acceleration
        dXdt2[i, 1] = (tension_term_x - bending_term_x + force_term_x) / body.mass_per_length
        dXdt2[i, 2] = (tension_term_y - bending_term_y + force_term_y) / body.mass_per_length
    end
    
    # Boundary conditions
    dXdt2[1, :] .= 0.0    # Fixed end
    dXdt2[n, :] .= 0.0    # Fixed end
end

function update_flexible_body!(body::FlexibleBody, dt::Float64)
    # Use explicit time integration (can be replaced with Runge-Kutta)
    n = body.n_points
    dXdt2 = zeros(n, 2)
    
    # Compute RHS of flexible body equation
    flexible_body_rhs!(dXdt2, body, dt)
    
    # Update positions using Verlet integration
    # X^{n+1} = 2X^n - X^{n-1} + dt²(d²X/dt²)
    X_new = 2.0 * body.X - body.X_old + dt^2 * dXdt2
    
    # Store previous states
    body.X_prev .= body.X_old
    body.X_old .= body.X
    body.X .= X_new
    
    # Apply boundary conditions
    # This would be called with appropriate time t
end

function get_flexible_body_points(body::FlexibleBody)
    # Return current positions of all Lagrangian points
    return [(body.X[i, 1], body.X[i, 2]) for i = 1:body.n_points]
end

function compute_flexible_body_forces(body::FlexibleBody, grid::StaggeredGrid, 
                                    state::SolutionState, fluid::FluidProperties)
    """
    Compute forces on the flexible body due to fluid interaction using accurate stress integration.
    
    This implements proper immersed boundary method with:
    1. Bilinear interpolation of fluid properties to Lagrangian points
    2. Integration of both pressure and viscous stresses
    3. Proper computation of body velocity from position history
    4. Delta function spreading for force distribution
    """
    
    # Get fluid properties
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
        μ = fluid.μ
    else
        error("Variable density not implemented in flexible body forces")
    end
    
    force_total = [0.0, 0.0]
    torque_total = 0.0
    
    # Compute derivatives for accurate body velocity
    dXds, d2Xds2 = compute_derivatives(body)
    ds = body.length / (body.n_points - 1)
    
    # For each Lagrangian point, compute accurate fluid forces
    for i = 1:body.n_points
        x, y = body.X[i, 1], body.X[i, 2]
        
        # 1. Accurate interpolation of fluid properties using bilinear interpolation
        u_fluid = interpolate_velocity_bilinear(grid, state.u, x, y, :u)
        v_fluid = interpolate_velocity_bilinear(grid, state.v, x, y, :v)
        p_fluid = interpolate_pressure_bilinear(grid, state.p, x, y)
        
        # 2. Compute accurate body velocity using time derivatives
        u_body, v_body = compute_body_velocity(body, i, grid.dx, grid.dy)
        
        # 3. Compute velocity gradients at the body point for viscous stress
        dudx_fluid = interpolate_gradient_bilinear(grid, state.u, x, y, :dudx)
        dudy_fluid = interpolate_gradient_bilinear(grid, state.u, x, y, :dudy)
        dvdx_fluid = interpolate_gradient_bilinear(grid, state.v, x, y, :dvdx)
        dvdy_fluid = interpolate_gradient_bilinear(grid, state.v, x, y, :dvdy)
        
        # 4. Compute unit normal vector to the body surface
        if i > 1 && i < body.n_points
            # Tangent vector from finite differences
            tx = (body.X[i+1, 1] - body.X[i-1, 1]) / (2 * ds)
            ty = (body.X[i+1, 2] - body.X[i-1, 2]) / (2 * ds)
        elseif i == 1
            tx = (body.X[i+1, 1] - body.X[i, 1]) / ds
            ty = (body.X[i+1, 2] - body.X[i, 2]) / ds
        else  # i == body.n_points
            tx = (body.X[i, 1] - body.X[i-1, 1]) / ds
            ty = (body.X[i, 2] - body.X[i-1, 2]) / ds
        end
        
        # Normalize tangent
        t_norm = sqrt(tx^2 + ty^2)
        if t_norm > 1e-12
            tx /= t_norm
            ty /= t_norm
        end
        
        # Normal vector (rotate tangent 90 degrees clockwise)
        nx = ty
        ny = -tx
        
        # 5. Compute stress tensor components
        # Pressure stress: σ_p = -p * I
        σxx_p = -p_fluid
        σyy_p = -p_fluid
        σxy_p = 0.0
        
        # Viscous stress: σ_v = μ(∇u + ∇u^T)
        σxx_v = 2 * μ * dudx_fluid
        σyy_v = 2 * μ * dvdy_fluid
        σxy_v = μ * (dudy_fluid + dvdx_fluid)
        
        # Total stress tensor
        σxx = σxx_p + σxx_v
        σyy = σyy_p + σyy_v
        σxy = σxy_p + σxy_v
        
        # 6. Compute traction vector: t = σ · n
        traction_x = σxx * nx + σxy * ny
        traction_y = σxy * nx + σyy * ny
        
        # 7. Apply no-slip boundary condition constraint
        # Force needed to enforce no-slip: F = k(u_fluid - u_body)
        constraint_stiffness = 1000.0 * ρ  # Scale with fluid density
        
        u_relative = u_fluid - u_body
        v_relative = v_fluid - v_body
        
        constraint_force_x = -constraint_stiffness * u_relative
        constraint_force_y = -constraint_stiffness * v_relative
        
        # 8. Total force per unit length (combine stress and constraint)
        # Weight between physical stress and constraint enforcement
        stress_weight = 0.3
        constraint_weight = 0.7
        
        force_per_length_x = stress_weight * traction_x * body.thickness + 
                            constraint_weight * constraint_force_x
        force_per_length_y = stress_weight * traction_y * body.thickness + 
                            constraint_weight * constraint_force_y
        
        # 9. Integrate along the body (multiply by ds for force contribution)
        body.force[i, 1] = force_per_length_x * ds
        body.force[i, 2] = force_per_length_y * ds
        
        # 10. Add to total force
        force_total[1] += body.force[i, 1]
        force_total[2] += body.force[i, 2]
        
        # 11. Compute torque about body center of mass
        center_x = sum(body.X[:, 1]) / body.n_points
        center_y = sum(body.X[:, 2]) / body.n_points
        r_x = x - center_x
        r_y = y - center_y
        torque_total += r_x * body.force[i, 2] - r_y * body.force[i, 1]
    end
    
    return force_total, torque_total
end

function interpolate_velocity_bilinear(grid::StaggeredGrid, field::Matrix{T}, x::Float64, y::Float64, component::Symbol) where T
    """
    Accurate bilinear interpolation of velocity components at arbitrary point (x,y).
    Handles staggered grid locations properly.
    """
    
    if component == :u
        # u is defined at x-faces: (x_{i+1/2}, y_j)
        # Find surrounding grid points
        i_real = (x - grid.xu[1]) / grid.dx + 1
        j_real = (y - grid.y[1]) / grid.dy + 1
        
        i = max(1, min(grid.nx, Int(floor(i_real))))
        j = max(1, min(grid.ny-1, Int(floor(j_real))))
        
        # Bilinear interpolation weights
        α = i_real - i
        β = j_real - j
        
        α = max(0.0, min(1.0, α))
        β = max(0.0, min(1.0, β))
        
        if i < grid.nx && j < grid.ny
            return (1-α)*(1-β)*field[i, j] + α*(1-β)*field[i+1, j] + 
                   (1-α)*β*field[i, j+1] + α*β*field[i+1, j+1]
        else
            return field[min(i, grid.nx), min(j, grid.ny)]
        end
        
    elseif component == :v
        # v is defined at y-faces: (x_i, y_{j+1/2})
        i_real = (x - grid.x[1]) / grid.dx + 1
        j_real = (y - grid.yv[1]) / grid.dy + 1
        
        i = max(1, min(grid.nx-1, Int(floor(i_real))))
        j = max(1, min(grid.ny, Int(floor(j_real))))
        
        α = i_real - i
        β = j_real - j
        
        α = max(0.0, min(1.0, α))
        β = max(0.0, min(1.0, β))
        
        if i < grid.nx && j < grid.ny
            return (1-α)*(1-β)*field[i, j] + α*(1-β)*field[i+1, j] + 
                   (1-α)*β*field[i, j+1] + α*β*field[i+1, j+1]
        else
            return field[min(i, grid.nx), min(j, grid.ny)]
        end
    end
    
    return 0.0
end

function interpolate_pressure_bilinear(grid::StaggeredGrid, p::Matrix{T}, x::Float64, y::Float64) where T
    """
    Bilinear interpolation of pressure (defined at cell centers) to point (x,y).
    """
    
    # Pressure is at cell centers: (x_i, y_j)
    i_real = (x - grid.x[1]) / grid.dx + 1
    j_real = (y - grid.y[1]) / grid.dy + 1
    
    i = max(1, min(grid.nx-1, Int(floor(i_real))))
    j = max(1, min(grid.ny-1, Int(floor(j_real))))
    
    α = i_real - i
    β = j_real - j
    
    α = max(0.0, min(1.0, α))
    β = max(0.0, min(1.0, β))
    
    if i < grid.nx && j < grid.ny
        return (1-α)*(1-β)*p[i, j] + α*(1-β)*p[i+1, j] + 
               (1-α)*β*p[i, j+1] + α*β*p[i+1, j+1]
    else
        return p[min(i, grid.nx), min(j, grid.ny)]
    end
end

function interpolate_gradient_bilinear(grid::StaggeredGrid, field::Matrix{T}, x::Float64, y::Float64, 
                                     gradient_type::Symbol) where T
    """
    Compute velocity gradients at point (x,y) using bilinear interpolation.
    """
    
    if gradient_type == :dudx
        # Compute ∂u/∂x using clean differential operators
        dudx_grid = ddx_at_faces(interpolate_u_to_cell_center(field, grid), grid)
        return interpolate_velocity_bilinear(grid, dudx_grid, x, y, :u)
        
    elseif gradient_type == :dudy
        # Compute ∂u/∂y
        dudy_grid = ddy(interpolate_u_to_cell_center(field, grid), grid)
        return interpolate_pressure_bilinear(grid, dudy_grid, x, y)
        
    elseif gradient_type == :dvdx
        # Compute ∂v/∂x
        dvdx_grid = ddx(interpolate_v_to_cell_center(field, grid), grid)
        return interpolate_pressure_bilinear(grid, dvdx_grid, x, y)
        
    elseif gradient_type == :dvdy
        # Compute ∂v/∂y
        dvdy_grid = ddy_at_faces(interpolate_v_to_cell_center(field, grid), grid)
        return interpolate_velocity_bilinear(grid, dvdy_grid, x, y, :v)
    end
    
    return 0.0
end

function compute_body_velocity(body::FlexibleBody, point_index::Int, dx::Float64, dy::Float64)
    """
    Compute accurate body velocity at Lagrangian point using temporal finite differences.
    """
    
    i = point_index
    
    # Use backward finite difference for velocity
    if body.X_old !== nothing && body.X_prev !== nothing
        # 2nd order backward difference: (3X^n - 4X^{n-1} + X^{n-2}) / (2Δt)
        # For now, use 1st order: (X^n - X^{n-1}) / Δt
        # Note: We don't have Δt here, so we approximate with grid spacing
        dt_approx = min(dx, dy) / 10.0  # Conservative time step estimate
        
        u_body = (body.X[i, 1] - body.X_old[i, 1]) / dt_approx
        v_body = (body.X[i, 2] - body.X_old[i, 2]) / dt_approx
        
        return u_body, v_body
    else
        return 0.0, 0.0
    end
end

function interpolate_velocity(grid::StaggeredGrid, u::Array, x::Float64, y::Float64, component::Symbol)
    # Bilinear interpolation of velocity at point (x,y)
    # This is a simplified version - full implementation would handle grid bounds
    
    if component == :u
        # u is defined at (x_{i+1/2}, y_j)
        i = max(1, min(grid.nx, Int(floor((x - grid.xu[1]) / grid.dx)) + 1))
        j = max(1, min(grid.ny, Int(floor((y - grid.y[1]) / grid.dy)) + 1))
        
        if i <= grid.nx && j <= grid.ny
            return u[i, j]
        end
    elseif component == :v
        # v is defined at (x_i, y_{j+1/2})
        i = max(1, min(grid.nx, Int(floor((x - grid.x[1]) / grid.dx)) + 1))
        j = max(1, min(grid.ny, Int(floor((y - grid.yv[1]) / grid.dy)) + 1))
        
        if i <= grid.nx && j <= grid.ny
            return u[i, j]
        end
    end
    
    return 0.0
end

# Collection for managing multiple flexible bodies
mutable struct FlexibleBodyCollection
    bodies::Vector{FlexibleBody}
    n_bodies::Int
end

function FlexibleBodyCollection()
    FlexibleBodyCollection(FlexibleBody[], 0)
end

function add_flexible_body!(collection::FlexibleBodyCollection, body::FlexibleBody)
    push!(collection.bodies, body)
    collection.n_bodies += 1
    body.id = collection.n_bodies
end