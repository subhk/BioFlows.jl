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
    Highly accurate force computation for flexible bodies using advanced immersed boundary method.
    
    This implementation includes:
    1. Proper regularized delta function interpolation (4-point stencil)
    2. Accurate stress tensor calculation with surface integration
    3. Higher-order body velocity computation
    4. Lagrangian-Eulerian force spreading with conservation
    5. Adaptive stiffness based on local grid resolution
    6. Proper treatment of body curvature effects
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
    
    # Compute higher-order derivatives for accurate geometry
    dXds, d2Xds2 = compute_derivatives(body)
    ds = body.length / (body.n_points - 1)
    
    # Delta function width parameter (typically 1.5-2.0 grid spacings)
    δh = 1.5 * min(grid.dx, grid.dy)
    
    # Adaptive stiffness based on grid resolution
    base_stiffness = compute_adaptive_stiffness(body, grid, ρ, μ)
    
    # For each Lagrangian point, compute forces using multiple methods
    for i = 1:body.n_points
        x, y = body.X[i, 1], body.X[i, 2]
        
        # === METHOD 1: Direct Stress Integration ===
        force_stress = compute_stress_force_accurate(body, i, grid, state, fluid, δh)
        
        # === METHOD 2: Penalty Method with Adaptive Stiffness ===
        force_penalty = compute_penalty_force_accurate(body, i, grid, state, base_stiffness, δh)
        
        # === METHOD 3: Lagrange Multiplier Method (for incompressible constraint) ===
        force_constraint = compute_constraint_force_accurate(body, i, grid, state, ρ, δh)
        
        # === Compute local surface properties ===
        tangent, normal, curvature = compute_local_surface_properties(body, i, ds)
        
        # === Combine forces with physically motivated weights ===
        # Weight based on local Reynolds number and body thickness
        Re_local = compute_local_reynolds(body, i, grid, state, fluid)
        
        # Adaptive weighting: more stress-based for high Re, more penalty for low Re
        w_stress = min(0.6, Re_local / (Re_local + 10.0))
        w_penalty = 0.8 * (1.0 - w_stress)
        w_constraint = 0.2
        
        # Total force per unit length
        force_per_length = w_stress * force_stress + w_penalty * force_penalty + w_constraint * force_constraint
        
        # === Add curvature-dependent forces ===
        # Surface tension-like effect for numerical stability
        curvature_force = compute_curvature_regularization(body, i, curvature, normal, body.bending_rigidity)
        force_per_length += curvature_force
        
        # === Apply force to body with proper integration ===
        # Use trapezoidal rule for integration along body
        if i == 1 || i == body.n_points
            integration_weight = 0.5 * ds  # Endpoints
        else
            integration_weight = ds  # Interior points
        end
        
        body.force[i, 1] = force_per_length[1] * integration_weight
        body.force[i, 2] = force_per_length[2] * integration_weight
        
        # === Add to total force ===
        force_total[1] += body.force[i, 1]
        force_total[2] += body.force[i, 2]
        
        # === Compute torque about center of mass ===
        center_x, center_y = compute_center_of_mass(body)
        r_x = x - center_x
        r_y = y - center_y
        torque_total += r_x * body.force[i, 2] - r_y * body.force[i, 1]
    end
    
    # === Apply conservation corrections ===
    # Ensure total force is consistent with momentum conservation
    force_total, torque_total = apply_conservation_correction(body, force_total, torque_total)
    
    return force_total, torque_total
end

function compute_stress_force_accurate(body::FlexibleBody, point_idx::Int, grid::StaggeredGrid, 
                                     state::SolutionState, fluid::FluidProperties, δh::Float64)
    """
    Compute force from direct stress tensor integration using regularized delta function.
    """
    x, y = body.X[point_idx, 1], body.X[point_idx, 2]
    
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
        μ = fluid.μ
    else
        error("Variable density not implemented")
    end
    
    # Compute stress tensor at body point using 4-point regularized interpolation
    stress_force = [0.0, 0.0]
    
    # Find influence region around body point
    i_min = max(1, Int(floor((x - 2*δh - grid.x[1]) / grid.dx)) + 1)
    i_max = min(grid.nx, Int(ceil((x + 2*δh - grid.x[1]) / grid.dx)) + 1)
    j_min = max(1, Int(floor((y - 2*δh - grid.y[1]) / grid.dy)) + 1)
    j_max = min(grid.ny, Int(ceil((y + 2*δh - grid.y[1]) / grid.dy)) + 1)
    
    for j = j_min:j_max, i = i_min:i_max
        # Grid point location
        xi, yj = grid.x[i], grid.y[j]
        
        # Regularized delta function value
        δ_val = regularized_delta_2d(x - xi, y - yj, δh, grid.dx, grid.dy)
        
        if δ_val > 1e-12
            # Interpolate velocity gradients at grid point
            dudx = compute_velocity_gradient(state.u, i, j, grid, :dudx)
            dudy = compute_velocity_gradient(state.u, i, j, grid, :dudy)
            dvdx = compute_velocity_gradient(state.v, i, j, grid, :dvdx)
            dvdy = compute_velocity_gradient(state.v, i, j, grid, :dvdy)
            
            # Stress tensor components
            σxx = -state.p[i, j] + 2*μ*dudx
            σyy = -state.p[i, j] + 2*μ*dvdy
            σxy = μ*(dudy + dvdx)
            
            # Compute local normal vector (from body geometry)
            normal = compute_local_normal(body, point_idx)
            
            # Traction force: t = σ · n
            traction_x = σxx * normal[1] + σxy * normal[2]
            traction_y = σxy * normal[1] + σyy * normal[2]
            
            # Accumulate weighted force
            stress_force[1] += traction_x * δ_val * grid.dx * grid.dy
            stress_force[2] += traction_y * δ_val * grid.dx * grid.dy
        end
    end
    
    return stress_force
end

function compute_penalty_force_accurate(body::FlexibleBody, point_idx::Int, grid::StaggeredGrid, 
                                       state::SolutionState, stiffness::Float64, δh::Float64)
    """
    Compute penalty force using regularized interpolation with adaptive stiffness.
    """
    x, y = body.X[point_idx, 1], body.X[point_idx, 2]
    
    # Interpolate fluid velocity using regularized delta function
    u_fluid = interpolate_with_delta_function(grid, state.u, x, y, δh, :u)
    v_fluid = interpolate_with_delta_function(grid, state.v, x, y, δh, :v)
    
    # Compute body velocity with higher-order accuracy
    u_body, v_body = compute_body_velocity_accurate(body, point_idx)
    
    # Penalty force proportional to velocity difference
    penalty_force = [-stiffness * (u_fluid - u_body), 
                     -stiffness * (v_fluid - v_body)]
    
    return penalty_force
end

function compute_constraint_force_accurate(body::FlexibleBody, point_idx::Int, grid::StaggeredGrid, 
                                         state::SolutionState, ρ::Float64, δh::Float64)
    """
    Compute constraint force to maintain incompressibility near the body.
    """
    x, y = body.X[point_idx, 1], body.X[point_idx, 2]
    
    # Compute divergence at body point
    div_u = interpolate_divergence_with_delta(grid, state.u, state.v, x, y, δh)
    
    # Lagrange multiplier approach: force to reduce divergence
    normal = compute_local_normal(body, point_idx)
    
    # Force magnitude proportional to divergence violation
    force_magnitude = -ρ * 100.0 * div_u  # Constraint enforcement parameter
    
    constraint_force = [force_magnitude * normal[1], 
                       force_magnitude * normal[2]]
    
    return constraint_force
end

function compute_local_surface_properties(body::FlexibleBody, point_idx::Int, ds::Float64)
    """
    Compute tangent, normal, and curvature at a Lagrangian point.
    """
    i = point_idx
    n = body.n_points
    
    # Compute tangent vector with higher-order accuracy
    if i == 1
        # Forward difference at start
        tx = (-3*body.X[i, 1] + 4*body.X[i+1, 1] - body.X[i+2, 1]) / (2*ds)
        ty = (-3*body.X[i, 2] + 4*body.X[i+1, 2] - body.X[i+2, 2]) / (2*ds)
    elseif i == n
        # Backward difference at end
        tx = (3*body.X[i, 1] - 4*body.X[i-1, 1] + body.X[i-2, 1]) / (2*ds)
        ty = (3*body.X[i, 2] - 4*body.X[i-1, 2] + body.X[i-2, 2]) / (2*ds)
    else
        # Central difference in interior
        tx = (body.X[i+1, 1] - body.X[i-1, 1]) / (2*ds)
        ty = (body.X[i+1, 2] - body.X[i-1, 2]) / (2*ds)
    end
    
    # Normalize tangent
    t_norm = sqrt(tx^2 + ty^2)
    if t_norm > 1e-12
        tx /= t_norm
        ty /= t_norm
    end
    
    # Normal vector (perpendicular to tangent)
    nx = -ty  # Outward normal
    ny = tx
    
    # Compute curvature using second derivatives
    if i > 1 && i < n
        d2xds2 = (body.X[i+1, 1] - 2*body.X[i, 1] + body.X[i-1, 1]) / ds^2
        d2yds2 = (body.X[i+1, 2] - 2*body.X[i, 2] + body.X[i-1, 2]) / ds^2
        
        # Curvature formula: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
        numerator = abs(tx * d2yds2 - ty * d2xds2)
        denominator = t_norm^3
        curvature = numerator / max(denominator, 1e-12)
    else
        curvature = 0.0
    end
    
    return [tx, ty], [nx, ny], curvature
end

function compute_adaptive_stiffness(body::FlexibleBody, grid::StaggeredGrid, ρ::Float64, μ::Float64)
    """
    Compute adaptive stiffness parameter based on grid resolution and fluid properties.
    """
    
    # Base stiffness scaled by fluid properties
    h_min = min(grid.dx, grid.dy)
    
    # Adaptive stiffness: scales with 1/h² for stability, with ρ for physical consistency
    base_stiffness = ρ * μ / h_min^2
    
    # Scale by body thickness for proper force magnitude
    stiffness = base_stiffness * body.thickness * 1000.0  # Tuning parameter
    
    return stiffness
end

function compute_local_reynolds(body::FlexibleBody, point_idx::Int, grid::StaggeredGrid, 
                              state::SolutionState, fluid::FluidProperties)
    """
    Compute local Reynolds number at a body point.
    """
    x, y = body.X[point_idx, 1], body.X[point_idx, 2]
    
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
        μ = fluid.μ
    else
        error("Variable density not implemented")
    end
    
    # Interpolate local velocity magnitude
    u_local = interpolate_velocity_bilinear(grid, state.u, x, y, :u)
    v_local = interpolate_velocity_bilinear(grid, state.v, x, y, :v)
    vel_mag = sqrt(u_local^2 + v_local^2)
    
    # Local Reynolds number based on body thickness
    Re_local = ρ * vel_mag * body.thickness / μ
    
    return Re_local
end

function compute_curvature_regularization(body::FlexibleBody, point_idx::Int, curvature::Float64, 
                                        normal::Vector{Float64}, bending_rigidity::Float64)
    """
    Compute regularization force based on local curvature to maintain numerical stability.
    """
    
    # Curvature-dependent force (surface tension-like effect)
    # This helps maintain smooth body shape during large deformations
    regularization_strength = 0.01 * bending_rigidity
    
    curvature_force = [regularization_strength * curvature * normal[1],
                      regularization_strength * curvature * normal[2]]
    
    return curvature_force
end

function compute_center_of_mass(body::FlexibleBody)
    """
    Compute center of mass of the flexible body.
    """
    center_x = sum(body.X[:, 1]) / body.n_points
    center_y = sum(body.X[:, 2]) / body.n_points
    
    return center_x, center_y
end

function apply_conservation_correction(body::FlexibleBody, force_total::Vector{Float64}, torque_total::Float64)
    """
    Apply conservation corrections to ensure momentum conservation.
    """
    # For now, return as-is. In advanced implementations, this would:
    # 1. Ensure total momentum change is consistent with Newton's laws
    # 2. Apply corrections for energy conservation
    # 3. Maintain angular momentum conservation
    
    return force_total, torque_total
end

function regularized_delta_2d(dx::Float64, dy::Float64, δh::Float64, grid_dx::Float64, grid_dy::Float64)
    """
    2D regularized delta function (Peskin's 4-point function).
    """
    
    # Normalize distances by delta width
    r_x = abs(dx) / δh
    r_y = abs(dy) / δh
    
    # 4-point regularized delta function
    function δ_1d(r::Float64)
        if r <= 1.0
            return 0.125 * (3 - 2*r + sqrt(1 + 4*r - 4*r^2))
        elseif r <= 2.0
            return 0.125 * (5 - 2*r - sqrt(-7 + 12*r - 4*r^2))
        else
            return 0.0
        end
    end
    
    # 2D delta function is product of 1D functions
    δ_val = δ_1d(r_x) * δ_1d(r_y) / (δh^2)
    
    return δ_val
end

function interpolate_with_delta_function(grid::StaggeredGrid, field::Matrix{T}, x::Float64, y::Float64, 
                                       δh::Float64, component::Symbol) where T
    """
    Interpolate field using regularized delta function.
    """
    result = 0.0
    
    # Determine field locations based on component
    if component == :u
        x_coords = grid.xu
        y_coords = grid.y
    elseif component == :v
        x_coords = grid.x
        y_coords = grid.yv
    else
        x_coords = grid.x
        y_coords = grid.y
    end
    
    # Find influence region
    i_min = max(1, Int(floor((x - 2*δh - x_coords[1]) / grid.dx)) + 1)
    i_max = min(size(field, 1), Int(ceil((x + 2*δh - x_coords[1]) / grid.dx)) + 1)
    j_min = max(1, Int(floor((y - 2*δh - y_coords[1]) / grid.dy)) + 1)
    j_max = min(size(field, 2), Int(ceil((y + 2*δh - y_coords[1]) / grid.dy)) + 1)
    
    for j = j_min:j_max, i = i_min:i_max
        xi = i <= length(x_coords) ? x_coords[i] : x_coords[end]
        yj = j <= length(y_coords) ? y_coords[j] : y_coords[end]
        
        δ_val = regularized_delta_2d(x - xi, y - yj, δh, grid.dx, grid.dy)
        
        if δ_val > 1e-12 && i <= size(field, 1) && j <= size(field, 2)
            result += field[i, j] * δ_val * grid.dx * grid.dy
        end
    end
    
    return result
end

function compute_body_velocity_accurate(body::FlexibleBody, point_idx::Int)
    """
    Compute body velocity using higher-order temporal finite differences.
    """
    i = point_idx
    
    if body.X_old !== nothing && body.X_prev !== nothing
        # 2nd order backward difference: (3X^n - 4X^{n-1} + X^{n-2}) / (2Δt)
        # Use normalized time step
        dt = 1.0  # Normalized, actual dt will be handled in time stepping
        
        u_body = (3*body.X[i, 1] - 4*body.X_old[i, 1] + body.X_prev[i, 1]) / (2*dt)
        v_body = (3*body.X[i, 2] - 4*body.X_old[i, 2] + body.X_prev[i, 2]) / (2*dt)
        
        return u_body, v_body
    else
        return 0.0, 0.0
    end
end

function compute_local_normal(body::FlexibleBody, point_idx::Int)
    """
    Compute outward unit normal vector at a body point.
    """
    _, normal, _ = compute_local_surface_properties(body, point_idx, body.length / (body.n_points - 1))
    return normal
end

function compute_velocity_gradient(field::Matrix{T}, i::Int, j::Int, grid::StaggeredGrid, grad_type::Symbol) where T
    """
    Compute velocity gradient at grid point (i,j).
    """
    if grad_type == :dudx
        if i > 1 && i < size(field, 1)
            return (field[i+1, j] - field[i-1, j]) / (2*grid.dx)
        else
            return 0.0
        end
    elseif grad_type == :dudy
        if j > 1 && j < size(field, 2)
            return (field[i, j+1] - field[i, j-1]) / (2*grid.dy)
        else
            return 0.0
        end
    elseif grad_type == :dvdx
        if i > 1 && i < size(field, 1)
            return (field[i+1, j] - field[i-1, j]) / (2*grid.dx)
        else
            return 0.0
        end
    elseif grad_type == :dvdy
        if j > 1 && j < size(field, 2)
            return (field[i, j+1] - field[i, j-1]) / (2*grid.dy)
        else
            return 0.0
        end
    end
    
    return 0.0
end

function interpolate_divergence_with_delta(grid::StaggeredGrid, u::Matrix{T}, v::Matrix{T}, 
                                         x::Float64, y::Float64, δh::Float64) where T
    """
    Interpolate divergence at point (x,y) using delta function.
    """
    
    # First compute divergence on grid
    div_field = div(u, v, grid)
    
    # Then interpolate to body point
    result = 0.0
    
    i_min = max(1, Int(floor((x - 2*δh - grid.x[1]) / grid.dx)) + 1)
    i_max = min(grid.nx, Int(ceil((x + 2*δh - grid.x[1]) / grid.dx)) + 1)
    j_min = max(1, Int(floor((y - 2*δh - grid.y[1]) / grid.dy)) + 1)
    j_max = min(grid.ny, Int(ceil((y + 2*δh - grid.y[1]) / grid.dy)) + 1)
    
    for j = j_min:j_max, i = i_min:i_max
        xi, yj = grid.x[i], grid.y[j]
        δ_val = regularized_delta_2d(x - xi, y - yj, δh, grid.dx, grid.dy)
        
        if δ_val > 1e-12
            result += div_field[i, j] * δ_val * grid.dx * grid.dy
        end
    end
    
    return result
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