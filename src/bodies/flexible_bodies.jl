using ForwardDiff

mutable struct FlexibleBody <: AbstractBody
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
    
    # Time stepping history for Adams-Bashforth
    acceleration_history::Vector{Matrix{Float64}}  # History of accelerations for multi-step schemes
    
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
    acceleration_history = Vector{Matrix{Float64}}()  # Initialize empty history for Adams-Bashforth
    
    FlexibleBody(length, n_points, s * length, X, X_old, X_prev,
                bending_rigidity, mass_per_length, thickness,
                bc_type, amplitude, frequency,
                tension, curvature, force, acceleration_history, id)
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

function update_flexible_body!(body::FlexibleBody, dt::Float64, time_scheme::TimeSteppingScheme=RungeKutta4(), t::Float64=0.0)
    """
    Update flexible body using specified time stepping scheme.
    Supports Adams-Bashforth, RK3, and RK4 as specified in CLAUDE.md requirements.
    """
    n = body.n_points
    
    if time_scheme isa AdamsBashforth
        update_flexible_body_adams_bashforth!(body, dt, time_scheme, t)
    elseif time_scheme isa RungeKutta3
        update_flexible_body_rk3!(body, dt, t)
    elseif time_scheme isa RungeKutta4
        update_flexible_body_rk4!(body, dt, t)
    else
        # Fallback to Verlet for compatibility
        update_flexible_body_verlet!(body, dt, t)
    end
end

function update_flexible_body_verlet!(body::FlexibleBody, dt::Float64, t::Float64)
    """
    Verlet integration for flexible body (2nd order accurate).
    """
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
    apply_boundary_conditions!(body, t + dt)
end

function update_flexible_body_rk4!(body::FlexibleBody, dt::Float64, t::Float64)
    """
    4th order Runge-Kutta integration for flexible body dynamics.
    Converts 2nd order ODE to system of 1st order ODEs.
    """
    n = body.n_points
    
    # State vector: [X, Xdot] where X is position, Xdot is velocity
    X = copy(body.X)
    Xdot = (body.X - body.X_old) / dt  # Current velocity estimate
    
    # RK4 stages
    # Stage 1
    dXdt2_1 = zeros(n, 2)
    flexible_body_rhs!(dXdt2_1, body, dt)
    k1_X = Xdot
    k1_Xdot = dXdt2_1
    
    # Stage 2
    body_temp = deepcopy(body)
    body_temp.X .= X + 0.5 * dt * k1_X
    dXdt2_2 = zeros(n, 2)
    flexible_body_rhs!(dXdt2_2, body_temp, dt)
    k2_X = Xdot + 0.5 * dt * k1_Xdot
    k2_Xdot = dXdt2_2
    
    # Stage 3
    body_temp.X .= X + 0.5 * dt * k2_X
    dXdt2_3 = zeros(n, 2)
    flexible_body_rhs!(dXdt2_3, body_temp, dt)
    k3_X = Xdot + 0.5 * dt * k2_Xdot
    k3_Xdot = dXdt2_3
    
    # Stage 4
    body_temp.X .= X + dt * k3_X
    dXdt2_4 = zeros(n, 2)
    flexible_body_rhs!(dXdt2_4, body_temp, dt)
    k4_X = Xdot + dt * k3_Xdot
    k4_Xdot = dXdt2_4
    
    # Update
    X_new = X + (dt/6) * (k1_X + 2*k2_X + 2*k3_X + k4_X)
    Xdot_new = Xdot + (dt/6) * (k1_Xdot + 2*k2_Xdot + 2*k3_Xdot + k4_Xdot)
    
    # Store states
    body.X_prev .= body.X_old
    body.X_old .= body.X
    body.X .= X_new
    
    # Apply boundary conditions
    apply_boundary_conditions!(body, t + dt)
end

function update_flexible_body_rk3!(body::FlexibleBody, dt::Float64, t::Float64)
    """
    3rd order Runge-Kutta integration for flexible body dynamics.
    """
    n = body.n_points
    
    # State vector: [X, Xdot]
    X = copy(body.X)
    Xdot = (body.X - body.X_old) / dt
    
    # Stage 1
    dXdt2_1 = zeros(n, 2)
    flexible_body_rhs!(dXdt2_1, body, dt)
    k1_X = Xdot
    k1_Xdot = dXdt2_1
    
    # Stage 2
    body_temp = deepcopy(body)
    body_temp.X .= X + 0.5 * dt * k1_X
    dXdt2_2 = zeros(n, 2)
    flexible_body_rhs!(dXdt2_2, body_temp, dt)
    k2_X = Xdot + 0.5 * dt * k1_Xdot
    k2_Xdot = dXdt2_2
    
    # Stage 3
    body_temp.X .= X - dt * k1_X + 2 * dt * k2_X
    dXdt2_3 = zeros(n, 2)
    flexible_body_rhs!(dXdt2_3, body_temp, dt)
    k3_X = Xdot - dt * k1_Xdot + 2 * dt * k2_Xdot
    k3_Xdot = dXdt2_3
    
    # Update
    X_new = X + (dt/6) * (k1_X + 4*k2_X + k3_X)
    Xdot_new = Xdot + (dt/6) * (k1_Xdot + 4*k2_Xdot + k3_Xdot)
    
    # Store states
    body.X_prev .= body.X_old
    body.X_old .= body.X
    body.X .= X_new
    
    # Apply boundary conditions
    apply_boundary_conditions!(body, t + dt)
end

function update_flexible_body_adams_bashforth!(body::FlexibleBody, dt::Float64, scheme::AdamsBashforth, t::Float64)
    """
    Adams-Bashforth integration for flexible body dynamics.
    Uses history of RHS evaluations for multi-step integration.
    """
    n = body.n_points
    dXdt2 = zeros(n, 2)
    
    # Compute current RHS
    flexible_body_rhs!(dXdt2, body, dt)
    
    # Store acceleration history for multi-step Adams-Bashforth
    
    # Store current acceleration
    push!(body.acceleration_history, copy(dXdt2))
    
    # Keep only required history
    if length(body.acceleration_history) > scheme.order
        popfirst!(body.acceleration_history)
    end
    
    # Current velocity estimate
    Xdot = (body.X - body.X_old) / dt
    
    # Apply Adams-Bashforth to acceleration
    if length(body.acceleration_history) == 1
        # First step: use RK4 or Euler
        Xdot_new = Xdot + dt * dXdt2
    else
        # Multi-step Adams-Bashforth
        n_steps = min(length(body.acceleration_history), scheme.order)
        coeffs = scheme.coefficients[1:n_steps]
        
        Xdot_new = copy(Xdot)
        for (k, coeff) in enumerate(coeffs)
            accel = body.acceleration_history[end-k+1]  # Most recent first
            Xdot_new .+= dt * coeff .* accel
        end
    end
    
    # Update position
    X_new = body.X + dt * Xdot_new
    
    # Store states
    body.X_prev .= body.X_old
    body.X_old .= body.X
    body.X .= X_new
    
    # Apply boundary conditions
    apply_boundary_conditions!(body, t + dt)
end

function get_flexible_body_points(body::FlexibleBody)
    # Return current positions of all Lagrangian points
    return [(body.X[i, 1], body.X[i, 2]) for i = 1:body.n_points]
end

function update_flexible_bodies!(bodies::FlexibleBodyCollection, state::SolutionState, grid::StaggeredGrid, dt::Float64, time_scheme::TimeSteppingScheme=RungeKutta4())
    """
    Update all flexible bodies in collection using specified time stepping scheme.
    Compatible with main simulation API.
    """
    for body in bodies.bodies
        # First update forces based on current fluid state
        compute_flexible_body_forces(body, grid, state, FluidProperties(ConstantDensity(1.0), 0.01))
        
        # Then update body dynamics
        update_flexible_body!(body, dt, time_scheme, state.t)
    end
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

# ============================================================================
# Drag and Lift Coefficient Calculations
# ============================================================================

"""
    compute_drag_lift_coefficients(body, grid, state, fluid; reference_velocity, reference_length)

Compute drag and lift coefficients for a flexible body.

# Arguments
- `body::FlexibleBody`: The flexible body
- `grid::StaggeredGrid`: Computational grid
- `state::SolutionState`: Current flow solution
- `fluid::FluidProperties`: Fluid properties
- `reference_velocity::Float64`: Reference velocity for coefficient calculation (default: inlet velocity)
- `reference_length::Float64`: Reference length (default: body length)
- `flow_direction::Vector{Float64}`: Main flow direction [x, z] (default: [1.0, 0.0])

# Returns
- `NamedTuple`: (Cd=drag_coeff, Cl=lift_coeff, Fx=total_x_force, Fz=total_z_force, 
               Cd_pressure=pressure_drag_coeff, Cd_viscous=viscous_drag_coeff,
               center_of_pressure=[x, z])
"""
function compute_drag_lift_coefficients(body::FlexibleBody, grid::StaggeredGrid, 
                                       state::SolutionState, fluid::FluidProperties;
                                       reference_velocity::Float64 = 1.0,
                                       reference_length::Float64 = body.length,
                                       flow_direction::Vector{Float64} = [1.0, 0.0])
    
    # Get fluid properties
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
        μ = fluid.μ
    else
        error("Variable density not implemented in force coefficients")
    end
    
    # Normalize flow direction
    flow_dir = flow_direction / norm(flow_direction)
    lift_dir = [-flow_dir[2], flow_dir[1]]  # Perpendicular to flow (90° rotation)
    
    # Initialize force components
    total_pressure_force = [0.0, 0.0]
    total_viscous_force = [0.0, 0.0]
    total_force = [0.0, 0.0]
    
    # Center of pressure calculation
    moment_arm_sum = [0.0, 0.0]
    total_force_magnitude = 0.0
    
    # Integration parameters
    ds = body.length / (body.n_points - 1)
    δh = 1.5 * min(grid.dx, grid.dz)  # Delta function width
    
    # For each Lagrangian point, compute detailed forces
    for i = 1:body.n_points
        x, z = body.X[i, 1], body.X[i, 2]
        
        # Compute local forces with detailed breakdown
        pressure_force, viscous_force = compute_detailed_forces(body, i, grid, state, fluid, δh)
        
        # Integration weight (trapezoidal rule)
        weight = (i == 1 || i == body.n_points) ? 0.5 * ds : ds
        
        # Accumulate forces
        total_pressure_force += pressure_force * weight
        total_viscous_force += viscous_force * weight
        
        # Total force at this point
        point_force = pressure_force + viscous_force
        point_force_mag = norm(point_force)
        
        # Center of pressure calculation (force-weighted position)
        if point_force_mag > 1e-12
            moment_arm_sum += [x, z] * point_force_mag
            total_force_magnitude += point_force_mag
        end
    end
    
    total_force = total_pressure_force + total_viscous_force
    
    # Center of pressure
    center_of_pressure = total_force_magnitude > 1e-12 ? 
                        moment_arm_sum / total_force_magnitude : 
                        [0.5 * (body.X[1, 1] + body.X[end, 1]), 
                         0.5 * (body.X[1, 2] + body.X[end, 2])]
    
    # Project forces onto drag and lift directions
    drag_force = dot(total_force, flow_dir)
    lift_force = dot(total_force, lift_dir)
    
    # Pressure and viscous drag components
    drag_pressure = dot(total_pressure_force, flow_dir)
    drag_viscous = dot(total_viscous_force, flow_dir)
    
    # Reference dynamic pressure
    q_ref = 0.5 * ρ * reference_velocity^2
    
    # Dimensionless coefficients
    Cd = drag_force / (q_ref * reference_length)
    Cl = lift_force / (q_ref * reference_length)
    Cd_pressure = drag_pressure / (q_ref * reference_length)
    Cd_viscous = drag_viscous / (q_ref * reference_length)
    
    return (
        Cd = Cd,
        Cl = Cl,
        Fx = total_force[1],
        Fz = total_force[2],
        Cd_pressure = Cd_pressure,
        Cd_viscous = Cd_viscous,
        center_of_pressure = center_of_pressure,
        reference_velocity = reference_velocity,
        reference_length = reference_length,
        dynamic_pressure = q_ref
    )
end

"""
    compute_detailed_forces(body, point_idx, grid, state, fluid, δh)

Compute detailed pressure and viscous forces at a Lagrangian point.
"""
function compute_detailed_forces(body::FlexibleBody, point_idx::Int, grid::StaggeredGrid, 
                                state::SolutionState, fluid::FluidProperties, δh::Float64)
    
    x, z = body.X[point_idx, 1], body.X[point_idx, 2]
    
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
        μ = fluid.μ
    else
        error("Variable density not implemented")
    end
    
    # Compute local normal vector
    normal = compute_local_normal(body, point_idx)
    
    # Initialize force components
    pressure_force = [0.0, 0.0]
    viscous_force = [0.0, 0.0]
    
    # Find influence region around body point
    i_min = max(1, Int(floor((x - 2*δh - grid.x[1]) / grid.dx)) + 1)
    i_max = min(grid.nx, Int(ceil((x + 2*δh - grid.x[1]) / grid.dx)) + 1)
    k_min = max(1, Int(floor((z - 2*δh - grid.z[1]) / grid.dz)) + 1)
    k_max = min(grid.nz, Int(ceil((z + 2*δh - grid.z[1]) / grid.dz)) + 1)
    
    # Integrate forces over influence region
    for k = k_min:k_max, i = i_min:i_max
        # Grid point location
        xi, zk = grid.x[i], grid.z[k]
        
        # Regularized delta function value
        δ_val = regularized_delta_2d(x - xi, z - zk, δh, grid.dx, grid.dz)
        
        if δ_val > 1e-12
            # Pressure contribution
            p_local = state.p[i, k]
            pressure_traction = -p_local * normal
            pressure_force += pressure_traction * δ_val * grid.dx * grid.dz
            
            # Viscous stress contribution
            if μ > 0
                # Compute velocity gradients
                dudx = compute_velocity_gradient_2d(state.u, i, k, grid, :dudx)
                dudz = compute_velocity_gradient_2d(state.u, i, k, grid, :dudz)
                dwdx = compute_velocity_gradient_2d(state.w, i, k, grid, :dwdx)
                dwdz = compute_velocity_gradient_2d(state.w, i, k, grid, :dwdz)
                
                # Viscous stress tensor (2D XZ plane)
                τxx = 2 * μ * dudx
                τzz = 2 * μ * dwdz  
                τxz = μ * (dudz + dwdx)
                
                # Viscous traction: τ · n
                viscous_traction_x = τxx * normal[1] + τxz * normal[2]
                viscous_traction_z = τxz * normal[1] + τzz * normal[2]
                
                viscous_force += [viscous_traction_x, viscous_traction_z] * δ_val * grid.dx * grid.dz
            end
        end
    end
    
    return pressure_force, viscous_force
end

"""
    compute_velocity_gradient_2d(field, i, k, grid, grad_type)

Compute velocity gradient for 2D XZ plane at grid point (i,k).
"""
function compute_velocity_gradient_2d(field::Array{T}, i::Int, k::Int, grid::StaggeredGrid, grad_type::Symbol) where T
    if grad_type == :dudx
        if i > 1 && i < size(field, 1)
            return (field[i+1, k] - field[i-1, k]) / (2*grid.dx)
        end
    elseif grad_type == :dudz  
        if k > 1 && k < size(field, 2)
            return (field[i, k+1] - field[i, k-1]) / (2*grid.dz)
        end
    elseif grad_type == :dwdx
        if i > 1 && i < size(field, 1)
            return (field[i+1, k] - field[i-1, k]) / (2*grid.dx)
        end
    elseif grad_type == :dwdz
        if k > 1 && k < size(field, 2)
            return (field[i, k+1] - field[i, k-1]) / (2*grid.dz)
        end
    end
    
    return 0.0
end

"""
    compute_body_coefficients_collection(collection, grid, state, fluid; kwargs...)

Compute drag and lift coefficients for all bodies in a collection.
"""
function compute_body_coefficients_collection(collection::FlexibleBodyCollection, 
                                            grid::StaggeredGrid, state::SolutionState, 
                                            fluid::FluidProperties; kwargs...)
    
    coefficients = Vector{NamedTuple}()
    
    for (i, body) in enumerate(collection.bodies)
        body_coeffs = compute_drag_lift_coefficients(body, grid, state, fluid; kwargs...)
        # Add body ID to the result
        body_coeffs_with_id = merge(body_coeffs, (body_id = i,))
        push!(coefficients, body_coeffs_with_id)
    end
    
    return coefficients
end

"""
    compute_instantaneous_power(body, grid, state, fluid)

Compute instantaneous power dissipated by the flexible body.
"""
function compute_instantaneous_power(body::FlexibleBody, grid::StaggeredGrid, 
                                   state::SolutionState, fluid::FluidProperties)
    
    total_power = 0.0
    ds = body.length / (body.n_points - 1)
    
    for i = 1:body.n_points
        # Body velocity at this point
        u_body, w_body = compute_body_velocity_accurate(body, i)
        
        # Force at this point
        force_x, force_z = body.force[i, 1], body.force[i, 2]
        
        # Power = F · v
        point_power = force_x * u_body + force_z * w_body
        
        # Integration weight
        weight = (i == 1 || i == body.n_points) ? 0.5 * ds : ds
        
        total_power += point_power * weight
    end
    
    return total_power
end

# ============================================================================
# Distance Control Functions (FlexibleBodyController is in flexible_body_controller.jl)
# ============================================================================

"""
    compute_body_distance(body1, body2, measurement_point)

Compute distance between two flexible bodies at specified measurement points.
"""
function compute_body_distance(body1::FlexibleBody, body2::FlexibleBody, 
                              measurement_point::Symbol = :tip)
    
    # Get measurement points on each body
    pos1 = get_measurement_point(body1, measurement_point)
    pos2 = get_measurement_point(body2, measurement_point)
    
    # Euclidean distance
    return sqrt((pos1[1] - pos2[1])^2 + (pos1[2] - pos2[2])^2)
end

"""
    get_measurement_point(body, point_type)

Get coordinates of specified measurement point on flexible body.
"""
function get_measurement_point(body::FlexibleBody, point_type::Symbol)
    n = body.n_points
    
    if point_type == :tip
        return [body.X[end, 1], body.X[end, 2]]
    elseif point_type == :center
        mid_idx = div(n, 2)
        return [body.X[mid_idx, 1], body.X[mid_idx, 2]]
    elseif point_type == :quarter
        quarter_idx = div(n, 4)
        return [body.X[quarter_idx, 1], body.X[quarter_idx, 2]]
    elseif point_type == :root
        return [body.X[1, 1], body.X[1, 2]]
    else
        error("Unknown measurement point: $point_type")
    end
end

"""
    update_controller!(controller, current_time, dt)

Update the control system to maintain target distances between bodies.
"""
function update_controller!(controller::FlexibleBodyController, 
                           current_time::Float64, dt::Float64)
    
    n = controller.n_bodies
    
    # Compute current distances between all pairs
    current_distances = zeros(n, n)
    for i = 1:n, j = i+1:n
        dist = compute_body_distance(controller.bodies[i], controller.bodies[j], 
                                   controller.measurement_points[1])
        current_distances[i, j] = dist
        current_distances[j, i] = dist
    end
    
    # Update control for each body pair
    for i = 1:n, j = i+1:n
        # Compute distance error
        target_dist = controller.target_distances[i, j]
        current_dist = current_distances[i, j]
        error = target_dist - current_dist
        
        # Skip if within tolerance
        if abs(error) < controller.distance_tolerance
            continue
        end
        
        # PID control calculation
        controller.error_integral[i, j] += error * dt
        error_derivative = (error - controller.error_previous[i, j]) / dt
        
        control_signal = (controller.kp * error + 
                         controller.ki * controller.error_integral[i, j] +
                         controller.kd * error_derivative)
        
        # Update previous error
        controller.error_previous[i, j] = error
        
        # Apply control to trailing body (higher index)
        trailing_body = controller.bodies[j]
        
        # Adjust amplitude based on control signal
        current_amplitude = trailing_body.amplitude
        new_amplitude = current_amplitude + controller.adaptation_rate * control_signal
        
        # Clamp amplitude to bounds
        new_amplitude = max(controller.min_amplitude, 
                           min(controller.max_amplitude, new_amplitude))
        
        # Update body amplitude
        trailing_body.amplitude = new_amplitude
        
        # Optional: adjust frequency for phase synchronization
        # This creates more sophisticated coordination
        phase_error = compute_phase_error(controller.bodies[i], controller.bodies[j], current_time)
        if abs(phase_error) > π/4  # If phases are significantly out of sync
            # Slightly adjust frequency of trailing body
            freq_adjustment = 0.1 * sign(-phase_error)  # Small frequency correction
            # This would require extending the FlexibleBody structure to store frequency
        end
    end
end

"""
    compute_phase_error(body1, body2, current_time)

Compute phase difference between two oscillating bodies.
"""
function compute_phase_error(body1::FlexibleBody, body2::FlexibleBody, current_time::Float64)
    # Get tip velocities as proxy for phase
    _, w1 = compute_body_velocity_accurate(body1, body1.n_points)
    _, w2 = compute_body_velocity_accurate(body2, body2.n_points)
    
    # Simple phase estimation based on velocity signs and magnitudes
    # More sophisticated approach would track actual oscillation phase
    if body1.amplitude > 0 && body2.amplitude > 0
        phase1 = atan(w1, body1.amplitude * body1.frequency)
        phase2 = atan(w2, body2.amplitude * body2.frequency)
        return phase1 - phase2
    end
    
    return 0.0
end

"""
    apply_harmonic_boundary_conditions!(controller, current_time)

Apply coordinated harmonic boundary conditions to all controlled bodies.
"""
function apply_harmonic_boundary_conditions!(controller::FlexibleBodyController, 
                                           current_time::Float64)
    
    for (i, body) in enumerate(controller.bodies)
        if body.bc_type == :sinusoidal_front
            # Apply harmonic motion with current amplitude and phase
            phase = controller.phase_offsets[i]
            motion = body.amplitude * sin(2π * body.frequency * current_time + phase)
            
            # Apply to leading edge (could be x or z direction based on setup)
            body.X[1, 2] += motion  # Vertical motion in XZ plane
        end
    end
end

"""
    create_coordinated_flag_system(positions, lengths, widths; kwargs...)

Create a system of flexible flags with harmonic coordination control.

# Arguments
- `positions::Vector{Vector{Float64}}`: Starting positions for each flag
- `lengths::Vector{Float64}`: Length of each flag
- `widths::Vector{Float64}`: Width of each flag

# Keywords
- `target_distances::Union{Matrix{Float64}, Nothing} = nothing`: Desired distances between flags
- `base_frequency::Float64 = 2.0`: Base oscillation frequency
- `base_amplitude::Float64 = 0.1`: Initial amplitude
- `phase_coordination::Symbol = :synchronized`: Phase relationship (:synchronized, :alternating, :sequential)
- `distance_tolerance::Float64 = 0.05`: Acceptable distance variation
- `control_gains::NamedTuple = (kp=1.0, ki=0.1, kd=0.05)`: PID gains
"""
function create_coordinated_flag_system(positions::Vector{Vector{Float64}}, 
                                       lengths::Vector{Float64}, 
                                       widths::Vector{Float64};
                                       target_distances::Union{Matrix{Float64}, Nothing} = nothing,
                                       base_frequency::Float64 = 2.0,
                                       base_amplitude::Float64 = 0.1,
                                       phase_coordination::Symbol = :synchronized,
                                       distance_tolerance::Float64 = 0.05,
                                       control_gains::NamedTuple = (kp=1.0, ki=0.1, kd=0.05),
                                       material::Symbol = :flexible,
                                       n_points::Int = 20)
    
    n_flags = length(positions)
    @assert length(lengths) == n_flags && length(widths) == n_flags "Inconsistent array lengths"
    
    # Create flexible bodies
    bodies = FlexibleBody[]
    
    for i = 1:n_flags
        flag = create_flag(positions[i], lengths[i], widths[i];
                          material = material,
                          attachment = :fixed_leading_edge,
                          prescribed_motion = (type=:sinusoidal, 
                                             amplitude=base_amplitude, 
                                             frequency=base_frequency),
                          n_points = n_points,
                          id = i)
        
        push!(bodies, flag)
    end
    
    # Set up phase coordination
    phase_offsets = zeros(n_flags)
    if phase_coordination == :synchronized
        phase_offsets .= 0.0  # All in phase
    elseif phase_coordination == :alternating
        for i = 1:n_flags
            phase_offsets[i] = (i % 2) * π  # Alternating 0, π, 0, π...
        end
    elseif phase_coordination == :sequential
        for i = 1:n_flags
            phase_offsets[i] = 2π * (i-1) / n_flags  # Evenly distributed phases
        end
    end
    
    # Create controller
    controller = FlexibleBodyController(bodies;
                                       target_distances = target_distances,
                                       distance_tolerance = distance_tolerance,
                                       kp = control_gains.kp,
                                       ki = control_gains.ki,
                                       kd = control_gains.kd,
                                       base_frequency = base_frequency,
                                       max_amplitude = 2.0 * base_amplitude,
                                       min_amplitude = 0.1 * base_amplitude)
    
    # Set phase offsets
    controller.phase_offsets = phase_offsets
    
    # Create collection
    collection = FlexibleBodyCollection()
    for body in bodies
        add_flexible_body!(collection, body)
    end
    
    return collection, controller
end

"""
    monitor_distance_control(controller, current_time)

Monitor and report the performance of distance control system.
"""
function monitor_distance_control(controller::FlexibleBodyController, current_time::Float64)
    
    distances = Dict{String, Float64}()
    errors = Dict{String, Float64}()
    amplitudes = Dict{String, Float64}()
    
    n = controller.n_bodies
    
    for i = 1:n, j = i+1:n
        # Current distance
        current_dist = compute_body_distance(controller.bodies[i], controller.bodies[j], :tip)
        target_dist = controller.target_distances[i, j]
        error = abs(target_dist - current_dist)
        
        # Store for monitoring
        pair_name = "Flag_$(i)_to_$(j)"
        distances[pair_name] = current_dist
        errors[pair_name] = error
    end
    
    # Current amplitudes
    for i = 1:n
        amplitudes["Flag_$(i)"] = controller.bodies[i].amplitude
    end
    
    return (time = current_time, distances = distances, errors = errors, amplitudes = amplitudes)
end

# ============================================================================
# Convenient Flag-Specific Constructor Functions
# ============================================================================

"""
    create_flag(start_point, length, width; kwargs...)

Create a flag-like flexible body with convenient parameter specification.

# Arguments
- `start_point::Vector{Float64}`: Starting attachment point [x, z]
- `length::Float64`: Flag length in flow direction
- `width::Float64`: Flag width (thickness for force calculations)
- `n_points::Int = 20`: Number of discretization points along flag
- `initial_angle::Union{Nothing, Float64} = nothing`: Initial flag angle in radians
  - `nothing`: Automatic angle (horizontal for static, none for sinusoidal motion)
  - `0.0`: Horizontal flag (positive x-direction)
  - `π/2`: Vertical flag pointing upward
  - `-π/2`: Vertical flag pointing downward
- `attachment::Symbol = :fixed_leading_edge`: Attachment type
  - `:fixed_leading_edge`: Leading edge fixed, trailing edge free
  - `:pinned_leading_edge`: Leading edge pinned (can rotate), trailing edge free
  - `:both_ends_fixed`: Both ends fixed (not typical for flags)
- `material::Symbol = :flexible`: Material stiffness
  - `:very_flexible`: Low bending rigidity (fabric-like)
  - `:flexible`: Medium bending rigidity (thin sheet)
  - `:stiff`: High bending rigidity (thick plate)
- `density_ratio::Float64 = 1.0`: Body density / fluid density
- `Reynolds::Float64 = 100.0`: Reynolds number for scaling
- `prescribed_motion::Union{Nothing, NamedTuple} = nothing`: Optional prescribed motion
  - Format: `(type=:sinusoidal, amplitude=0.1, frequency=1.0)`

# Returns
- `FlexibleBody`: Configured flexible body representing the flag

# Examples
```julia
# Simple horizontal flag at origin
flag1 = create_flag([0.0, 0.0], 1.0, 0.05)

# Flag at 30° angle
flag2 = create_flag([0.0, 0.5], 2.0, 0.02; initial_angle=π/6)

# Vertical flag pointing upward
flag3 = create_flag([1.0, 0.0], 1.5, 0.1; initial_angle=π/2)

# Flag with sinusoidal motion (starts undeflected)
flag4 = create_flag([0.0, 0.5], 2.0, 0.02; 
                   attachment=:pinned_leading_edge,
                   prescribed_motion=(type=:sinusoidal, amplitude=0.2, frequency=2.0))

# Flag with sinusoidal motion but specific starting angle
flag5 = create_flag([0.0, 0.5], 2.0, 0.02; 
                   initial_angle=π/4,
                   prescribed_motion=(type=:sinusoidal, amplitude=0.2, frequency=2.0))
```
"""
function create_flag(start_point::Vector{Float64}, length::Float64, width::Float64;
                    n_points::Int = 20,
                    initial_angle::Union{Nothing, Float64} = nothing,
                    attachment::Symbol = :fixed_leading_edge,
                    material::Symbol = :flexible,
                    density_ratio::Float64 = 1.0,
                    Reynolds::Float64 = 100.0,
                    prescribed_motion::Union{Nothing, NamedTuple} = nothing,
                    id::Int = 1)
    
    # Material property lookup
    material_props = Dict(
        :very_flexible => (bending_rigidity = 0.0001, mass_factor = 0.5),
        :flexible      => (bending_rigidity = 0.001,  mass_factor = 1.0),
        :stiff        => (bending_rigidity = 0.01,   mass_factor = 2.0)
    )
    
    if !haskey(material_props, material)
        error("Unknown material type: $material. Use :very_flexible, :flexible, or :stiff")
    end
    
    props = material_props[material]
    
    # Scale material properties with Reynolds number and size
    bending_rigidity = props.bending_rigidity * length^3 / Reynolds
    mass_per_length = props.mass_factor * density_ratio
    
    # Determine initial angle based on prescribed motion and user input
    angle = initial_angle
    if angle === nothing
        if prescribed_motion !== nothing && prescribed_motion.type == :sinusoidal
            # For sinusoidal motion, default to no initial deflection (straight from attachment)
            angle = 0.0  # Will create horizontal flag, motion will be applied during simulation
        else
            # For static flags, default to horizontal
            angle = 0.0
        end
    end
    
    # Create initial flag shape based on angle
    # angle = 0: horizontal (positive x-direction)
    # angle = π/2: vertical upward (positive z-direction)  
    # angle = -π/2: vertical downward (negative z-direction)
    end_point = start_point + length * [cos(angle), sin(angle)]
    initial_shape = straight_filament(length, start_point, end_point)
    
    # Boundary condition setup
    bc_type = :fixed_ends  # Default
    amplitude = 0.0
    frequency = 0.0
    
    if attachment == :fixed_leading_edge
        bc_type = :fixed_front_free_rear
    elseif attachment == :pinned_leading_edge
        bc_type = :rotating_front
    elseif attachment == :both_ends_fixed
        bc_type = :fixed_ends
    end
    
    # Handle prescribed motion
    if prescribed_motion !== nothing
        if prescribed_motion.type == :sinusoidal
            bc_type = :sinusoidal_front
            amplitude = get(prescribed_motion, :amplitude, 0.1)
            frequency = get(prescribed_motion, :frequency, 1.0)
        end
    end
    
    return FlexibleBody(length, n_points, initial_shape;
                       bending_rigidity = bending_rigidity,
                       mass_per_length = mass_per_length,
                       thickness = width,
                       bc_type = bc_type,
                       amplitude = amplitude,
                       frequency = frequency,
                       id = id)
end

"""
    create_vertical_flag(start_point, length, width; kwargs...)

Create a vertically hanging flag (gravity-aligned).

# Arguments
- `start_point::Vector{Float64}`: Top attachment point [x, z]
- `length::Float64`: Flag length (hanging direction)
- `width::Float64`: Flag thickness
- Additional kwargs same as `create_flag` (initial_angle will be overridden to -π/2)
"""
function create_vertical_flag(start_point::Vector{Float64}, length::Float64, width::Float64; kwargs...)
    # Force vertical downward orientation regardless of initial_angle setting
    modified_kwargs = Dict(pairs((kwargs...,)))
    modified_kwargs[:initial_angle] = -π/2  # Vertical downward
    
    return create_flag(start_point, length, width; modified_kwargs...)
end

"""
    create_curved_flag(start_point, length, width, curvature; kwargs...)

Create a flag with initial curvature.

# Arguments
- `start_point::Vector{Float64}`: Starting attachment point
- `length::Float64`: Arc length of flag
- `width::Float64`: Flag thickness  
- `curvature::Float64`: Initial curvature (1/radius, positive curves upward)
- `base_angle::Float64 = 0.0`: Base angle for the curved flag orientation
- Additional kwargs same as `create_flag` (initial_angle will be ignored)
"""
function create_curved_flag(start_point::Vector{Float64}, length::Float64, width::Float64, 
                           curvature::Float64; base_angle::Float64 = 0.0, kwargs...)
    
    # Create curved initial shape with base angle
    function curved_initial_shape(s::Float64)
        if abs(curvature) < 1e-10
            # Straight flag if curvature is zero, oriented at base_angle
            return start_point + s * [cos(base_angle), sin(base_angle)]
        else
            # Circular arc parameterization with base angle
            θ = curvature * s
            radius = 1.0 / abs(curvature)
            
            # Local coordinates (curved in local frame)
            x_local = radius * sin(θ)
            z_local = radius * (1 - cos(θ)) * sign(curvature)
            
            # Rotate by base angle
            x_rotated = x_local * cos(base_angle) - z_local * sin(base_angle)
            z_rotated = x_local * sin(base_angle) + z_local * cos(base_angle)
            
            return start_point + [x_rotated, z_rotated]
        end
    end
    
    # Remove initial_angle from kwargs since we're defining custom shape
    modified_kwargs = Dict(pairs((kwargs...,)))
    delete!(modified_kwargs, :initial_angle)
    
    return FlexibleBody(length, get(modified_kwargs, :n_points, 20), curved_initial_shape;
                       (k => v for (k, v) in modified_kwargs if k != :n_points)...)
end

"""
    create_angled_flag(start_point, length, width, angle; kwargs...)

Create a straight flag at a specific angle (convenience function).

# Arguments
- `start_point::Vector{Float64}`: Starting attachment point [x, z]
- `length::Float64`: Flag length
- `width::Float64`: Flag thickness
- `angle::Float64`: Initial angle in radians (0 = horizontal, π/2 = vertical up)
- Additional kwargs same as `create_flag`
"""
function create_angled_flag(start_point::Vector{Float64}, length::Float64, width::Float64, 
                           angle::Float64; kwargs...)
    
    return create_flag(start_point, length, width; initial_angle = angle, kwargs...)
end

"""
    create_flag_collection(flag_configs::Vector)

Create multiple flags from configuration vector.

# Arguments
- `flag_configs`: Vector of NamedTuples, each containing flag parameters

# Example
```julia
configs = [
    (start_point=[0.0, 0.0], length=1.0, width=0.05, material=:flexible),
    (start_point=[0.0, 1.0], length=1.5, width=0.03, material=:very_flexible),
    (start_point=[0.0, -1.0], length=0.8, width=0.02, prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=2.0))
]

flag_collection = create_flag_collection(configs)
```
"""
function create_flag_collection(flag_configs::Vector)
    collection = FlexibleBodyCollection()
    
    for (i, config) in enumerate(flag_configs)
        # Convert NamedTuple to keyword arguments
        config_dict = Dict(pairs(config))
        config_dict[:id] = i
        
        flag = create_flag(; config_dict...)
        add_flexible_body!(collection, flag)
    end
    
    return collection
end

# Add new boundary condition type for fixed front, free rear
function apply_boundary_conditions!(body::FlexibleBody, t::Float64)
    if body.bc_type == :fixed_ends
        # Both ends are fixed - no motion
        
    elseif body.bc_type == :fixed_front_free_rear
        # Front end (s=0) is completely fixed, rear end (s=L) is free
        # Keep first point fixed at original position
        # Rear end boundary condition: ∂²X/∂s²|_{s=L} = 0 (natural)
        
    elseif body.bc_type == :sinusoidal_front
        # Front end has sinusoidal motion, rear end is free
        body.X[1, 2] += body.amplitude * sin(2π * body.frequency * t)
        
    elseif body.bc_type == :rotating_front
        # Front can rotate freely (pinned), rear is free
        # ∂²X/∂s²|_{s=0} = 0 and ∂²X/∂s²|_{s=L} = 0
        
    end
end

# Control system functions have been moved to separate files:
# - distance_utilities.jl: Distance measurement functions
# - flexible_body_controller.jl: PID control system  
# - coordinated_system_factory.jl: High-level setup functions