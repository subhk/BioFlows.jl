function navier_stokes_rhs_2d(state::SolutionState, grid::StaggeredGrid, 
                              fluid::FluidProperties, bc::BoundaryConditions)
    nx, ny = grid.nx, grid.ny
    
    # Allocate temporary arrays
    div_u = zeros(nx, ny)
    adv_u = zeros(nx+1, ny)
    adv_v = zeros(nx, ny+1)
    diff_u = zeros(nx+1, ny)
    diff_v = zeros(nx, ny+1)
    dpdx = zeros(nx+1, ny)
    dpdy = zeros(nx, ny+1)
    
    # Compute divergence of velocity
    divergence_2d!(div_u, state.u, state.v, grid)
    
    # Compute pressure gradients
    gradient_pressure_2d!(dpdx, dpdy, state.p, grid)
    
    # Compute advection terms
    advection_2d!(adv_u, adv_v, state.u, state.v, grid)
    
    # Compute diffusion terms using proper staggered grid operators
    compute_diffusion_2d!(diff_u, diff_v, state.u, state.v, fluid, grid)
    
    # Density
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
    else
        error("Variable density not implemented")
    end
    
    # Assemble momentum equations: ∂u/∂t = -∇⋅(uu) + ν∇²u - (1/ρ)∇p
    rhs_u = -adv_u + diff_u - dpdx ./ ρ
    rhs_v = -adv_v + diff_v - dpdy ./ ρ
    
    return (u = rhs_u, v = rhs_v, div = div_u)
end

mutable struct NavierStokesSolver2D <: AbstractSolver
    grid::StaggeredGrid
    fluid::FluidProperties
    bc::BoundaryConditions
    time_scheme::TimeSteppingScheme
    multigrid_solver::Any
    pressure_correction::Bool
    
    # Work arrays
    u_star::Matrix{Float64}
    v_star::Matrix{Float64}
    phi::Matrix{Float64}  # Pressure correction
    rhs_p::Matrix{Float64}  # Pressure Poisson RHS
end

function NavierStokesSolver2D(grid::StaggeredGrid, fluid::FluidProperties, 
                             bc::BoundaryConditions, time_scheme::TimeSteppingScheme;
                             pressure_correction::Bool=true)
    nx, ny = grid.nx, grid.ny
    
    # Initialize work arrays
    u_star = zeros(nx+1, ny)
    v_star = zeros(nx, ny+1)
    phi = zeros(nx, ny)
    rhs_p = zeros(nx, ny)
    
    NavierStokesSolver2D(grid, fluid, bc, time_scheme, nothing, pressure_correction,
                        u_star, v_star, phi, rhs_p)
end

function solve_step_2d!(solver::NavierStokesSolver2D, state_new::SolutionState, 
                       state_old::SolutionState, dt::Float64)
    
    if solver.pressure_correction
        # Fractional step method (projection method)
        solve_projection_step_2d!(solver, state_new, state_old, dt)
    else
        # Coupled solution (more complex, not implemented here)
        error("Coupled Navier-Stokes solver not implemented")
    end
end

function solve_projection_step_2d!(solver::NavierStokesSolver2D, state_new::SolutionState,
                                  state_old::SolutionState, dt::Float64)
    grid = solver.grid
    fluid = solver.fluid
    bc = solver.bc
    nx, ny = grid.nx, grid.ny
    
    # Step 1: Predictor step (without pressure gradient)
    # Solve: (u* - u^n)/dt = -∇⋅(uu) + ν∇²u
    predictor_rhs = compute_predictor_rhs_2d(state_old, grid, fluid)
    
    # Apply time stepping scheme
    state_predictor = deepcopy(state_old)
    time_step!(state_predictor, state_old, 
              (state, args...) -> predictor_rhs, 
              solver.time_scheme, dt, grid, fluid, bc)
    
    # Step 2: Apply boundary conditions to predictor velocity
    apply_boundary_conditions!(grid, state_predictor, bc, state_old.t + dt)
    
    # Step 3: Solve pressure Poisson equation
    # ∇²φ = (1/dt) ∇⋅u*
    divergence_2d!(solver.rhs_p, state_predictor.u, state_predictor.v, grid)
    solver.rhs_p .*= 1.0 / dt
    
    # Solve Poisson equation for pressure correction
    solve_pressure_poisson_2d!(solver.phi, solver.rhs_p, grid, bc)
    
    # Step 4: Velocity correction
    # u^{n+1} = u* - dt∇φ
    correct_velocity_2d!(state_new, state_predictor, solver.phi, dt, grid)
    
    # Step 5: Pressure update
    if solver.fluid.ρ isa ConstantDensity
        ρ = solver.fluid.ρ.ρ
    else
        error("Variable density not implemented")
    end
    
    state_new.p .= state_old.p .+ solver.phi .* ρ
    state_new.t = state_old.t + dt
    state_new.step = state_old.step + 1
    
    # Apply final boundary conditions
    apply_boundary_conditions!(grid, state_new, bc, state_new.t)
end

function compute_predictor_rhs_2d(state::SolutionState, grid::StaggeredGrid, 
                                 fluid::FluidProperties)
    nx, ny = grid.nx, grid.ny
    
    adv_u = zeros(nx+1, ny)
    adv_v = zeros(nx, ny+1)
    diff_u = zeros(nx+1, ny)
    diff_v = zeros(nx, ny+1)
    
    # Compute advection and diffusion terms (no pressure)
    advection_2d!(adv_u, adv_v, state.u, state.v, grid)
    compute_diffusion_2d!(diff_u, diff_v, state.u, state.v, fluid, grid)
    
    rhs_u = -adv_u + diff_u
    rhs_v = -adv_v + diff_v
    
    return (u = rhs_u, v = rhs_v)
end

function solve_pressure_poisson_2d!(phi::Matrix{Float64}, rhs::Matrix{Float64}, 
                                   grid::StaggeredGrid, bc::BoundaryConditions)
    # Use staggered-aware multigrid solver for optimal performance
    solver = MultigridPoissonSolver(grid; solver_type=:staggered, tolerance=1e-8)
    solve_poisson!(solver, phi, rhs, grid, bc)
end

function correct_velocity_2d!(state_new::SolutionState, state_predictor::SolutionState,
                             phi::Matrix{Float64}, dt::Float64, grid::StaggeredGrid)
    nx, ny = grid.nx, grid.ny
    
    # Use proper staggered grid differential operators
    dpdx_faces, dpdy_faces = grad(phi, grid)
    
    # u correction: u = u* - dt * ∂φ/∂x (at u-velocity locations)
    state_new.u .= state_predictor.u .- dt .* dpdx_faces
    
    # v correction: v = v* - dt * ∂φ/∂y (at v-velocity locations)
    state_new.v .= state_predictor.v .- dt .* dpdy_faces
end