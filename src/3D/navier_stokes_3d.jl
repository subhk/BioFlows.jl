function navier_stokes_rhs_3d(state::SolutionState, grid::StaggeredGrid,
                              fluid::FluidProperties, bc::BoundaryConditions)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # Allocate temporary arrays
    div_u = zeros(nx, ny, nz)
    adv_u = zeros(nx+1, ny, nz)
    adv_v = zeros(nx, ny+1, nz)
    adv_w = zeros(nx, ny, nz+1)
    diff_u = zeros(nx+1, ny, nz)
    diff_v = zeros(nx, ny+1, nz)
    diff_w = zeros(nx, ny, nz+1)
    dpdx = zeros(nx+1, ny, nz)
    dpdy = zeros(nx, ny+1, nz)
    dpdz = zeros(nx, ny, nz+1)
    
    # Compute divergence of velocity
    divergence_3d!(div_u, state.u, state.v, state.w, grid)
    
    # Compute pressure gradients
    gradient_pressure_3d!(dpdx, dpdy, dpdz, state.p, grid)
    
    # Compute advection terms
    advection_3d!(adv_u, adv_v, adv_w, state.u, state.v, state.w, grid)
    
    # Compute diffusion terms
    compute_diffusion_3d!(diff_u, diff_v, diff_w, state.u, state.v, state.w, fluid, grid)
    
    # Density
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
    else
        error("Variable density not implemented")
    end
    
    # Assemble momentum equations
    rhs_u = -adv_u + diff_u - dpdx ./ ρ
    rhs_v = -adv_v + diff_v - dpdy ./ ρ
    rhs_w = -adv_w + diff_w - dpdz ./ ρ
    
    return (u = rhs_u, v = rhs_v, w = rhs_w, div = div_u)
end

mutable struct NavierStokesSolver3D <: AbstractSolver
    grid::StaggeredGrid
    fluid::FluidProperties
    bc::BoundaryConditions
    time_scheme::TimeSteppingScheme
    multigrid_solver::Union{MultigridPoissonSolver, Any, Nothing}
    pressure_correction::Bool
    
    # Work arrays
    u_star::Array{Float64,3}
    v_star::Array{Float64,3}
    w_star::Array{Float64,3}
    phi::Array{Float64,3}
    rhs_p::Array{Float64,3}
end

function NavierStokesSolver3D(grid::StaggeredGrid, fluid::FluidProperties,
                             bc::BoundaryConditions, time_scheme::TimeSteppingScheme;
                             pressure_correction::Bool=true)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # Initialize work arrays
    u_star = zeros(nx+1, ny, nz)
    v_star = zeros(nx, ny+1, nz)
    w_star = zeros(nx, ny, nz+1)
    phi = zeros(nx, ny, nz)
    rhs_p = zeros(nx, ny, nz)
    
    # Create multigrid solver for optimal performance
    mg_solver = MultigridPoissonSolver(grid)
    
    NavierStokesSolver3D(grid, fluid, bc, time_scheme, mg_solver, pressure_correction,
                        u_star, v_star, w_star, phi, rhs_p)
end

function solve_step_3d!(solver::NavierStokesSolver3D, state_new::SolutionState,
                       state_old::SolutionState, dt::Float64)
    
    if solver.pressure_correction
        solve_projection_step_3d!(solver, state_new, state_old, dt)
    else
        error("Coupled Navier-Stokes solver not implemented")
    end
end

function solve_projection_step_3d!(solver::NavierStokesSolver3D, state_new::SolutionState,
                                  state_old::SolutionState, dt::Float64)
    grid = solver.grid
    fluid = solver.fluid
    bc = solver.bc
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # Step 1: Predictor step
    predictor_rhs = compute_predictor_rhs_3d(state_old, grid, fluid)
    
    state_predictor = deepcopy(state_old)
    time_step!(state_predictor, state_old,
              (state, args...) -> predictor_rhs,
              solver.time_scheme, dt, grid, fluid, bc)
    
    # Step 2: Apply boundary conditions to predictor velocity
    apply_boundary_conditions!(grid, state_predictor, bc, state_old.t + dt)
    
    # Step 3: Solve pressure Poisson equation
    divergence_3d!(solver.rhs_p, state_predictor.u, state_predictor.v, state_predictor.w, grid)
    solver.rhs_p .*= 1.0 / dt
    
    # Use multigrid solver for optimal performance
    if solver.multigrid_solver !== nothing
        solve_poisson!(solver.multigrid_solver, solver.phi, solver.rhs_p, grid, bc)
    else
        error("Multigrid solver required for pressure solution. Create solver with MultigridPoissonSolver.")
    end
    
    # Step 4: Velocity correction
    correct_velocity_3d!(state_new, state_predictor, solver.phi, dt, grid)
    
    # Step 5: Pressure update
    if solver.fluid.ρ isa ConstantDensity
        ρ = solver.fluid.ρ.ρ
    else
        error("Variable density not implemented")
    end
    
    state_new.p .= state_old.p .+ solver.phi .* ρ
    state_new.t = state_old.t + dt
    state_new.step = state_old.step + 1
    
    apply_boundary_conditions!(grid, state_new, bc, state_new.t)
end

function compute_predictor_rhs_3d(state::SolutionState, grid::StaggeredGrid,
                                 fluid::FluidProperties)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    adv_u = zeros(nx+1, ny, nz)
    adv_v = zeros(nx, ny+1, nz)
    adv_w = zeros(nx, ny, nz+1)
    diff_u = zeros(nx+1, ny, nz)
    diff_v = zeros(nx, ny+1, nz)
    diff_w = zeros(nx, ny, nz+1)
    
    advection_3d!(adv_u, adv_v, adv_w, state.u, state.v, state.w, grid)
    compute_diffusion_3d!(diff_u, diff_v, diff_w, state.u, state.v, state.w, fluid, grid)
    
    rhs_u = -adv_u + diff_u
    rhs_v = -adv_v + diff_v
    rhs_w = -adv_w + diff_w
    
    return (u = rhs_u, v = rhs_v, w = rhs_w)
end


function correct_velocity_3d!(state_new::SolutionState, state_predictor::SolutionState,
                             phi::Array{Float64,3}, dt::Float64, grid::StaggeredGrid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # Use proper staggered grid differential operators
    dpdx_faces, dpdy_faces, dpdz_faces = grad(phi, grid)
    
    # u correction: u = u* - dt * ∂φ/∂x (at u-velocity locations)
    state_new.u .= state_predictor.u .- dt .* dpdx_faces
    
    # v correction: v = v* - dt * ∂φ/∂y (at v-velocity locations)
    state_new.v .= state_predictor.v .- dt .* dpdy_faces
    
    # w correction: w = w* - dt * ∂φ/∂z (at w-velocity locations)
    state_new.w .= state_predictor.w .- dt .* dpdz_faces
end