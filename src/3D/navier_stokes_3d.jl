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
    diffusion_3d!(diff_u, diff_v, diff_w, state.u, state.v, state.w, fluid, grid)
    
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
    multigrid_solver::Any
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
    
    NavierStokesSolver3D(grid, fluid, bc, time_scheme, nothing, pressure_correction,
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
    
    solve_pressure_poisson_3d!(solver.phi, solver.rhs_p, grid, bc)
    
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
    diffusion_3d!(diff_u, diff_v, diff_w, state.u, state.v, state.w, fluid, grid)
    
    rhs_u = -adv_u + diff_u
    rhs_v = -adv_v + diff_v
    rhs_w = -adv_w + diff_w
    
    return (u = rhs_u, v = rhs_v, w = rhs_w)
end

function solve_pressure_poisson_3d!(phi::Array{Float64,3}, rhs::Array{Float64,3},
                                   grid::StaggeredGrid, bc::BoundaryConditions)
    # Placeholder for multigrid solver
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    
    phi .= 0.0
    
    # Simple iterative solver (placeholder)
    for iter = 1:1000
        phi_old = copy(phi)
        
        for k = 2:nz-1, j = 2:ny-1, i = 2:nx-1
            phi[i,j,k] = (
                (phi[i+1,j,k] + phi[i-1,j,k]) / dx^2 + 
                (phi[i,j+1,k] + phi[i,j-1,k]) / dy^2 +
                (phi[i,j,k+1] + phi[i,j,k-1]) / dz^2 - 
                rhs[i,j,k]
            ) / (2/dx^2 + 2/dy^2 + 2/dz^2)
        end
        
        # Apply boundary conditions (homogeneous Neumann)
        phi[1, :, :] .= phi[2, :, :]
        phi[nx, :, :] .= phi[nx-1, :, :]
        phi[:, 1, :] .= phi[:, 2, :]
        phi[:, ny, :] .= phi[:, ny-1, :]
        phi[:, :, 1] .= phi[:, :, 2]
        phi[:, :, nz] .= phi[:, :, nz-1]
        
        if maximum(abs.(phi - phi_old)) < 1e-10
            break
        end
    end
end

function correct_velocity_3d!(state_new::SolutionState, state_predictor::SolutionState,
                             phi::Array{Float64,3}, dt::Float64, grid::StaggeredGrid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    
    # u correction
    for k = 1:nz, j = 1:ny, i = 2:nx
        dphidx = (phi[i, j, k] - phi[i-1, j, k]) / dx
        state_new.u[i, j, k] = state_predictor.u[i, j, k] - dt * dphidx
    end
    state_new.u[1, :, :] = state_predictor.u[1, :, :]
    state_new.u[nx+1, :, :] = state_predictor.u[nx+1, :, :]
    
    # v correction
    for k = 1:nz, j = 2:ny, i = 1:nx
        dphidy = (phi[i, j, k] - phi[i, j-1, k]) / dy
        state_new.v[i, j, k] = state_predictor.v[i, j, k] - dt * dphidy
    end
    state_new.v[:, 1, :] = state_predictor.v[:, 1, :]
    state_new.v[:, ny+1, :] = state_predictor.v[:, ny+1, :]
    
    # w correction
    for k = 2:nz, j = 1:ny, i = 1:nx
        dphidz = (phi[i, j, k] - phi[i, j, k-1]) / dz
        state_new.w[i, j, k] = state_predictor.w[i, j, k] - dt * dphidz
    end
    state_new.w[:, :, 1] = state_predictor.w[:, :, 1]
    state_new.w[:, :, nz+1] = state_predictor.w[:, :, nz+1]
end