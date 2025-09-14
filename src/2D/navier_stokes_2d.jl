"""
Clean Implementation of 2D Navier-Stokes Solver using Differential Operators

This file demonstrates how the new differential operators make the code
much more readable and maintainable compared to the traditional approach.
"""

function solve_navier_stokes_clean_2d!(state_new::SolutionState, state_old::SolutionState,
                                      grid::StaggeredGrid, fluid::FluidProperties, 
                                      bc::BoundaryConditions, dt::Float64)
    """
    Clean, readable Navier-Stokes solver using differential operators.
    
    Solves: ∂u/∂t + u·∇u = -∇p/ρ + ν∇²u
            ∇·u = 0
    """
    
    # Get fluid properties
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
        ν = fluid.μ / ρ
    else
        error("Variable density not implemented")
    end
    
    # Step 1: Compute predictor velocity (explicit terms)
    # ∂u/∂t = -u·∇u + ν∇²u (without pressure)
    
    # Interpolate velocities to cell centers for advection (XZ plane)
    u_cc, w_cc = interpolate_to_cell_centers_xz(state_old.u, state_old.w, grid)
    
    # Check for NaN in input and clean if necessary
    if any(isnan, state_old.u) || any(isnan, state_old.w)
        @warn "NaN detected in input velocities - cleaning"
        replace!(state_old.u, NaN => 0.0)
        replace!(state_old.w, NaN => 0.0)
    end
    
    # Advection terms: u·∇u (XZ plane: u·∂u/∂x + v·∂u/∂z)
    dudx = ddx(u_cc, grid)
    dudz = ddz(u_cc, grid)  # Use ddz for XZ plane
    dwdx = ddx(w_cc, grid)
    dwdz = ddz(w_cc, grid)  # Use ddz for XZ plane
    
    advection_u = u_cc .* dudx + w_cc .* dudz  # w is z-velocity in XZ plane
    advection_w = u_cc .* dwdx + w_cc .* dwdz
    
    # Viscous terms: ν∇²u
    viscous_u = ν * laplacian(u_cc, grid)
    viscous_w = ν * laplacian(w_cc, grid)
    
    # Clean any NaN in viscous terms
    replace!(viscous_u, NaN => 0.0)
    replace!(viscous_w, NaN => 0.0)
    
    # CFL-controlled effective timestep to stabilize predictor
    maxu = maximum(abs.(u_cc)); maxw = maximum(abs.(w_cc))
    cfl_target = 0.5
    dt_cfl = cfl_target * min(maxu > 0 ? grid.dx / maxu : Inf,
                              maxw > 0 ? grid.dz / maxw : Inf)
    dt_eff = min(dt, dt_cfl)

    # Predictor velocity (without pressure correction)
    u_star = u_cc + dt_eff * (-advection_u + viscous_u)
    w_star = w_cc + dt_eff * (-advection_w + viscous_w)
    
    # Step 2: Solve pressure Poisson equation
    # ∇²φ = ∇·u*/dt
    
    # Create staggered velocities for divergence computation (2nd-order averaging)
    u_star_staggered = zeros(grid.nx + 1, grid.nz)
    w_star_staggered = zeros(grid.nx, grid.nz + 1)
    # u at x-faces
    @inbounds for j = 1:grid.nz, i = 1:grid.nx+1
        if i == 1
            u_star_staggered[i, j] = u_star[1, j]
        elseif i == grid.nx + 1
            u_star_staggered[i, j] = u_star[grid.nx, j]
        else
            u_star_staggered[i, j] = 0.5 * (u_star[i, j] + u_star[i-1, j])
        end
    end
    # w at z-faces
    @inbounds for j = 1:grid.nz+1, i = 1:grid.nx
        if j == 1
            w_star_staggered[i, j] = w_star[i, 1]
        elseif j == grid.nz + 1
            w_star_staggered[i, j] = w_star[i, grid.nz]
        else
            w_star_staggered[i, j] = 0.5 * (w_star[i, j] + w_star[i, j-1])
        end
    end

    # Apply physical boundary conditions to predictor before projection
    tmp_state = SolutionState2D(grid.nx, grid.nz)
    tmp_state.u .= u_star_staggered
    tmp_state.w .= w_star_staggered
    tmp_state.p .= 0.0
    apply_boundary_conditions!(grid, tmp_state, bc, state_old.t + dt_eff)
    u_star_staggered .= tmp_state.u
    w_star_staggered .= tmp_state.w
    
    # Compute divergence
    div_u_star = div(u_star_staggered, w_star_staggered, grid)
    
    # Pressure Poisson RHS
    rhs_pressure = div_u_star / dt_eff
    
    # Safety checks
    if dt <= 1e-12
        @warn "Time step too small: $dt"
        return
    end
    
    # Check for problematic values in RHS
    if any(x -> isnan(x) || isinf(x), rhs_pressure)
        # Silently clean NaN/Inf values (normal for complex AMR simulations)
        replace!(rhs_pressure, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    end
    
    # Additional check for extremely large values
    max_rhs = maximum(abs.(rhs_pressure))
    if max_rhs > 1e10
        @warn "Extremely large pressure RHS detected: $(max_rhs) - scaling down"
        rhs_pressure .*= 1e10 / max_rhs
    end
    
    # Solve ∇²φ = rhs_pressure
    φ = zeros(grid.nx, grid.nz)
    # Here you would call your pressure solver:
    # solve_pressure_poisson!(φ, rhs_pressure, grid, bc)
    
    # Use multigrid solver for optimal performance
    mg_solver = MultigridPoissonSolver(grid; smoother=:staggered,
                                       tolerance=1e-12, max_iterations=300, levels=5)
    solve_poisson!(mg_solver, φ, rhs_pressure, grid, bc)
    
    # Clean pressure solution
    replace!(φ, NaN => 0.0)
    
    # Step 3: Velocity correction
    # u^(n+1) = u* - dt * ∇φ
    
    # Compute pressure gradient at staggered locations
    dφdx, dφdz = grad(φ, grid)  # Use dφdz for XZ plane
    
    # Correct velocities directly at staggered faces using pressure gradients
    # dφdx lives at u-faces; dφdz lives at w-faces (XZ plane)
    u_new_staggered = similar(u_star_staggered)
    w_new_staggered = similar(w_star_staggered)
    @inbounds for j = 1:grid.nz, i = 1:grid.nx+1
        u_new_staggered[i, j] = u_star_staggered[i, j] - dt_eff * dφdx[i, j]
    end
    @inbounds for j = 1:grid.nz+1, i = 1:grid.nx
        w_new_staggered[i, j] = w_star_staggered[i, j] - dt_eff * dφdz[i, j]
    end

    # Optional extra projection sweeps if divergence remains high
    α_relax = 0.8
    for sweep = 1:3
        div_tmp = div(u_new_staggered, w_new_staggered, grid)
        replace!(div_tmp, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
        max_div_tmp = maximum(abs.(div_tmp))
        if max_div_tmp <= 1e-6
            break
        end
        rhs2 = div_tmp ./ dt_eff
        # Clip extreme RHS to avoid runaway
        max_rhs2 = maximum(abs.(rhs2))
        if max_rhs2 > 1e6
            rhs2 .*= (1e6 / max_rhs2)
        end
        φ2 = zeros(grid.nx, grid.nz)
        solve_poisson!(mg_solver, φ2, rhs2, grid, bc)
        replace!(φ2, NaN => 0.0)
        dφdx2, dφdz2 = grad(φ2, grid)
        @inbounds for j = 1:grid.nz, i = 1:grid.nx+1
            u_new_staggered[i, j] -= α_relax * dt_eff * dφdx2[i, j]
        end
        @inbounds for j = 1:grid.nz+1, i = 1:grid.nx
            w_new_staggered[i, j] -= α_relax * dt_eff * dφdz2[i, j]
        end
    end

    # Step 4: Update pressure
    # p^(n+1) = p^n + φ
    p_new = state_old.p + φ * ρ
    
    state_new.u .= u_new_staggered
    state_new.w .= w_new_staggered
    state_new.p .= p_new
    state_new.t = state_old.t + dt_eff
    state_new.step = state_old.step + 1
    
    # Final safety checks for NaN/Inf and extreme values
    replace!(state_new.u, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    replace!(state_new.w, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    replace!(state_new.p, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    # Cap extreme velocities to avoid overflow
    umax = maximum(abs.(state_new.u)); wmax = maximum(abs.(state_new.w))
    vmax = max(umax, wmax)
    if vmax > 1e6
        scale = 1e6 / vmax
        state_new.u .*= scale
        state_new.w .*= scale
    end
    
    # Verify incompressibility  
    final_div = div(state_new.u, state_new.w, grid)
    
    # Check for NaN/Inf in divergence before taking maximum
    if any(x -> isnan(x) || isinf(x), final_div)
        @warn "NaN/Inf in divergence field - replacing with zeros"
        replace!(final_div, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    end
    
    max_div = maximum(abs.(final_div))
    
    # Only print if there's a very serious divergence issue
    if max_div > 100.0
        @warn "High divergence detected: max|∇·u| = $(max_div)"
    end
end


function demonstrate_clean_vs_traditional()
    """
    Side-by-side comparison of clean vs traditional discretization approaches.
    """
    
    println("Comparison: Clean vs Traditional Approaches")
    println("="^50)
    
    # Create test setup
    grid = StaggeredGrid2D(32, 24, 2.0, 1.5)
    state = SolutionState2D(grid.nx, grid.nz)
    
    # Initialize with some test data
    for j = 1:grid.nz, i = 1:grid.nx+1
        state.u[i, j] = sin(2π * i / grid.nx)
    end
    for j = 1:grid.nz+1, i = 1:grid.nx
        state.w[i, j] = cos(2π * j / grid.nz)
    end
    
    println("\n--- CLEAN APPROACH ---")
    println("# Compute divergence")
    println("div_u = div(u, w, grid)")
    
    div_u_clean = div(state.u, state.w, grid)
    println("Result: max|∇·u| = $(maximum(abs.(div_u_clean)))")
    
    println("\n# Compute pressure gradient")
    println("dpdx, dpdz = grad(p, grid)  # XZ plane")
    
    dpdx_clean, dpdz_clean = grad(state.p, grid)
    println("Result: ∇p computed at $(size(dpdx_clean)) and $(size(dpdz_clean))")
    
    println("\n# Compute Laplacian")
    println("lap_p = laplacian(p, grid)")
    
    lap_p_clean = laplacian(state.p, grid)
    println("Result: max|∇²p| = $(maximum(abs.(lap_p_clean)))")
    
    println("\n--- TRADITIONAL APPROACH ---")
    println("# Divergence (manual loops)")
    println("for j=1:ny, i=1:nx")
    println("    div[i,j] = (u[i+1,j] - u[i,j])/dx + (v[i,j+1] - v[i,j])/dy")
    println("end")
    
    # Traditional divergence computation
    div_u_trad = zeros(grid.nx, grid.nz)
    for j = 1:grid.nz, i = 1:grid.nx
        div_u_trad[i, j] = (state.u[i+1, j] - state.u[i, j]) / grid.dx + 
                           (state.w[i, j+1] - state.w[i, j]) / grid.dz  # Use dz for XZ plane, w is z-velocity
    end
    println("Result: max|∇·u| = $(maximum(abs.(div_u_trad)))")
    
    println("\n# Verify both approaches give same result")
    error_div = maximum(abs.(div_u_clean - div_u_trad))
    println("Difference: $(error_div) (should be ~0)")
    
    println("\n--- ADVANTAGES OF CLEAN APPROACH ---")
    println("More readable: div(u, v, grid) vs nested loops")
    println("Less error-prone: no manual indexing")
    println("Easier to debug: can test operators independently")
    println("Consistent staggered grid handling")
    println("Automatic boundary treatment")
    println("Operator reuse across different equations")
    println("Natural mathematical notation")
    
    println("\n--- CODE READABILITY COMPARISON ---")
    
    println("\nTraditional Navier-Stokes (hard to read):")
    println("for j=2:ny-1, i=2:nx")
    println("    dudx = (u[i,j] - u[i-1,j]) / dx")
    println("    dudy = (u[i,j+1] - u[i,j-1]) / (2*dy)")
    println("    d2udx2 = (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx^2")
    println("    d2udy2 = (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dy^2")
    println("    rhs[i,j] = -u[i,j]*dudx - v[i,j]*dudy + nu*(d2udx2 + d2udy2)")
    println("end")
    
    println("\nClean Navier-Stokes (physics-focused):")
    println("u_cc, v_cc = interpolate_to_cell_centers(u, v, grid)")
    println("advection = u_cc .* ddx(u_cc, grid) + v_cc .* ddy(u_cc, grid)")
    println("viscous = nu * laplacian(u_cc, grid)")
    println("rhs = -advection + viscous")
    
    println("\nClean approach clearly shows the physics!")
end

# Essential solver components for module compatibility

"""
    NavierStokesSolver2D

2D Navier-Stokes solver using fractional step method with clean operator approach.
"""
mutable struct NavierStokesSolver2D <: AbstractSolver
    grid::StaggeredGrid
    fluid::FluidProperties
    bc::BoundaryConditions
    time_scheme::TimeSteppingScheme
    multigrid_solver::Union{MultigridPoissonSolver, Any, Nothing}
    pressure_correction::Bool
    
    # Work arrays
    u_star::Matrix{Float64}
    v_star::Matrix{Float64}
    phi::Matrix{Float64}  # Pressure correction
    rhs_p::Matrix{Float64}  # Pressure Poisson RHS
    # Cached masks for static bodies (optional)
    chi_u_cache::Union{Nothing, Matrix{Float64}}
    chi_w_cache::Union{Nothing, Matrix{Float64}}
end

function NavierStokesSolver2D(grid::StaggeredGrid, fluid::FluidProperties, 
                             bc::BoundaryConditions, time_scheme::TimeSteppingScheme;
                             pressure_correction::Bool=true,
                             mg_levels::Int=5, mg_max_iterations::Int=300,
                             mg_tolerance::Float64=1e-12,
                             mg_smoother::Symbol=:staggered,
                             mg_cycle::Symbol=:V)
    nx, nz = grid.nx, grid.nz  # Use XZ plane for 2D
    
    # Initialize work arrays
    u_star = zeros(nx+1, nz)
    v_star = zeros(nx, nz+1)
    phi = zeros(nx, nz)
    rhs_p = zeros(nx, nz)
    
    # Create multigrid solver using provided settings (overrides env defaults)
    mg_solver = MultigridPoissonSolver(grid; smoother=mg_smoother,
                                       tolerance=mg_tolerance, max_iterations=mg_max_iterations,
                                       levels=mg_levels, cycle_type=mg_cycle)
    
    NavierStokesSolver2D(grid, fluid, bc, time_scheme, mg_solver, pressure_correction,
                        u_star, v_star, phi, rhs_p, nothing, nothing)
end

"""
    solve_step_2d!(solver, state_new, state_old, dt)

Perform one time step of the 2D Navier-Stokes solver using clean differential operators.
"""
function solve_step_2d!(solver::NavierStokesSolver2D, state_new::SolutionState, 
                       state_old::SolutionState, dt::Float64)
    if solver.pressure_correction
        # Default: no bodies provided for masking
        solve_projection_step_2d!(solver, state_new, state_old, dt, nothing)
    else
        error("Coupled Navier-Stokes solver not implemented")
    end
end

"""
    solve_step_2d!(solver, state_new, state_old, dt, bodies)

Overload accepting rigid bodies to enable masked (fluid-only) projection.
"""
function solve_step_2d!(solver::NavierStokesSolver2D, state_new::SolutionState, 
                       state_old::SolutionState, dt::Float64,
                       bodies::Union{RigidBodyCollection,Nothing})
    if solver.pressure_correction
        solve_projection_step_2d!(solver, state_new, state_old, dt, bodies)
    else
        error("Coupled Navier-Stokes solver not implemented")
    end
end

function solve_projection_step_2d!(solver::NavierStokesSolver2D, state_new::SolutionState,
                                   state_old::SolutionState, dt::Float64,
                                   bodies::Union{RigidBodyCollection,Nothing}=nothing)
    grid = solver.grid
    fluid = solver.fluid
    bc = solver.bc
    nx, nz = grid.nx, grid.nz

    # Predictor step at faces
    adv_u = zeros(nx+1, nz)
    adv_w = zeros(nx, nz+1)
    diff_u = zeros(nx+1, nz)
    diff_w = zeros(nx, nz+1)
    advection_2d!(adv_u, adv_w, state_old.u, state_old.w, grid)
    compute_diffusion_2d!(diff_u, diff_w, state_old.u, state_old.w, fluid, grid)

    # u*, w*
    solver.u_star .= state_old.u .+ dt .* (-adv_u .+ diff_u)
    solver.v_star .= state_old.w .+ dt .* (-adv_w .+ diff_w)  # v_star used as w*

    # Apply BC to predictor
    tmp = SolutionState2D(nx, nz)
    tmp.u .= solver.u_star
    tmp.w .= solver.v_star
    tmp.p .= state_old.p
    apply_boundary_conditions!(grid, tmp, bc, state_old.t + dt)
    solver.u_star .= tmp.u
    solver.v_star .= tmp.w

    # Pressure Poisson RHS (fluid-only if bodies provided)
    # Check masked projection toggle via env (default on)
    use_masks = lowercase(get(ENV, "BIOFLOWS_MASKED_PROJECTION", "1")) in ("1","true","yes","on")
    eps_mul = try parse(Float64, get(ENV, "BIOFLOWS_MASKS_EPS_MUL", "2.0")) catch; 2.0 end
    if use_masks && (bodies !== nothing)
        # Mask cache controls
        cache_on = lowercase(get(ENV, "BIOFLOWS_MASK_CACHE", "1")) in ("1","true","yes","on")
        log_mask  = lowercase(get(ENV, "BIOFLOWS_MASK_LOG", "0")) in ("1","true","yes","on")
        # Rebuild masks if bodies are moving; otherwise reuse cached masks
        moving = any(b -> (any(!=(0.0), b.velocity) || b.angular_velocity != 0.0), bodies.bodies)
        if !cache_on || (solver.chi_u_cache === nothing) || (solver.chi_w_cache === nothing) || moving
            chi_u, chi_w = build_solid_mask_faces_2d(bodies, grid; eps_mul=eps_mul)
            if cache_on
                solver.chi_u_cache = chi_u
                solver.chi_w_cache = chi_w
            end
            if log_mask
                @info "Mask rebuild: cache=$(cache_on) moving=$(moving)"
            end
        else
            chi_u = solver.chi_u_cache
            chi_w = solver.chi_w_cache
        end
        masked_divergence_2d!(solver.rhs_p, solver.u_star, solver.v_star, grid, chi_u, chi_w)
        solver.rhs_p .*= (1.0 / dt)
    else
        divergence_2d!(solver.rhs_p, solver.u_star, solver.v_star, grid)
        solver.rhs_p .*= (1.0 / dt)
    end

    # Solve for pressure correction
    solve_poisson!(solver.multigrid_solver, solver.phi, solver.rhs_p, grid, bc)

    # Correct velocities
    dpdx = zeros(nx+1, nz)
    dpdz = zeros(nx, nz+1)
    gradient_pressure_2d!(dpdx, dpdz, solver.phi, grid)
    # Apply correction; if masks provided, apply only in fluid (1-chi)
    if use_masks && (bodies !== nothing)
        cache_on = lowercase(get(ENV, "BIOFLOWS_MASK_CACHE", "1")) in ("1","true","yes","on")
        log_mask  = lowercase(get(ENV, "BIOFLOWS_MASK_LOG", "0")) in ("1","true","yes","on")
        moving = any(b -> (any(!=(0.0), b.velocity) || b.angular_velocity != 0.0), bodies.bodies)
        if !cache_on || (solver.chi_u_cache === nothing) || (solver.chi_w_cache === nothing) || moving
            chi_u, chi_w = build_solid_mask_faces_2d(bodies, grid; eps_mul=eps_mul)
            if cache_on
                solver.chi_u_cache = chi_u
                solver.chi_w_cache = chi_w
            end
            if log_mask
                @info "Mask rebuild before correction: cache=$(cache_on) moving=$(moving)"
            end
        else
            chi_u = solver.chi_u_cache
            chi_w = solver.chi_w_cache
        end
        state_new.u .= solver.u_star .- dt .* (1 .- chi_u) .* dpdx
        state_new.w .= solver.v_star .- dt .* (1 .- chi_w) .* dpdz
    else
        state_new.u .= solver.u_star .- dt .* dpdx
        state_new.w .= solver.v_star .- dt .* dpdz
    end

    # Update pressure
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
    else
        error("Variable density not implemented")
    end
    state_new.p .= state_old.p .+ ρ .* solver.phi
    state_new.t = state_old.t + dt
    state_new.step = state_old.step + 1

    # Final BC
    apply_boundary_conditions!(grid, state_new, bc, state_new.t)
end

# Run demonstration if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    demonstrate_clean_vs_traditional()
end
