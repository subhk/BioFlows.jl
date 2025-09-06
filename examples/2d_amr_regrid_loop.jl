# 2D AMR regridding loop (XZ plane)
# Run with: julia --project examples/2d_amr_regrid_loop.jl

using BioFlows

function main()
    # Base 2D config
    nx, nz = 64, 32
    Lx, Lz = 2.0, 1.0
    dt = 1e-3
    steps = 30

    config = BioFlows.create_2d_simulation_config(
        nx=nx, nz=nz, Lx=Lx, Lz=Lz,
        dt=dt, final_time=steps*dt,
        output_interval=0.05,
    )

    # Add a rigid circle to drive refinement
    config = BioFlows.add_rigid_circle!(config, [Lx/2, Lz/2], 0.12)

    # Create solver and initial state
    solver = BioFlows.create_solver(config)
    state_old = BioFlows.initialize_simulation(config; initial_conditions=:uniform_flow)
    state_new = deepcopy(state_old)

    # AMR criteria and integrated solver
    crit = BioFlows.AdaptiveRefinementCriteria(
        velocity_gradient_threshold=0.0,
        pressure_gradient_threshold=0.3,
        vorticity_threshold=1e9,
        body_distance_threshold=0.25,
        max_refinement_level=2,
        min_grid_size=min(solver.grid.dx, solver.grid.dz)/4,
    )

    amr_solver = BioFlows.create_amr_integrated_solver(solver, crit; amr_frequency=5)

    last_count = 0
    for step in 1:steps
        BioFlows.amr_solve_step!(amr_solver, state_new, state_old, dt, config.rigid_bodies)

        # Report refinement changes periodically
        if step % amr_solver.amr_frequency == 0
            count = length(amr_solver.refined_grid.refined_cells_2d)
            println("step=$(step): refined_cells=$(count) (Δ=$(count - last_count))")
            last_count = count
        end

        # Swap states
        state_old, state_new = state_new, state_old
    end

    println("Done. Final refined cells: ", length(amr_solver.refined_grid.refined_cells_2d))
end

main()
