# Single Flexible Flag — Positions Only NetCDF Output
#
# Run with:
#   julia --project examples/single_flag_positions.jl

using BioFlows

function main()
    # Simulation config (we won't solve the fluid here; just use grid metadata)
    config = create_2d_simulation_config(
        nx=64, nz=32,
        Lx=2.0, Lz=1.0,
        dt=0.002, final_time=0.2,
        output_interval=0.01,
        output_file="single_flag_positions",
    )

    # Create a single flexible flag with prescribed sinusoidal motion
    flag = create_flag([0.4, 0.5], 0.4, 0.015;
                       n_points=25,
                       prescribed_motion=(type=:sinusoidal, amplitude=0.02, frequency=2.0))

    bodies = FlexibleBodyCollection()
    BioFlows.add_flexible_body!(bodies, flag)

    # Get a grid from a solver instance (for metadata only)
    solver = create_solver(config)
    grid = solver.grid

    # Create a lightweight NetCDF writer for body positions only
    writer = BioFlows.create_position_only_writer("single_flag_positions.nc", grid, bodies;
                                                  time_interval=config.output_config.time_interval,
                                                  save_mode=:time_interval)

    # Minimal time loop: apply boundary conditions and save positions
    t = 0.0
    iter = 0
    Δt = config.dt
    Tfinal = config.final_time

    println("Saving single flag positions to NetCDF...")
    while t <= Tfinal + 1e-12
        iter += 1
        # Update flag boundary condition (leading-edge sinusoidal motion)
        BioFlows.apply_boundary_conditions!(flag, t)
        # Save only positions
        BioFlows.save_body_positions_only!(writer, bodies, t, iter)
        t += Δt
    end

    BioFlows.close!(writer)
    println("Done. File: single_flag_positions.nc")
end

main()
