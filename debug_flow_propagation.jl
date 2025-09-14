#!/usr/bin/env julia
# Diagnostic script to understand why flow is not propagating

using BioFlows

# Small test case parameters
const NX, NZ = 50, 20  # Small grid
const LX, LZ = 2.0, 1.0  # Small domain
const UIN, RHO, NU = 1.0, 1.0, 0.01  # Higher viscosity for stability
const DT, TFINAL = 0.001, 0.1  # Very small timestep, short simulation
const D = 0.1; const R = D/2
const XC, ZC = 0.5, 0.5

function debug_main()
    println("=== Flow Propagation Debug ===")
    println("Re=$(round(UIN*D/NU,digits=1)), grid=$(NX)x$(NZ), dt=$(DT)")

    # Simple configuration - no cylinder first
    config = create_2d_simulation_config(
        nx=NX, nz=NZ, Lx=LX, Lz=LZ,
        density_value=RHO, nu=NU,
        inlet_velocity=UIN,
        outlet_type=:pressure, wall_type=:no_slip,
        dt=DT, final_time=TFINAL,
        use_mpi=false, adaptive_refinement=false,
        output_interval=TFINAL+1, output_file="/tmp/debug_flow",
        poisson_max_iterations=100,
        poisson_tolerance=1e-8,
    )

    # No cylinder - just open channel
    solver = create_solver(config)
    state_old = initialize_simulation(config, initial_conditions=:quiescent)

    # Initialize with uniform flow
    println("Initial u field: min=$(minimum(state_old.u)), max=$(maximum(state_old.u))")
    println("Initial w field: min=$(minimum(state_old.w)), max=$(maximum(state_old.w))")

    state_new = deepcopy(state_old)
    nsteps = round(Int, TFINAL/DT)
    println("Running $(nsteps) steps...")

    for step = 1:nsteps
        # Store old maxes
        old_maxu = maximum(abs, state_old.u)
        old_maxw = maximum(abs, state_old.w)

        # Apply boundary conditions first
        BioFlows.apply_boundary_conditions!(solver.grid, state_old, config.bc, step * DT)

        # Check boundary application
        if step <= 3
            println("Step $step BEFORE solve:")
            println("  Left u boundary (inlet): $(state_old.u[1, 1:min(5,end)])")
            println("  Right u boundary: $(state_old.u[end, 1:min(5,end)])")
            println("  First interior u: $(state_old.u[2, 1:min(5,end)])")
        end

        # Single time step
        solve_step_2d!(solver, state_new, state_old, DT, config.rigid_bodies)

        # Check for issues
        new_maxu = maximum(abs, state_new.u)
        new_maxw = maximum(abs, state_new.w)

        if step <= 10 || step % 10 == 0
            println("Step $(step): max|u|=$(round(new_maxu,digits=6)) max|w|=$(round(new_maxw,digits=6))")

            # Check if flow is propagating
            interior_u = state_new.u[2:end-1, :]
            nonzero_interior = count(x -> abs(x) > 1e-10, interior_u)
            println("  Non-zero interior u cells: $nonzero_interior / $(length(interior_u))")

            # Check divergence
            u_div = zeros(NX, NZ)
            for j = 1:NZ, i = 1:NX
                if i < NX && j < NZ
                    dudx = (state_new.u[i+1, j] - state_new.u[i, j]) / solver.grid.dx
                    dwdz = (state_new.w[i, j+1] - state_new.w[i, j]) / solver.grid.dz
                    u_div[i, j] = dudx + dwdz
                end
            end
            max_div = maximum(abs, u_div)
            println("  Max divergence: $(round(max_div, digits=8))")
        end

        # Swap states
        state_old, state_new = state_new, state_old

        # Early exit if completely stagnant
        if step > 5 && new_maxu < 1e-12
            println("Flow appears stagnant at step $step")
            break
        end
    end

    println("\n=== Final State Analysis ===")
    println("Final max|u|: $(maximum(abs, state_old.u))")
    println("Final max|w|: $(maximum(abs, state_old.w))")

    # Analyze velocity profile
    println("\nVelocity profiles:")
    mid_z = div(NZ, 2)
    println("u at mid-height z=$mid_z:")
    for i = 1:min(10, NX)
        x = (i-0.5) * solver.grid.dx
        println("  x=$(round(x,digits=3)): u=$(round(state_old.u[i, mid_z],digits=6))")
    end
end

debug_main()