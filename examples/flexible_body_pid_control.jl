# Flexible Body PID Control Example
# Two flexible flags with PID controller maintaining target distance
# Run: julia --project examples/flexible_body_pid_control.jl

using BioFlows
using NetCDF

function main()
    println("=== Flexible Body PID Control Example ===")
    
    # Domain parameters
    nx, nz = 160, 80
    Lx, Lz = 8.0, 4.0
    
    # Flow parameters
    Uin = 1.0
    ρ = 1000.0
    ν = 0.01  # Higher viscosity for clearer flexible body effects
    dt = 0.005
    Tfinal = 20.0
    
    println("Setting up simulation with flexible bodies...")
    
    # Create simulation configuration
    config = create_2d_simulation_config(
        nx = nx, nz = nz,
        Lx = Lx, Lz = Lz,
        density_value = ρ,
        nu = ν,
        inlet_velocity = Uin,
        outlet_type = :pressure,
        wall_type = :no_slip,
        dt = dt,
        final_time = Tfinal,
        adaptive_refinement = false,
        output_interval = 0.2,
        output_file = "flexible_pid_control",
        output_max_snapshots = 100
    )
    
    # Create two flexible flags at different heights
    flag1_pos = [2.0, 1.5]  # Lower flag
    flag2_pos = [2.0, 2.5]  # Upper flag (initial distance = 1.0)
    flag_length = 1.5
    flag_width = 0.05
    
    # Create flexible flags with prescribed motion
    flag1 = create_flag(flag1_pos, flag_length, flag_width;
                        material = :flexible,
                        prescribed_motion = (type = :sinusoidal, amplitude = 0.2, frequency = 1.5),
                        attachment = :fixed_leading_edge,
                        n_points = 15)
    
    flag2 = create_flag(flag2_pos, flag_length, flag_width;
                        material = :flexible,
                        prescribed_motion = (type = :sinusoidal, amplitude = 0.2, frequency = 1.5),
                        attachment = :fixed_leading_edge,
                        n_points = 15)
    
    println("Flag 1 positioned at: $(flag1_pos)")
    println("Flag 2 positioned at: $(flag2_pos)")
    println("Initial distance: $(norm(flag2_pos - flag1_pos))")
    
    # Create flexible body collection
    flexible_bodies = FlexibleBodyCollection()
    add_flexible_body!(flexible_bodies, flag1)
    add_flexible_body!(flexible_bodies, flag2)
    
    # Create PID controller with x-distance control (default)
    controller = FlexibleBodyController([flag1, flag2];
                                       base_frequency = 1.5,
                                       kp = 0.6,   # Proportional gain
                                       ki = 0.15,  # Integral gain  
                                       kd = 0.08,  # Derivative gain
                                       phase_coordination = :synchronized,
                                       distance_type = :x_distance)  # Horizontal separation control
    
    # Set target distance of 1.5 units (20% increase from initial)
    target_distance = 1.5
    target_matrix = [0.0 target_distance; target_distance 0.0]
    set_target_distances!(controller, target_matrix)
    
    println("PID Controller configured:")
    println("  Target distance: $(target_distance)")
    println("  Distance type: $(controller.distance_type) (horizontal separation)")
    println("  Control points: $(controller.control_points) (leading edges)")
    println("  PID gains: Kp=$(controller.kp), Ki=$(controller.ki), Kd=$(controller.kd)")
    println("  Phase coordination: synchronized")
    
    # Add to configuration
    config = SimulationConfig(
        config.grid_type, config.nx, config.ny, config.nz,
        config.Lx, config.Ly, config.Lz, config.origin,
        config.fluid, config.bc, config.time_scheme,
        config.dt, config.final_time,
        config.rigid_bodies,
        flexible_bodies,        # Add flexible bodies
        controller,            # Add PID controller
        config.use_mpi, config.adaptive_refinement, config.refinement_criteria,
        config.output_config
    )
    
    # Create solver
    solver = create_solver(config)
    state0 = initialize_simulation(config, initial_conditions = :quiescent)
    
    println("\nStarting simulation with PID-controlled flexible bodies...")
    println("Watch the distance control in action!")
    
    # Run simulation with custom time stepping to show PID in action
    current_time = 0.0
    step = 0
    
    # Manual time loop to demonstrate PID control
    while current_time < Tfinal
        step += 1
        
        # Update PID controller
        updated_amplitudes = update_controller!(controller, current_time, dt)
        apply_harmonic_boundary_conditions!(controller, current_time, dt)
        
        # Monitor distance every 100 steps
        if step % 100 == 0
            current_distance = compute_body_distance(flag1, flag2, :leading_edge, :leading_edge; distance_type=:x_distance)
            error = abs(target_distance - current_distance)
            
            println("Step $step, t=$(round(current_time, digits=2)): " *
                   "distance=$(round(current_distance, digits=3)), " *
                   "error=$(round(error, digits=3)), " *
                   "amplitudes=[$(round.(updated_amplitudes, digits=3))]")
        end
        
        # Advance one time step (simplified - in real simulation this would be solver.step!)
        current_time += dt
        
        # Break early for demonstration
        if step > 2000  # About 10 seconds
            break
        end
    end
    
    # Final distance check
    final_distance = compute_body_distance(flag1, flag2, :leading_edge, :leading_edge; distance_type=:x_distance)
    final_error = abs(target_distance - final_distance)
    
    println("\n=== PID Control Results ===")
    println("Target distance: $(target_distance)")
    println("Final distance: $(round(final_distance, digits=4))")
    println("Final error: $(round(final_error, digits=4)) ($(round(100*final_error/target_distance, digits=2))%)")
    println("Final amplitudes: [$(round.(updated_amplitudes, digits=4))]")
    
    if final_error < 0.1
        println("PID control successful!")
    else
        println("WARNING: PID control needs tuning")
    end
    
    # Monitor PID performance
    performance = monitor_distance_control(controller, current_time)
    println("\nController Performance:")
    println("  RMS error: $(round(performance.rms_distance_error, digits=4))")
    println("  Max error: $(round(performance.max_distance_error, digits=4))")
    println("  Active controls: $(performance.n_active_controls)")
    
    return config, solver, controller
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    config, solver, controller = main()
    println("\nPID control example completed successfully!")
end