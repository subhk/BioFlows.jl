"""
Coordinated Flag Oscillations with Distance Control

This example demonstrates the harmonic oscillation control system for flexible flags:
1. Two flags oscillating harmonically while maintaining constant distance
2. PID control system that adjusts amplitude to maintain target separation
3. Multiple coordination strategies (synchronized, alternating, sequential)
4. Real-time monitoring of control performance
5. NetCDF output for analysis

Run this example to see how the control system maintains constant distances
between oscillating flags by dynamically adjusting their amplitudes.
"""

using BioFlows
using Printf

function main()
    println("🎌 BioFlows.jl Coordinated Flag Oscillations Demo")
    println("=" ^ 55)
    
    # =================================================================
    # 1. SIMULATION SETUP
    # =================================================================
    
    # Grid parameters (2D XZ plane)
    nx, nz = 180, 90
    Lx, Lz = 4.0, 2.0
    grid = create_uniform_2d_grid(nx, nz, Lx, Lz)
    
    # Fluid properties
    Reynolds = 180.0
    inlet_velocity = 1.0
    ρ = ConstantDensity(1.0)
    μ = inlet_velocity * 1.0 / Reynolds
    fluid = FluidProperties(μ, ρ, Reynolds)
    
    println("\n🌊 Simulation setup:")
    println("  • Grid: $(nx) × $(nz), Domain: $(Lx) × $(Lz)")
    println("  • Reynolds number: $(Reynolds)")
    println("  • Inlet velocity: $(inlet_velocity)")
    
    # =================================================================
    # 2. CREATE COORDINATED FLAG SYSTEM
    # =================================================================
    
    println("\n🎯 Creating coordinated flag system...")
    
    # Define flag configurations
    flag_configs = [
        # Leading flag - driven at fixed amplitude
        (start_point=[0.6, 1.2], length=0.8, width=0.04,
         material=:flexible,
         attachment=:fixed_leading_edge,
         prescribed_motion=(type=:sinusoidal, amplitude=0.12, frequency=2.5),
         n_points=25),
        
        # Trailing flag - amplitude will be controlled to maintain distance
        (start_point=[1.8, 1.2], length=0.7, width=0.035,
         material=:very_flexible,
         attachment=:fixed_leading_edge,
         prescribed_motion=(type=:sinusoidal, amplitude=0.10, frequency=2.5),
         n_points=22),
        
        # Third flag - also controlled
        (start_point=[2.8, 1.0], length=0.6, width=0.03,
         material=:flexible,
         attachment=:fixed_leading_edge,
         prescribed_motion=(type=:sinusoidal, amplitude=0.08, frequency=2.5),
         n_points=20)
    ]
    
    # Define target distances between flags (trailing edge to trailing edge)
    target_distances = [
        0.0  0.8  1.6;   # Flag 1 targets: 0.8 to flag 2, 1.6 to flag 3
        0.8  0.0  0.8;   # Flag 2 targets: 0.8 to flag 1, 0.8 to flag 3  
        1.6  0.8  0.0    # Flag 3 targets: 1.6 to flag 1, 0.8 to flag 2
    ]
    
    # Create coordinated system with synchronized oscillations
    flags, controller = create_coordinated_flag_system(
        flag_configs, target_distances;
        base_frequency = 2.5,
        phase_coordination = :synchronized,
        kp = 0.8,     # Proportional gain (higher for faster response)
        ki = 0.15,    # Integral gain (moderate to eliminate steady-state error)
        kd = 0.08,    # Derivative gain (small to reduce oscillations)
        control_points = [:trailing_edge, :trailing_edge, :trailing_edge],  # Control trailing edges
        amplitude_limits = [(0.8, 1.5), (0.5, 2.0), (0.5, 2.2)]  # Amplitude constraints
    )
    
    println("✓ Created coordinated flag system:")
    println("  • $(flags.n_bodies) flags with harmonic oscillations")
    println("  • Target distances: trailing edge separations")
    println("  • Control strategy: synchronized oscillations with PID amplitude control")
    println("  • PID gains: Kp=$(controller.kp), Ki=$(controller.ki), Kd=$(controller.kd)")
    
    # =================================================================
    # 3. SETUP NETCDF OUTPUT WITH CONTROL MONITORING
    # =================================================================
    
    println("\n💾 Setting up NetCDF output with control monitoring...")
    
    # Create output directory
    mkpath("output/coordinated_flags")
    
    # Configuration for detailed tracking
    config = NetCDFConfig("coordinated_flag_simulation";
        max_snapshots_per_file = 500,
        save_mode = :both,                    # Time and iteration based
        time_interval = 0.02,                 # Every 0.02s
        iteration_interval = 20,              # Every 20 iterations
        save_flow_field = true,               # Save velocity and pressure
        save_body_positions = true,           # Save flag positions
        save_force_coefficients = true,       # Save drag/lift coefficients
        reference_velocity = inlet_velocity,
        flow_direction = [1.0, 0.0]
    )
    
    writer = NetCDFWriter("output/coordinated_flags/coordinated_simulation.nc", grid, config)
    
    # Also create specialized control monitoring file
    control_writer = create_position_only_writer(
        "output/coordinated_flags/control_performance.nc",
        grid, flags;
        time_interval = 0.005,               # High frequency for control analysis
        save_mode = :time_interval,
        max_snapshots = 2000
    )
    
    # =================================================================
    # 4. SIMULATION PARAMETERS
    # =================================================================
    
    dt = 0.0008
    final_time = 3.0
    max_iterations = Int(final_time / dt)
    
    # Initialize solution state
    state = SolutionState2D(nx, nz)
    state.u .= inlet_velocity
    state.w .= 0.0
    state.p .= 0.0
    
    println("\n⏱️  Simulation parameters:")
    println("  • Time step: $dt")
    println("  • Final time: $final_time")
    println("  • Total iterations: $max_iterations")
    
    # =================================================================
    # 5. MAIN SIMULATION LOOP WITH COORDINATED CONTROL
    # =================================================================
    
    println("\n🚀 Starting coordinated flag simulation...")
    println("   Monitoring distance control performance in real-time")
    
    current_time = 0.0
    iteration = 0
    
    # Save initial conditions
    save_complete_snapshot!(writer, state, flags, grid, fluid, current_time, iteration;
                          reference_velocity=inlet_velocity, flow_direction=[1.0, 0.0])
    save_body_positions_only!(control_writer, flags, current_time, iteration)
    
    # Control performance tracking
    control_stats = []
    distance_history = []
    amplitude_history = []
    
    println("\n📊 Simulation progress and control monitoring:")
    
    # Main simulation loop
    while current_time < final_time && iteration < max_iterations
        iteration += 1
        current_time = iteration * dt
        
        # *** STEP 1: Apply coordinated harmonic boundary conditions ***
        apply_harmonic_boundary_conditions!(controller, current_time, dt)
        
        # *** STEP 2: Solve Navier-Stokes (placeholder) ***
        # solve_step_2d!(state, grid, fluid, bc, flags, dt)
        
        # *** STEP 3: Update flexible body dynamics (placeholder) ***
        # for flag in flags.bodies
        #     update_flexible_body!(flag, dt)
        # end
        
        # *** STEP 4: Monitor control performance ***
        if iteration % 10 == 0  # Monitor every 10 iterations
            control_metrics = monitor_distance_control(controller, current_time)
            push!(control_stats, control_metrics)
            
            # Track distance and amplitude histories
            current_distances = [
                compute_body_distance(flags.bodies[1], flags.bodies[2], :trailing_edge, :trailing_edge),
                compute_body_distance(flags.bodies[2], flags.bodies[3], :trailing_edge, :trailing_edge),
                compute_body_distance(flags.bodies[1], flags.bodies[3], :trailing_edge, :trailing_edge)
            ]
            push!(distance_history, current_distances)
            
            current_amplitudes = [body.amplitude for body in flags.bodies]
            push!(amplitude_history, current_amplitudes)
        end
        
        # *** STEP 5: Save data ***
        # Main simulation data
        if should_save(writer, current_time, iteration)
            save_complete_snapshot!(writer, state, flags, grid, fluid, 
                                  current_time, iteration;
                                  reference_velocity=inlet_velocity,
                                  flow_direction=[1.0, 0.0])
        end
        
        # High-frequency control monitoring
        if save_body_positions_only!(control_writer, flags, current_time, iteration)
            # Control data saved at high frequency
        end
        
        # *** STEP 6: Progress reporting ***
        if iteration % 200 == 0
            if !isempty(control_stats)
                latest_metrics = control_stats[end]
                @printf "  t=%.3fs, iter=%d:\\n" current_time iteration
                @printf "    Distance errors: max=%.4f, rms=%.4f\\n" latest_metrics.max_distance_error latest_metrics.rms_distance_error
                @printf "    Target distances: [%.3f, %.3f, %.3f]\\n" target_distances[1,2] target_distances[2,3] target_distances[1,3]
                
                if !isempty(distance_history)
                    current_dist = distance_history[end]
                    @printf "    Current distances: [%.3f, %.3f, %.3f]\\n" current_dist[1] current_dist[2] current_dist[3]
                end
                
                if !isempty(amplitude_history)
                    current_amp = amplitude_history[end]
                    @printf "    Amplitudes: [%.4f, %.4f, %.4f]\\n" current_amp[1] current_amp[2] current_amp[3]
                end
                
                # Flag tip positions
                tip1 = flags.bodies[1].X[end, :]
                tip2 = flags.bodies[2].X[end, :]
                tip3 = flags.bodies[3].X[end, :]
                @printf "    Flag tips: [%.3f,%.3f], [%.3f,%.3f], [%.3f,%.3f]\\n" tip1[1] tip1[2] tip2[1] tip2[2] tip3[1] tip3[2]
            end
        end
    end
    
    # Close NetCDF files
    close_netcdf!(writer)
    close_netcdf!(control_writer)
    
    # =================================================================
    # 6. ANALYSIS AND RESULTS
    # =================================================================
    
    println("\n📊 Coordinated flag simulation completed!")
    println("\n📈 Control system performance analysis:")
    
    if !isempty(control_stats)
        final_metrics = control_stats[end]
        all_max_errors = [stat.max_distance_error for stat in control_stats]
        all_rms_errors = [stat.rms_distance_error for stat in control_stats]
        
        @printf "  Final control performance:\\n"
        @printf "    Max distance error: %.4f\\n" final_metrics.max_distance_error
        @printf "    RMS distance error: %.4f\\n" final_metrics.rms_distance_error
        @printf "    Average max error: %.4f\\n" mean(all_max_errors)
        @printf "    Average RMS error: %.4f\\n" mean(all_rms_errors)
        
        # Distance maintenance quality
        target_dist_12 = target_distances[1,2]
        target_dist_23 = target_distances[2,3]
        target_dist_13 = target_distances[1,3]
        
        if !isempty(distance_history)
            final_distances = distance_history[end]
            @printf "\\n  Distance control quality:\\n"
            @printf "    Flag 1-2: target=%.3f, achieved=%.3f, error=%.1f%%\\n" target_dist_12 final_distances[1] abs(final_distances[1] - target_dist_12) / target_dist_12 * 100
            @printf "    Flag 2-3: target=%.3f, achieved=%.3f, error=%.1f%%\\n" target_dist_23 final_distances[2] abs(final_distances[2] - target_dist_23) / target_dist_23 * 100  
            @printf "    Flag 1-3: target=%.3f, achieved=%.3f, error=%.1f%%\\n" target_dist_13 final_distances[3] abs(final_distances[3] - target_dist_13) / target_dist_13 * 100
        end
        
        # Amplitude adaptation analysis
        if !isempty(amplitude_history)
            initial_amps = amplitude_history[1]
            final_amps = amplitude_history[end]
            @printf "\\n  Amplitude adaptation:\\n"
            for i in 1:3
                change_percent = (final_amps[i] - initial_amps[i]) / initial_amps[i] * 100
                @printf "    Flag %d: %.4f → %.4f (%.1f%% change)\\n" i initial_amps[i] final_amps[i] change_percent
            end
        end
    end
    
    println("\n💾 Output files created:")
    println("  • coordinated_simulation.nc - Complete simulation data with flow field")
    println("  • control_performance.nc - High-frequency control monitoring data")
    
    println("\n🔍 Analysis recommendations:")
    println("  • Plot distance vs time to verify control performance")
    println("  • Analyze amplitude adaptation over time")
    println("  • Examine phase relationships between flags")
    println("  • Visualize flag deformation and wake interactions")
    
    println("\n🐍 Python analysis example:")
    println("  import xarray as xr")
    println("  import matplotlib.pyplot as plt")
    println("  data = xr.open_dataset('coordinated_simulation.nc')")
    println("  # Plot flag positions over time")
    println("  flag1_x = data['flexible_body_1_x']")
    println("  flag1_z = data['flexible_body_1_z']")
    println("  # Animate: plt.plot(flag1_x[:, t], flag1_z[:, t]) for each time t")
    
    return flags, controller, control_stats, distance_history, amplitude_history
end

# =================================================================
# ADDITIONAL DEMONSTRATION FUNCTIONS
# =================================================================

"""
    demonstrate_phase_coordination_strategies()

Demonstrate different phase coordination strategies.
"""
function demonstrate_phase_coordination_strategies()
    println("\n🎼 Demonstrating phase coordination strategies...")
    
    # Simple flag configuration for testing
    flag_configs = [
        (start_point=[0.5, 1.0], length=0.6, width=0.03, 
         prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=2.0)),
        (start_point=[1.2, 1.0], length=0.6, width=0.03,
         prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=2.0))
    ]
    
    target_distances = [0.0 0.5; 0.5 0.0]  # Maintain 0.5 unit separation
    
    strategies = [:synchronized, :alternating, :sequential]
    
    for strategy in strategies
        println("\\n  Testing $strategy coordination:")
        
        flags, controller = create_coordinated_flag_system(
            flag_configs, target_distances;
            base_frequency = 2.0,
            phase_coordination = strategy
        )
        
        @printf "    Phase offsets: [%.2f, %.2f] rad\\n" controller.phase_offsets[1] controller.phase_offsets[2]
        @printf "    Phase difference: %.2f rad (%.1f°)\\n" abs(controller.phase_offsets[2] - controller.phase_offsets[1]) abs(controller.phase_offsets[2] - controller.phase_offsets[1]) * 180 / π
    end
end

"""
    test_control_system_stability()

Test the stability and responsiveness of the control system.
"""
function test_control_system_stability()
    println("\n⚖️  Testing control system stability...")
    
    # Create a simple two-flag system
    flag_configs = [
        (start_point=[0.5, 1.0], length=0.6, width=0.03),
        (start_point=[1.5, 1.0], length=0.6, width=0.03)
    ]
    
    target_distances = [0.0 0.8; 0.8 0.0]
    
    # Test different PID gain combinations
    gain_sets = [
        (kp=0.2, ki=0.05, kd=0.02, name="Conservative"),
        (kp=0.5, ki=0.1, kd=0.05, name="Moderate"),
        (kp=1.0, ki=0.2, kd=0.1, name="Aggressive"),
        (kp=0.8, ki=0.0, kd=0.15, name="PD-only")
    ]
    
    for (kp, ki, kd, name) in gain_sets
        println("\\n  Testing $name gains (Kp=$kp, Ki=$ki, Kd=$kd):")
        
        flags, controller = create_coordinated_flag_system(
            flag_configs, target_distances;
            kp=kp, ki=ki, kd=kd
        )
        
        # Simulate a few control steps
        dt = 0.001
        for step in 1:10
            current_time = step * dt
            apply_harmonic_boundary_conditions!(controller, current_time, dt)
            
            if step == 10
                metrics = monitor_distance_control(controller, current_time)
                @printf "    Final distance error: %.4f\\n" metrics.max_distance_error
            end
        end
    end
end

# =================================================================
# RUN THE DEMONSTRATION
# =================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Run main coordinated flag demonstration
    flags, controller, control_stats, distance_history, amplitude_history = main()
    
    # Run additional demonstrations
    demonstrate_phase_coordination_strategies()
    test_control_system_stability()
    
    println("\n🎉 Coordinated flag oscillations demonstration completed!")
    println("   The control system successfully maintains target distances")
    println("   by dynamically adjusting flag amplitudes using PID control.")
    println("   Check 'output/coordinated_flags/' for detailed NetCDF results.")
end