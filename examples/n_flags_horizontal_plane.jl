"""
N-Flags Horizontal Plane Distance Control Examples

This file demonstrates coordinated control of N-number of flags where flags
on the same horizontal plane maintain specified distances during oscillation.

Examples:
1. 4 flags on single horizontal plane
2. 6 flags on two horizontal planes  
3. 8 flags on three horizontal planes
4. Large system with 12+ flags
5. Dynamic reconfiguration of distances
6. Performance analysis for large N
"""

using BioFlows
using Printf

# =================================================================
# EXAMPLE 1: 4 FLAGS ON SINGLE HORIZONTAL PLANE
# =================================================================

"""
    example_four_flags_single_plane()

Four flags on same horizontal plane maintaining equal spacing.
"""
function example_four_flags_single_plane()
    println("🎌 Example 1: Four Flags on Single Horizontal Plane")
    println("=" ^ 55)
    
    # Grid setup
    grid = create_uniform_2d_grid(200, 80, 5.0, 2.0)
    
    # Four flags at same z-level with different x-positions
    flag_configs = [
        (start_point=[0.5, 1.0], length=0.6, width=0.04, 
         material=:flexible,
         prescribed_motion=(type=:sinusoidal, amplitude=0.10, frequency=2.0)),
        
        (start_point=[1.3, 1.0], length=0.5, width=0.035,
         material=:flexible, 
         prescribed_motion=(type=:sinusoidal, amplitude=0.08, frequency=2.0)),
        
        (start_point=[2.0, 1.0], length=0.45, width=0.03,
         material=:very_flexible,
         prescribed_motion=(type=:sinusoidal, amplitude=0.06, frequency=2.0)),
        
        (start_point=[2.6, 1.0], length=0.4, width=0.025,
         material=:very_flexible,
         prescribed_motion=(type=:sinusoidal, amplitude=0.05, frequency=2.0))
    ]
    
    # All flags on plane 1 (z=1.0) with 0.6 unit target separation
    target_separations = Dict(1 => 0.6)
    
    # Print configuration analysis
    print_horizontal_plane_analysis(flag_configs, target_separations)
    
    # Create coordinated system
    flags, controller, groups = setup_horizontal_plane_system(
        flag_configs, target_separations;
        base_frequency = 2.0,
        phase_coordination = :sequential,  # Sequential phases for wave effect
        kp = 0.8, ki = 0.15, kd = 0.08,
        control_scale_factor = 0.12
    )
    
    # Print system summary
    print_system_summary(flags, controller)
    
    # Simulate distance control
    println("\n🔄 Simulating distance control...")
    dt = 0.001
    
    # Track distances over time
    distance_history = []
    
    for step in 1:50
        current_time = step * dt
        
        # Apply control
        apply_harmonic_boundary_conditions!(controller, current_time, dt)
        
        # Measure current distances every 10 steps
        if step % 10 == 0
            distances = []
            # Measure consecutive flag distances
            for i in 1:3
                dist = compute_body_distance(flags.bodies[i], flags.bodies[i+1], 
                                           :trailing_edge, :trailing_edge)
                push!(distances, dist)
            end
            push!(distance_history, distances)
            
            @printf "  t=%.3f: distances=[%.4f, %.4f, %.4f], target=0.6\n" current_time distances[1] distances[2] distances[3]
        end
    end
    
    # Final analysis
    if !isempty(distance_history)
        final_distances = distance_history[end]
        errors = [abs(d - 0.6) for d in final_distances]
        @printf "\n📊 Final performance:\n"
        @printf "   Distance errors: [%.4f, %.4f, %.4f]\n" errors[1] errors[2] errors[3]
        @printf "   Max error: %.4f (%.1f%%)\n" maximum(errors) maximum(errors)/0.6*100
        @printf "   RMS error: %.4f\n" sqrt(mean(errors.^2))
    end
    
    println("✅ Four-flag single plane example completed!")
    return flags, controller, groups, distance_history
end

# =================================================================
# EXAMPLE 2: 6 FLAGS ON TWO HORIZONTAL PLANES
# =================================================================

"""
    example_six_flags_two_planes()

Six flags distributed on two horizontal planes.
"""
function example_six_flags_two_planes()
    println("\n🎌 Example 2: Six Flags on Two Horizontal Planes")
    println("=" ^ 55)
    
    # Flags on two different horizontal levels
    flag_configs = [
        # Upper plane (z = 1.2) - 4 flags
        (start_point=[0.4, 1.2], length=0.5, width=0.03,
         prescribed_motion=(type=:sinusoidal, amplitude=0.08, frequency=2.5)),
        (start_point=[1.0, 1.2], length=0.45, width=0.03,
         prescribed_motion=(type=:sinusoidal, amplitude=0.07, frequency=2.5)),
        (start_point=[1.5, 1.2], length=0.4, width=0.03,
         prescribed_motion=(type=:sinusoidal, amplitude=0.06, frequency=2.5)),
        (start_point=[2.0, 1.2], length=0.35, width=0.03,
         prescribed_motion=(type=:sinusoidal, amplitude=0.05, frequency=2.5)),
        
        # Lower plane (z = 0.8) - 2 flags  
        (start_point=[0.7, 0.8], length=0.4, width=0.025,
         prescribed_motion=(type=:sinusoidal, amplitude=0.06, frequency=2.5)),
        (start_point=[1.3, 0.8], length=0.35, width=0.025,
         prescribed_motion=(type=:sinusoidal, amplitude=0.05, frequency=2.5))
    ]
    
    # Different separations for each plane
    target_separations = Dict(
        1 => 0.45,  # Upper plane: 0.45 unit separation
        2 => 0.55   # Lower plane: 0.55 unit separation
    )
    
    # Analyze configuration
    print_horizontal_plane_analysis(flag_configs, target_separations)
    
    # Create system with alternating phases
    flags, controller, groups = setup_horizontal_plane_system(
        flag_configs, target_separations;
        base_frequency = 2.5,
        phase_coordination = :alternating,  # Upper/lower planes alternate
        kp = 0.7, ki = 0.12, kd = 0.06
    )
    
    print_system_summary(flags, controller)
    
    # Monitor distance control for both planes
    println("\n🔄 Monitoring both planes...")
    dt = 0.001
    
    for step in 1:40
        current_time = step * dt
        apply_harmonic_boundary_conditions!(controller, current_time, dt)
        
        if step % 15 == 0
            println("\\n  t=$(round(current_time, digits=3)):")
            
            # Upper plane distances (flags 1-4)
            upper_distances = []
            for i in 1:3
                dist = compute_body_distance(flags.bodies[i], flags.bodies[i+1],
                                           :trailing_edge, :trailing_edge)
                push!(upper_distances, dist)
            end
            @printf "    Upper plane: [%.3f, %.3f, %.3f] (target: 0.45)\\n" upper_distances[1] upper_distances[2] upper_distances[3]
            
            # Lower plane distance (flags 5-6)
            lower_distance = compute_body_distance(flags.bodies[5], flags.bodies[6],
                                                  :trailing_edge, :trailing_edge)
            @printf "    Lower plane: [%.3f] (target: 0.55)\\n" lower_distance
        end
    end
    
    println("✅ Six-flag two-plane example completed!")
    return flags, controller, groups
end

# =================================================================
# EXAMPLE 3: 8 FLAGS ON THREE HORIZONTAL PLANES
# =================================================================

"""
    example_eight_flags_three_planes()

Eight flags distributed across three horizontal planes.
"""
function example_eight_flags_three_planes()
    println("\n🎌 Example 3: Eight Flags on Three Horizontal Planes") 
    println("=" ^ 55)
    
    # Three horizontal planes with different numbers of flags
    flag_configs = [
        # Top plane (z = 1.4) - 2 flags
        (start_point=[0.8, 1.4], length=0.4, width=0.025),
        (start_point=[1.4, 1.4], length=0.35, width=0.025),
        
        # Middle plane (z = 1.0) - 4 flags  
        (start_point=[0.3, 1.0], length=0.5, width=0.03),
        (start_point=[0.9, 1.0], length=0.45, width=0.03),
        (start_point=[1.4, 1.0], length=0.4, width=0.03),
        (start_point=[1.8, 1.0], length=0.35, width=0.03),
        
        # Bottom plane (z = 0.6) - 2 flags
        (start_point=[0.6, 0.6], length=0.35, width=0.02),
        (start_point=[1.1, 0.6], length=0.3, width=0.02)
    ]
    
    # Different target separations for each plane
    target_separations = Dict(
        1 => 0.5,   # Top plane
        2 => 0.4,   # Middle plane
        3 => 0.45   # Bottom plane
    )
    
    # Validate and analyze
    is_valid, messages = validate_horizontal_plane_configuration(flag_configs, target_separations)
    
    if !is_valid
        println("❌ Configuration validation failed:")
        for msg in messages
            if startswith(msg, "ERROR")
                println("   $msg")
            end
        end
        return nothing
    end
    
    print_horizontal_plane_analysis(flag_configs, target_separations)
    
    # Create system with sequential phases
    flags, controller, groups = setup_horizontal_plane_system(
        flag_configs, target_separations;
        base_frequency = 1.8,
        phase_coordination = :sequential,
        kp = 0.6, ki = 0.1, kd = 0.05
    )
    
    print_system_summary(flags, controller)
    
    # Test distance measurements for all planes
    println("\n📏 Initial distance measurements:")
    for (plane_idx, group) in enumerate(groups)
        if length(group) > 1
            println("   Plane $plane_idx:")
            for i in 1:(length(group)-1)
                flag_i, flag_j = group[i], group[i+1]
                dist = compute_body_distance(flags.bodies[flag_i], flags.bodies[flag_j],
                                           :trailing_edge, :trailing_edge)
                target = target_separations[plane_idx]
                @printf "     Flag %d ↔ Flag %d: %.4f (target: %.3f, error: %.4f)\\n" flag_i flag_j dist target abs(dist - target)
            end
        end
    end
    
    println("✅ Eight-flag three-plane example completed!")
    return flags, controller, groups
end

# =================================================================
# EXAMPLE 4: LARGE SYSTEM (12+ FLAGS)
# =================================================================

"""
    example_large_n_flag_system(n_flags::Int = 12)

Large system with N flags for performance testing.
"""
function example_large_n_flag_system(n_flags::Int = 12)
    println("\n🎌 Example 4: Large N-Flag System (N = $n_flags)")
    println("=" ^ 55)
    
    if n_flags < 4
        println("❌ Need at least 4 flags for meaningful large system test")
        return nothing
    end
    
    # Distribute flags across multiple horizontal planes
    n_planes = max(2, div(n_flags, 4))  # 2-4 flags per plane typically
    plane_z_coords = range(0.6, 1.4, length=n_planes)
    
    println("📋 Generating $n_flags flags across $n_planes horizontal planes...")
    
    flag_configs = []
    target_separations = Dict{Int, Float64}()
    
    flags_per_plane = div(n_flags, n_planes)
    remaining_flags = n_flags % n_planes
    
    flag_idx = 1
    for (plane_idx, z_coord) in enumerate(plane_z_coords)
        # Number of flags for this plane
        n_flags_this_plane = flags_per_plane + (plane_idx <= remaining_flags ? 1 : 0)
        
        # X-positions for this plane
        x_positions = range(0.3, 3.0, length=n_flags_this_plane)
        
        # Target separation (varies by plane)
        separation = 0.4 + 0.1 * (plane_idx - 1)  # 0.4, 0.5, 0.6, ...
        target_separations[plane_idx] = separation
        
        @printf "   Plane %d (z=%.2f): %d flags, separation=%.2f\\n" plane_idx z_coord n_flags_this_plane separation
        
        for x_pos in x_positions
            # Varying properties
            length = 0.3 + 0.2 * rand()  # 0.3 to 0.5
            width = 0.02 + 0.015 * rand()  # 0.02 to 0.035
            amplitude = 0.04 + 0.06 * rand()  # 0.04 to 0.10
            
            config = (
                start_point = [x_pos, z_coord],
                length = length,
                width = width,
                prescribed_motion = (type=:sinusoidal, amplitude=amplitude, frequency=2.0)
            )
            
            push!(flag_configs, config)
            flag_idx += 1
        end
    end
    
    # Validate large system
    is_valid, messages = validate_horizontal_plane_configuration(flag_configs, target_separations)
    
    if !is_valid
        println("❌ Large system validation failed")
        return nothing
    end
    
    # Create system (use moderate control gains for stability)
    println("\n🔧 Creating large coordinated system...")
    flags, controller, groups = setup_horizontal_plane_system(
        flag_configs, target_separations;
        base_frequency = 1.5,
        phase_coordination = :sequential,
        kp = 0.4, ki = 0.08, kd = 0.04,  # Conservative gains for large system
        control_scale_factor = 0.08
    )
    
    print_system_summary(flags, controller)
    
    # Performance test
    println("\n⚡ Performance testing large system...")
    
    dt = 0.001
    start_time = time()
    
    for step in 1:30
        current_time = step * dt
        apply_harmonic_boundary_conditions!(controller, current_time, dt)
    end
    
    elapsed_time = time() - start_time
    control_updates_per_second = 30 / elapsed_time
    
    @printf "   Performance: %.1f control updates/second\\n" control_updates_per_second
    @printf "   Time per update: %.3f ms\\n" elapsed_time * 1000 / 30
    
    # Distance control quality check
    println("\n📊 Distance control quality:")
    all_errors = Float64[]
    
    for (plane_idx, group) in enumerate(groups)
        if length(group) > 1
            target = target_separations[plane_idx]
            plane_errors = Float64[]
            
            for i in 1:(length(group)-1)
                flag_i, flag_j = group[i], group[i+1]
                current_dist = compute_body_distance(flags.bodies[flag_i], flags.bodies[flag_j],
                                                   :trailing_edge, :trailing_edge)
                error = abs(current_dist - target)
                push!(plane_errors, error)
                push!(all_errors, error)
            end
            
            @printf "   Plane %d: max_error=%.4f, rms_error=%.4f\\n" plane_idx maximum(plane_errors) sqrt(mean(plane_errors.^2))
        end
    end
    
    if !isempty(all_errors)
        @printf "   Overall: max_error=%.4f, rms_error=%.4f\\n" maximum(all_errors) sqrt(mean(all_errors.^2))
    end
    
    println("✅ Large system example completed!")
    return flags, controller, groups
end

# =================================================================
# EXAMPLE 5: DYNAMIC DISTANCE RECONFIGURATION
# =================================================================

"""
    example_dynamic_reconfiguration()

Demonstrate dynamic reconfiguration of target distances during simulation.
"""
function example_dynamic_reconfiguration()
    println("\n🎌 Example 5: Dynamic Distance Reconfiguration")
    println("=" ^ 55)
    
    # Start with 4 flags
    flag_configs = [
        (start_point=[0.4, 1.0], length=0.4, width=0.03,
         prescribed_motion=(type=:sinusoidal, amplitude=0.06, frequency=2.0)),
        (start_point=[1.0, 1.0], length=0.4, width=0.03,
         prescribed_motion=(type=:sinusoidal, amplitude=0.06, frequency=2.0)),
        (start_point=[1.6, 1.0], length=0.4, width=0.03,
         prescribed_motion=(type=:sinusoidal, amplitude=0.06, frequency=2.0)),
        (start_point=[2.2, 1.0], length=0.4, width=0.03,
         prescribed_motion=(type=:sinusoidal, amplitude=0.06, frequency=2.0))
    ]
    
    # Initial target separation
    initial_separations = Dict(1 => 0.5)
    
    flags, controller, groups = setup_horizontal_plane_system(
        flag_configs, initial_separations;
        kp = 0.8, ki = 0.15, kd = 0.08
    )
    
    println("🔄 Dynamic reconfiguration test:")
    dt = 0.001
    
    # Phase 1: Initial separation (0.5)
    println("\\n   Phase 1: Target separation = 0.5")
    for step in 1:20
        current_time = step * dt
        apply_harmonic_boundary_conditions!(controller, current_time, dt)
    end
    
    # Measure distances
    distances_phase1 = []
    for i in 1:3
        dist = compute_body_distance(flags.bodies[i], flags.bodies[i+1], :trailing_edge, :trailing_edge)
        push!(distances_phase1, dist)
    end
    @printf "     Achieved distances: [%.4f, %.4f, %.4f]\\n" distances_phase1[1] distances_phase1[2] distances_phase1[3]
    
    # Phase 2: Change to tighter spacing (0.35)
    println("\\n   Phase 2: Changing target separation to 0.35")
    new_distance_matrix = create_horizontal_distance_matrix(flag_configs, Dict(1 => 0.35))[1]
    set_target_distances!(controller, new_distance_matrix)
    reset_controller_state!(controller)  # Reset integral terms for clean transition
    
    for step in 1:30
        current_time = (20 + step) * dt
        apply_harmonic_boundary_conditions!(controller, current_time, dt)
    end
    
    distances_phase2 = []
    for i in 1:3
        dist = compute_body_distance(flags.bodies[i], flags.bodies[i+1], :trailing_edge, :trailing_edge)
        push!(distances_phase2, dist)
    end
    @printf "     Achieved distances: [%.4f, %.4f, %.4f]\\n" distances_phase2[1] distances_phase2[2] distances_phase2[3]
    
    # Phase 3: Change to looser spacing (0.7)
    println("\\n   Phase 3: Changing target separation to 0.7")
    new_distance_matrix = create_horizontal_distance_matrix(flag_configs, Dict(1 => 0.7))[1]
    set_target_distances!(controller, new_distance_matrix)
    reset_controller_state!(controller)
    
    for step in 1:30
        current_time = (50 + step) * dt
        apply_harmonic_boundary_conditions!(controller, current_time, dt)
    end
    
    distances_phase3 = []
    for i in 1:3
        dist = compute_body_distance(flags.bodies[i], flags.bodies[i+1], :trailing_edge, :trailing_edge)
        push!(distances_phase3, dist)
    end
    @printf "     Achieved distances: [%.4f, %.4f, %.4f]\\n" distances_phase3[1] distances_phase3[2] distances_phase3[3]
    
    # Analysis
    println("\\n📊 Reconfiguration performance:")
    errors_1 = [abs(d - 0.5) for d in distances_phase1]
    errors_2 = [abs(d - 0.35) for d in distances_phase2] 
    errors_3 = [abs(d - 0.7) for d in distances_phase3]
    
    @printf "   Phase 1 errors (target 0.5): max=%.4f, rms=%.4f\\n" maximum(errors_1) sqrt(mean(errors_1.^2))
    @printf "   Phase 2 errors (target 0.35): max=%.4f, rms=%.4f\\n" maximum(errors_2) sqrt(mean(errors_2.^2))
    @printf "   Phase 3 errors (target 0.7): max=%.4f, rms=%.4f\\n" maximum(errors_3) sqrt(mean(errors_3.^2))
    
    println("✅ Dynamic reconfiguration example completed!")
    return flags, controller, [distances_phase1, distances_phase2, distances_phase3]
end

# =================================================================
# MAIN RUNNER FUNCTION
# =================================================================

"""
    run_all_n_flag_examples()

Run all N-flag horizontal plane examples.
"""
function run_all_n_flag_examples()
    println("🚀 Running All N-Flag Horizontal Plane Examples")
    println("=" ^ 65)
    
    results = []
    
    # Example 1: 4 flags, single plane
    result1 = example_four_flags_single_plane()
    push!(results, result1)
    
    # Example 2: 6 flags, two planes
    result2 = example_six_flags_two_planes()
    push!(results, result2)
    
    # Example 3: 8 flags, three planes
    result3 = example_eight_flags_three_planes()
    push!(results, result3)
    
    # Example 4: Large system (12 flags)
    result4 = example_large_n_flag_system(12)
    push!(results, result4)
    
    # Example 5: Dynamic reconfiguration
    result5 = example_dynamic_reconfiguration()
    push!(results, result5)
    
    println("\n🎉 All N-Flag Examples Completed Successfully!")
    println("\n📝 Summary:")
    println("   ✅ Single plane control (4 flags)")
    println("   ✅ Multi-plane control (6 flags on 2 planes)")  
    println("   ✅ Complex multi-plane (8 flags on 3 planes)")
    println("   ✅ Large system performance (12+ flags)")
    println("   ✅ Dynamic reconfiguration capabilities")
    
    println("\n🔑 Key Capabilities Demonstrated:")
    println("   • Automatic detection of horizontal planes")
    println("   • N-flag distance control with arbitrary N")
    println("   • Multi-plane coordination")
    println("   • Performance scaling for large systems")
    println("   • Dynamic distance reconfiguration")
    println("   • Robust validation and error handling")
    
    return results
end

# =================================================================
# PERFORMANCE ANALYSIS FUNCTIONS
# =================================================================

"""
    benchmark_n_flag_performance(n_values::Vector{Int} = [4, 8, 12, 16, 20])

Benchmark performance for different numbers of flags.
"""
function benchmark_n_flag_performance(n_values::Vector{Int} = [4, 8, 12, 16, 20])
    println("\n⚡ N-Flag Performance Benchmark")
    println("=" ^ 40)
    
    results = []
    
    for n in n_values
        println("\\n   Testing N = $n flags:")
        
        # Generate configuration
        flag_configs = []
        for i in 1:n
            z_coord = 1.0 + 0.3 * (i % 3)  # 3 different z-levels
            x_coord = 0.3 + (i-1) * 0.4
            
            config = (
                start_point = [x_coord, z_coord],
                length = 0.4,
                width = 0.03,
                prescribed_motion = (type=:sinusoidal, amplitude=0.06, frequency=2.0)
            )
            push!(flag_configs, config)
        end
        
        # Setup system
        target_seps = Dict(1 => 0.5, 2 => 0.45, 3 => 0.55)
        flags, controller, groups = setup_horizontal_plane_system(flag_configs, target_seps;
                                                                 kp=0.4, ki=0.08, kd=0.04)
        
        # Benchmark control updates
        dt = 0.001
        n_steps = 50
        
        start_time = time()
        for step in 1:n_steps
            current_time = step * dt
            apply_harmonic_boundary_conditions!(controller, current_time, dt)
        end
        elapsed_time = time() - start_time
        
        updates_per_second = n_steps / elapsed_time
        time_per_update = elapsed_time * 1000 / n_steps
        
        @printf "     Updates/sec: %.1f, Time/update: %.3f ms\\n" updates_per_second time_per_update
        
        push!(results, (n_flags=n, updates_per_sec=updates_per_second, time_per_update=time_per_update))
    end
    
    println("\\n📊 Performance Summary:")
    @printf "   %-8s %-15s %-15s\\n" "N Flags" "Updates/sec" "Time/update (ms)"
    @printf "   %-8s %-15s %-15s\\n" "-------" "-----------" "----------------"
    for result in results
        @printf "   %-8d %-15.1f %-15.3f\\n" result.n_flags result.updates_per_sec result.time_per_update
    end
    
    return results
end

# Run examples when file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_n_flag_examples()
end