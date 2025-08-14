"""
Simple Coordinated Flags Examples

This file contains focused, easy-to-debug examples for the coordinated flag system.
Each example demonstrates specific aspects of the control system.

Examples:
1. Two flags with basic distance control
2. Three flags in chain configuration
3. Control parameter tuning demonstration
4. Distance measurement validation
5. Phase coordination strategies
"""

using BioFlow
using Printf

# =================================================================
# EXAMPLE 1: BASIC TWO-FLAG SYSTEM
# =================================================================

"""
    example_two_flags()

Simplest example: Two flags maintaining constant separation.
"""
function example_two_flags()
    println("🎌 Example 1: Basic Two-Flag Distance Control")
    println("=" ^ 50)
    
    # Simple grid
    grid = create_uniform_2d_grid(100, 50, 2.0, 1.0)
    
    # Flag configurations
    leading_flag = (
        start_point = [0.3, 0.5],
        length = 0.6,
        width = 0.03,
        material = :flexible,
        prescribed_motion = (type=:sinusoidal, amplitude=0.08, frequency=2.0)
    )
    
    trailing_flag = (
        start_point = [1.0, 0.5], 
        length = 0.5,
        width = 0.025,
        material = :flexible,
        prescribed_motion = (type=:sinusoidal, amplitude=0.06, frequency=2.0)
    )
    
    # Create coordinated system
    flags, controller = setup_simple_two_flag_system(
        leading_flag_config = leading_flag,
        trailing_flag_config = trailing_flag,
        target_separation = 0.4,  # Maintain 0.4 unit separation
        kp = 0.5, ki = 0.1, kd = 0.05,
        phase_coordination = :synchronized
    )
    
    # Print system summary
    print_system_summary(flags, controller)
    
    # Simulate a few time steps
    println("\n🔄 Running short simulation...")
    dt = 0.001
    
    for step in 1:20
        current_time = step * dt
        
        # Apply control
        apply_harmonic_boundary_conditions!(controller, current_time, dt)
        
        # Monitor every 5 steps
        if step % 5 == 0
            metrics = monitor_distance_control(controller, current_time)
            distance = compute_body_distance(flags.bodies[1], flags.bodies[2], 
                                           :trailing_edge, :trailing_edge)
            @printf "  t=%.3f: distance=%.4f, target=0.4, error=%.4f\n" current_time distance (0.4 - distance)
        end
    end
    
    println("✅ Two-flag example completed!")
    return flags, controller
end

# =================================================================
# EXAMPLE 2: THREE-FLAG CHAIN
# =================================================================

"""
    example_three_flag_chain()

Three flags in a chain with sequential control.
"""
function example_three_flag_chain()
    println("\n🎌 Example 2: Three-Flag Chain Configuration")
    println("=" ^ 50)
    
    # Flag configurations
    flag_configs = [
        (start_point=[0.2, 0.5], length=0.5, width=0.03, 
         prescribed_motion=(type=:sinusoidal, amplitude=0.06, frequency=1.5)),
        (start_point=[0.8, 0.5], length=0.4, width=0.025,
         prescribed_motion=(type=:sinusoidal, amplitude=0.05, frequency=1.5)),  
        (start_point=[1.3, 0.5], length=0.35, width=0.02,
         prescribed_motion=(type=:sinusoidal, amplitude=0.04, frequency=1.5))
    ]
    
    # Chain separations: 0.3 between flag1-flag2, 0.25 between flag2-flag3
    separations = [0.3, 0.25]
    
    # Create chain system
    flags, controller = setup_multi_flag_chain(
        flag_configs, separations;
        phase_coordination = :sequential,  # Sequential phases
        kp = 0.6, ki = 0.12, kd = 0.06
    )
    
    # Print configuration
    print_system_summary(flags, controller)
    
    # Test distance measurements
    println("\n📏 Initial distance measurements:")
    print_distance_analysis(flags.bodies, controller.control_points, controller.target_distances)
    
    println("✅ Three-flag chain example completed!")
    return flags, controller
end

# =================================================================
# EXAMPLE 3: CONTROL PARAMETER TUNING
# =================================================================

"""
    example_control_tuning()

Demonstrate the effect of different PID parameters.
"""
function example_control_tuning()
    println("\n🎌 Example 3: Control Parameter Tuning")
    println("=" ^ 50)
    
    # Base configuration
    flag_configs = [
        (start_point=[0.4, 0.5], length=0.5, width=0.03),
        (start_point=[1.0, 0.5], length=0.4, width=0.025)
    ]
    
    distance_matrix = [0.0 0.35; 0.35 0.0]
    
    # Test different PID settings
    pid_settings = [
        (kp=0.2, ki=0.05, kd=0.02, name="Conservative"),
        (kp=0.5, ki=0.1, kd=0.05, name="Moderate"), 
        (kp=1.0, ki=0.2, kd=0.1, name="Aggressive"),
        (kp=0.8, ki=0.0, kd=0.15, name="PD-only")
    ]
    
    println("\n🔧 Testing different PID parameters:")
    
    for (kp, ki, kd, name) in pid_settings
        println("\n   Testing $name gains (Kp=$kp, Ki=$ki, Kd=$kd):")
        
        # Create system with these gains
        flags, controller = create_coordinated_flag_system(
            flag_configs, distance_matrix;
            kp=kp, ki=ki, kd=kd
        )
        
        # Simulate several control steps
        dt = 0.001
        errors = Float64[]
        
        for step in 1:15
            current_time = step * dt
            apply_harmonic_boundary_conditions!(controller, current_time, dt)
            
            # Measure error
            current_dist = compute_body_distance(flags.bodies[1], flags.bodies[2],
                                               :trailing_edge, :trailing_edge)
            error = abs(0.35 - current_dist)
            push!(errors, error)
        end
        
        # Report performance
        final_error = errors[end]
        avg_error = mean(errors[end-5:end])  # Average of last 5 steps
        @printf "     Final error: %.5f, Avg error (last 5): %.5f\n" final_error avg_error
    end
    
    println("✅ Control tuning example completed!")
end

# =================================================================
# EXAMPLE 4: DISTANCE MEASUREMENT VALIDATION
# =================================================================

"""
    example_distance_validation()

Validate distance measurement functions with known geometries.
"""
function example_distance_validation()
    println("\n🎌 Example 4: Distance Measurement Validation")
    println("=" ^ 50)
    
    # Create simple test flags at known positions
    flag1 = create_flag([0.0, 0.0], 0.4, 0.02; n_points=5)
    flag2 = create_flag([1.0, 0.0], 0.3, 0.02; n_points=4)  # 1.0 units apart
    
    println("   Flag 1: start=[0.0, 0.0], length=0.4")
    println("   Flag 2: start=[1.0, 0.0], length=0.3")
    println("   Expected leading edge distance: 1.0")
    println("   Expected trailing edge distance: 1.0 - 0.4 + 0.3 = 0.9")
    
    # Test different measurement points
    measurements = [
        (:leading_edge, :leading_edge, "Leading to Leading"),
        (:trailing_edge, :trailing_edge, "Trailing to Trailing"),
        (:center, :center, "Center to Center"),
        (:leading_edge, :trailing_edge, "Leading to Trailing")
    ]
    
    println("\n📐 Distance measurements:")
    for (point1, point2, description) in measurements
        distance = compute_body_distance(flag1, flag2, point1, point2)
        @printf "   %s: %.4f\n" description distance
    end
    
    # Test multi-body distance matrix
    bodies = [flag1, flag2]
    control_points = [:trailing_edge, :trailing_edge]
    distance_matrix = compute_multi_body_distances(bodies, control_points)
    
    println("\n📊 Distance matrix:")
    display(distance_matrix)
    
    # Test bounding box calculations
    println("\n📦 Bounding boxes:")
    for (i, flag) in enumerate([flag1, flag2])
        bbox = compute_body_bounding_box(flag)
        @printf "   Flag %d: width=%.3f, height=%.3f, center=[%.3f, %.3f]\n" i bbox.width bbox.height (bbox.x_min + bbox.x_max)/2 (bbox.z_min + bbox.z_max)/2
    end
    
    println("✅ Distance validation example completed!")
    return flag1, flag2
end

# =================================================================
# EXAMPLE 5: PHASE COORDINATION STRATEGIES
# =================================================================

"""
    example_phase_coordination()

Demonstrate different phase coordination strategies.
"""
function example_phase_coordination()
    println("\n🎌 Example 5: Phase Coordination Strategies")
    println("=" ^ 50)
    
    # Standard configuration for testing
    flag_configs = [
        (start_point=[0.3, 0.5], length=0.4, width=0.02,
         prescribed_motion=(type=:sinusoidal, amplitude=0.05, frequency=2.0)),
        (start_point=[0.8, 0.5], length=0.4, width=0.02,
         prescribed_motion=(type=:sinusoidal, amplitude=0.05, frequency=2.0)),
        (start_point=[1.3, 0.5], length=0.4, width=0.02,
         prescribed_motion=(type=:sinusoidal, amplitude=0.05, frequency=2.0))
    ]
    
    distance_matrix = [0.0 0.25 0.5; 0.25 0.0 0.25; 0.5 0.25 0.0]
    
    strategies = [:synchronized, :alternating, :sequential]
    
    println("\n🎼 Testing phase coordination strategies:")
    
    for strategy in strategies
        println("\n   Strategy: $strategy")
        
        flags, controller = create_coordinated_flag_system(
            flag_configs, distance_matrix;
            phase_coordination = strategy,
            base_frequency = 2.0
        )
        
        # Show phase offsets
        @printf "     Phase offsets: [" 
        for (i, offset) in enumerate(controller.phase_offsets)
            @printf "%.2f" offset
            if i < length(controller.phase_offsets)
                @printf ", "
            end
        end
        @printf "] rad\n"
        
        # Show phase differences in degrees
        @printf "     Phase differences (degrees): ["
        for i in 2:length(controller.phase_offsets)
            diff_deg = (controller.phase_offsets[i] - controller.phase_offsets[1]) * 180 / π
            @printf "%.1f" diff_deg
            if i < length(controller.phase_offsets)
                @printf ", "
            end
        end
        @printf "]\n"
        
        # Show relative timing at t=0.1s
        t = 0.1
        @printf "     Relative positions at t=%.1fs: [" t
        for (i, offset) in enumerate(controller.phase_offsets)
            phase = 2π * controller.base_frequency * t + offset
            position = sin(phase)
            @printf "%.3f" position
            if i < length(controller.phase_offsets)
                @printf ", "
            end
        end
        @printf "]\n"
    end
    
    println("✅ Phase coordination example completed!")
end

# =================================================================
# MAIN RUNNER
# =================================================================

"""
    run_all_examples()

Run all simple coordinated flag examples.
"""
function run_all_examples()
    println("🚀 Running All Simple Coordinated Flag Examples")
    println("=" ^ 60)
    
    # Run each example
    flags1, controller1 = example_two_flags()
    flags2, controller2 = example_three_flag_chain() 
    example_control_tuning()
    flag1, flag2 = example_distance_validation()
    example_phase_coordination()
    
    println("\n🎉 All examples completed successfully!")
    println("   These examples demonstrate the key components of the coordinated flag system:")
    println("   • Distance-based PID control")
    println("   • Multiple coordination strategies") 
    println("   • Parameter tuning effects")
    println("   • Distance measurement accuracy")
    println("   • Phase coordination options")
    
    return (flags1, controller1), (flags2, controller2), (flag1, flag2)
end

# =================================================================
# INDIVIDUAL DEBUGGING FUNCTIONS
# =================================================================

"""
    debug_controller_step(controller::FlexibleBodyController, current_time::Float64, dt::Float64)

Step-by-step debugging of controller update process.
"""
function debug_controller_step(controller::FlexibleBodyController, current_time::Float64, dt::Float64)
    println("\n🔧 Debug Controller Step at t=$current_time:")
    
    n_bodies = length(controller.bodies)
    
    for i in 1:n_bodies
        for j in i+1:n_bodies
            if controller.target_distances[i, j] > 0.0
                println("\n   Control pair: Body $i → Body $j")
                
                # Measure current distance
                current_distance = compute_body_distance(
                    controller.bodies[i], controller.bodies[j],
                    controller.control_points[i], controller.control_points[j]
                )
                
                target = controller.target_distances[i, j]
                error = target - current_distance
                
                @printf "     Current distance: %.6f\n" current_distance
                @printf "     Target distance:  %.6f\n" target
                @printf "     Error:           %.6f\n" error
                
                # PID terms
                integral_term = controller.error_integral[i, j]
                derivative_term = (error - controller.error_previous[i, j]) / dt
                
                @printf "     Integral term:   %.6f\n" integral_term
                @printf "     Derivative term: %.6f\n" derivative_term
                
                # Control signal
                control_signal = (controller.kp * error + 
                                controller.ki * integral_term + 
                                controller.kd * derivative_term)
                
                @printf "     Control signal:  %.6f\n" control_signal
                
                # Amplitude adjustment
                amplitude_adjustment = control_signal * controller.control_scale_factor
                current_amplitude = controller.bodies[j].amplitude
                new_amplitude = current_amplitude + amplitude_adjustment
                
                @printf "     Current amplitude: %.6f\n" current_amplitude
                @printf "     Amplitude adjustment: %.6f\n" amplitude_adjustment
                @printf "     New amplitude: %.6f\n" new_amplitude
            end
        end
    end
end

"""
    test_single_control_pair()

Test control system with just one pair of flags for detailed debugging.
"""
function test_single_control_pair()
    println("\n🔬 Single Control Pair Test")
    println("=" ^ 40)
    
    # Minimal configuration
    flags, controller = setup_simple_two_flag_system(
        leading_flag_config = (start_point=[0.0, 0.0], length=0.3, width=0.02),
        trailing_flag_config = (start_point=[0.5, 0.0], length=0.2, width=0.02),
        target_separation = 0.4,
        kp = 0.5, ki = 0.1, kd = 0.05
    )
    
    # Step through several iterations with debugging
    dt = 0.001
    for step in 1:5
        current_time = step * dt
        
        println("\n" * "="^30 * " STEP $step " * "="^30)
        
        # Before control
        initial_distance = compute_body_distance(flags.bodies[1], flags.bodies[2],
                                                :trailing_edge, :trailing_edge)
        @printf "Before control: distance = %.6f\n" initial_distance
        
        # Debug the control step
        debug_controller_step(controller, current_time, dt)
        
        # Apply control
        apply_harmonic_boundary_conditions!(controller, current_time, dt)
        
        # After control  
        final_distance = compute_body_distance(flags.bodies[1], flags.bodies[2],
                                             :trailing_edge, :trailing_edge)
        @printf "After control: distance = %.6f\n" final_distance
        @printf "Distance change: %.6f\n" (final_distance - initial_distance)
    end
    
    return flags, controller
end

# Run examples when file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_examples()
end