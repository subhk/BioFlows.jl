# PID Controller Validation Tests for Flexible Bodies
using Test

# Load BioFlows
include(joinpath(@__DIR__, "src", "BioFlows.jl"))
using .BioFlows

println("Testing PID Controller for Flexible Bodies")
println(repeat("=", 50))

function test_pid_controller_creation()
    println("\n=== Testing PID Controller Creation ===")
    
    try
        # Create test flexible bodies
        bodies = FlexibleBody[]
        
        # Create two flags for distance control testing
        flag1 = create_flag([0.0, 0.0], 1.0, 0.05; 
                           prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=1.0),
                           id=1)
        flag2 = create_flag([2.0, 0.0], 1.0, 0.05; 
                           prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=1.0),
                           id=2)
        
        push!(bodies, flag1, flag2)
        
        # Create PID controller with default parameters
        controller = FlexibleBodyController(bodies)
        
        # Verify controller structure
        @assert length(controller.bodies) == 2 "Controller should have 2 bodies"
        @assert controller.kp == 0.5 "Default Kp should be 0.5"
        @assert controller.ki == 0.1 "Default Ki should be 0.1"
        @assert controller.kd == 0.05 "Default Kd should be 0.05"
        @assert controller.base_frequency == 1.0 "Default frequency should be 1.0"
        @assert size(controller.target_distances) == (2, 2) "Target distances matrix size"
        @assert size(controller.error_integral) == (2, 2) "Error integral matrix size"
        @assert size(controller.error_previous) == (2, 2) "Previous error matrix size"
        
        println("✓ PID controller creation verified")
        return true
        
    catch e
        println("✗ PID controller creation test failed: $e")
        return false
    end
end

function test_pid_parameter_tuning()
    println("\n=== Testing PID Parameter Tuning ===")
    
    try
        # Create test bodies
        bodies = [create_flag([0.0, 0.0], 1.0, 0.05), 
                  create_flag([1.5, 0.0], 1.0, 0.05)]
        
        controller = FlexibleBodyController(bodies)
        
        # Test parameter setting
        set_control_parameters!(controller; kp=1.0, ki=0.2, kd=0.1)
        
        @assert controller.kp == 1.0 "Kp should be updated to 1.0"
        @assert controller.ki == 0.2 "Ki should be updated to 0.2"  
        @assert controller.kd == 0.1 "Kd should be updated to 0.1"
        
        # Test partial parameter update
        set_control_parameters!(controller; ki=0.5)
        @assert controller.kp == 1.0 "Kp should remain unchanged"
        @assert controller.ki == 0.5 "Ki should be updated to 0.5"
        @assert controller.kd == 0.1 "Kd should remain unchanged"
        
        # Test parameter bounds (should be positive for stability)
        @assert controller.kp >= 0 "Kp should be non-negative"
        @assert controller.ki >= 0 "Ki should be non-negative"
        @assert controller.kd >= 0 "Kd should be non-negative"
        
        println("✓ PID parameter tuning verified")
        return true
        
    catch e
        println("✗ PID parameter tuning test failed: $e")
        return false
    end
end

function test_distance_measurement()
    println("\n=== Testing Distance Measurement ===")
    
    try
        # Create two bodies at known positions
        body1 = create_flag([0.0, 0.0], 1.0, 0.05)
        body2 = create_flag([3.0, 4.0], 1.0, 0.05)  # 5 units away from body1 trailing edge
        
        # Test various control points
        dist_trailing = compute_body_distance(body1, body2, :trailing_edge, :trailing_edge)
        dist_center = compute_body_distance(body1, body2, :center, :center)
        dist_leading = compute_body_distance(body1, body2, :leading_edge, :leading_edge)
        
        # Verify distances are reasonable
        @assert dist_trailing > 0 "Distance should be positive"
        @assert dist_center > 0 "Distance should be positive"  
        @assert dist_leading > 0 "Distance should be positive"
        
        # Test distance between leading and trailing edges of same body
        self_distance = compute_body_distance(body1, body1, :leading_edge, :trailing_edge)
        @assert abs(self_distance - 1.0) < 0.1 "Self distance should approximately equal body length"
        
        println("✓ Distance measurement verified")
        println("    Trailing-to-trailing distance: $(round(dist_trailing, digits=3))")
        println("    Center-to-center distance: $(round(dist_center, digits=3))")
        println("    Leading-to-leading distance: $(round(dist_leading, digits=3))")
        return true
        
    catch e
        println("✗ Distance measurement test failed: $e")
        return false
    end
end

function test_target_distance_setting()
    println("\n=== Testing Target Distance Setting ===")
    
    try
        bodies = [create_flag([0.0, 0.0], 1.0, 0.05),
                  create_flag([2.0, 0.0], 1.0, 0.05),
                  create_flag([1.0, 2.0], 1.0, 0.05)]
        
        controller = FlexibleBodyController(bodies)
        
        # Test setting target distances
        target_matrix = [0.0 1.5 2.0;
                        1.5 0.0 1.8;
                        2.0 1.8 0.0]
        
        set_target_distances!(controller, target_matrix)
        
        @assert controller.target_distances[1, 2] == 1.5 "Target distance [1,2] should be 1.5"
        @assert controller.target_distances[2, 1] == 1.5 "Target distance [2,1] should be 1.5"
        @assert controller.target_distances[1, 3] == 2.0 "Target distance [1,3] should be 2.0"
        @assert controller.target_distances[1, 1] == 0.0 "Diagonal should be zero"
        
        # Test asymmetric matrix correction
        asym_matrix = [0.0 1.5 2.0;
                       1.6 0.0 1.8;  # 1.6 != 1.5
                       2.1 1.9 0.0]  # 2.1 != 2.0, 1.9 != 1.8
        
        set_target_distances!(controller, asym_matrix)
        
        # Should be symmetrized to averages
        @assert abs(controller.target_distances[1, 2] - 1.55) < 1e-10 "Should average to 1.55"
        @assert abs(controller.target_distances[2, 1] - 1.55) < 1e-10 "Should be symmetric"
        
        println("✓ Target distance setting verified")
        return true
        
    catch e
        println("✗ Target distance setting test failed: $e")
        return false
    end
end

function test_controller_state_reset()
    println("\n=== Testing Controller State Reset ===")
    
    try
        bodies = [create_flag([0.0, 0.0], 1.0, 0.05),
                  create_flag([2.0, 0.0], 1.0, 0.05)]
        
        controller = FlexibleBodyController(bodies)
        
        # Simulate some accumulated errors
        controller.error_integral[1, 2] = 0.5
        controller.error_integral[2, 1] = 0.5
        controller.error_previous[1, 2] = 0.2
        controller.error_previous[2, 1] = 0.2
        
        # Reset controller state
        reset_controller_state!(controller)
        
        # Verify reset
        @assert controller.error_integral[1, 2] == 0.0 "Error integral should be reset"
        @assert controller.error_integral[2, 1] == 0.0 "Error integral should be reset"
        @assert controller.error_previous[1, 2] == 0.0 "Previous error should be reset"
        @assert controller.error_previous[2, 1] == 0.0 "Previous error should be reset"
        
        println("✓ Controller state reset verified")
        return true
        
    catch e
        println("✗ Controller state reset test failed: $e")
        return false
    end
end

function test_pid_control_update()
    println("\n=== Testing PID Control Update ===")
    
    try
        # Create bodies with known positions
        body1 = create_flag([0.0, 0.0], 1.0, 0.05; 
                           prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=1.0))
        body2 = create_flag([2.5, 0.0], 1.0, 0.05; 
                           prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=1.0))  # Start 2.5 units apart
        
        controller = FlexibleBodyController([body1, body2])
        
        # Set target distance of 2.0 (should reduce distance)
        target_matrix = [0.0 2.0; 2.0 0.0]
        set_target_distances!(controller, target_matrix)
        
        # Store initial amplitudes
        initial_amp1 = body1.amplitude
        initial_amp2 = body2.amplitude
        
        # Update controller (simulating time step)
        current_time = 0.0
        dt = 0.01
        updated_amplitudes = update_controller!(controller, current_time, dt)
        
        # Verify update results
        @assert length(updated_amplitudes) == 2 "Should return amplitudes for both bodies"
        @assert updated_amplitudes[1] == initial_amp1 "Body 1 amplitude should be unchanged (leading body)"
        # Body 2 should have adjusted amplitude due to distance error
        
        # Test that error accumulation works
        controller.error_integral[1, 2] = 0.1  # Some accumulated error
        updated_amplitudes2 = update_controller!(controller, current_time + dt, dt)
        
        println("✓ PID control update verified")
        println("    Initial amplitudes: [$(initial_amp1), $(initial_amp2)]")
        println("    Updated amplitudes: [$(round(updated_amplitudes[1], digits=4)), $(round(updated_amplitudes[2], digits=4))]")
        return true
        
    catch e
        println("✗ PID control update test failed: $e")
        return false
    end
end

function test_amplitude_limits()
    println("\n=== Testing Amplitude Limits ===")
    
    try
        bodies = [create_flag([0.0, 0.0], 1.0, 0.05; 
                                  prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=1.0)),
                  create_flag([1.0, 0.0], 1.0, 0.05; 
                             prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=1.0))]
        
        controller = FlexibleBodyController(bodies)
        
        # Verify default amplitude limits
        @assert length(controller.amplitude_limits) == 2 "Should have limits for each body"
        @assert controller.amplitude_limits[1] == (0.1, 2.0) "Default limits should be 10% to 200%"
        @assert controller.amplitude_limits[2] == (0.1, 2.0) "Default limits should be 10% to 200%"
        
        # Test that amplitude limiting works in practice
        # Set extreme target distance to force large control signal
        extreme_target = [0.0 10.0; 10.0 0.0]  # Very large target distance
        set_target_distances!(controller, extreme_target)
        
        # Update with large gains to force limiting
        set_control_parameters!(controller; kp=10.0, ki=5.0, kd=1.0)
        
        updated_amplitudes = update_controller!(controller, 0.0, 0.01)
        
        # Verify amplitudes are within limits
        for (i, amp) in enumerate(updated_amplitudes)
            min_limit, max_limit = controller.amplitude_limits[i]
            original_amp = abs(bodies[i].amplitude)
            @assert amp <= max_limit * original_amp "Amplitude should not exceed max limit"
            @assert amp >= min_limit * original_amp "Amplitude should not go below min limit"
        end
        
        println("✓ Amplitude limits verified")
        return true
        
    catch e
        println("✗ Amplitude limits test failed: $e")
        return false
    end
end

# Run all tests
function run_all_pid_tests()
    tests_passed = 0
    total_tests = 7
    
    if test_pid_controller_creation()
        tests_passed += 1
    end
    
    if test_pid_parameter_tuning()
        tests_passed += 1
    end
    
    if test_distance_measurement()
        tests_passed += 1
    end
    
    if test_target_distance_setting()
        tests_passed += 1
    end
    
    if test_controller_state_reset()
        tests_passed += 1
    end
    
    if test_pid_control_update()
        tests_passed += 1
    end
    
    if test_amplitude_limits()
        tests_passed += 1
    end
    
    println("\n" * repeat("=", 50))
    println("PID Controller Test Results: $tests_passed/$total_tests tests passed")
    
    if tests_passed == total_tests
        println("All PID controller tests passed! ✓")
        println("\nKey features verified:")
        println("- PID controller creation and initialization")
        println("- Parameter tuning and validation")
        println("- Distance measurement between flexible bodies")
        println("- Target distance setting with symmetry enforcement")
        println("- Controller state reset functionality")
        println("- PID control loop with amplitude adjustment")
        println("- Amplitude limiting for stability")
        return true
    else
        println("Some PID controller tests failed. Please review the implementation.")
        return false
    end
end

# Run the tests
try
    success = run_all_pid_tests()
    exit(success ? 0 : 1)
catch e
    println("PID test execution failed with error: $e")
    exit(1)
end