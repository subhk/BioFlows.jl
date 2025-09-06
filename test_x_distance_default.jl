# Test X-Distance Default Parameter for PID Controller
using Test

# Load BioFlows
include(joinpath(@__DIR__, "src", "BioFlows.jl"))
using .BioFlows

println("Testing X-Distance as Default Parameter")
println(repeat("=", 50))

function test_default_x_distance()
    println("\n=== Testing Default X-Distance Control ===")
    
    try
        # Create two flags with vertical separation (same x, different z)
        flag1 = create_flag([2.0, 1.0], 1.0, 0.05; 
                           prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=1.0))
        flag2 = create_flag([2.0, 3.0], 1.0, 0.05;  # Same x-position, different z
                           prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=1.0))
        
        # Create controller with defaults
        controller = FlexibleBodyController([flag1, flag2])
        
        # Verify defaults
        @assert controller.control_points[1] == :leading_edge "Default should be leading edge control"
        @assert controller.control_points[2] == :leading_edge "Default should be leading edge control"
        @assert controller.distance_type == :x_distance "Default should be x_distance"
        
        println("✓ Default parameters verified:")
        println("  Control points: $(controller.control_points)")
        println("  Distance type: $(controller.distance_type)")
        
        # Test distance measurement
        x_dist = compute_body_distance(flag1, flag2, :leading_edge, :leading_edge; distance_type=:x_distance)
        z_dist = compute_body_distance(flag1, flag2, :leading_edge, :leading_edge; distance_type=:z_distance)
        euclidean_dist = compute_body_distance(flag1, flag2, :leading_edge, :leading_edge; distance_type=:euclidean)
        
        println("  Distance measurements:")
        println("    X-distance: $(x_dist)")
        println("    Z-distance: $(z_dist)")  
        println("    Euclidean: $(euclidean_dist)")
        
        # Verify x-distance is 0 (same x-coordinate)
        @assert abs(x_dist) < 1e-10 "X-distance should be ~0 for vertically aligned flags"
        @assert abs(z_dist - 2.0) < 0.1 "Z-distance should be ~2.0"
        @assert abs(euclidean_dist - 2.0) < 0.1 "Euclidean distance should be ~2.0"
        
        println("✓ Distance measurements verified")
        return true
        
    catch e
        println("✗ Default x-distance test failed: $e")
        return false
    end
end

function test_x_distance_control()
    println("\n=== Testing X-Distance PID Control ===")
    
    try
        # Create flags with horizontal separation
        flag1 = create_flag([1.0, 2.0], 1.0, 0.05; 
                           prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=1.0))
        flag2 = create_flag([4.0, 2.0], 1.0, 0.05;  # Horizontal separation
                           prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=1.0))
        
        controller = FlexibleBodyController([flag1, flag2])
        
        # Set target x-distance of 2.0 (current is 3.0)
        target_distance = 2.0
        set_target_distances!(controller, [0.0 target_distance; target_distance 0.0])
        
        println("Setup:")
        println("  Initial x-separation: 3.0")
        println("  Target x-separation: $(target_distance)")
        
        # Simulate control
        dt = 0.01
        current_time = 0.0
        
        initial_distance = compute_body_distance(flag1, flag2, :leading_edge, :leading_edge; distance_type=:x_distance)
        
        for step = 1:200  # 2 seconds
            updated_amplitudes = update_controller!(controller, current_time, dt)
            apply_harmonic_boundary_conditions!(controller, current_time, dt)
            current_time += dt
        end
        
        final_distance = compute_body_distance(flag1, flag2, :leading_edge, :leading_edge; distance_type=:x_distance)
        error = abs(target_distance - final_distance)
        
        println("Results:")
        println("  Initial x-distance: $(round(initial_distance, digits=3))")
        println("  Final x-distance: $(round(final_distance, digits=3))")
        println("  Target: $(target_distance)")
        println("  Error: $(round(error, digits=3))")
        
        # Verify control is working (error should be reduced)
        initial_error = abs(target_distance - initial_distance)
        @assert error <= initial_error "Control should reduce or maintain error"
        @assert error < 1.5 "Final error should be reasonable"
        
        println("✓ X-distance PID control working")
        return true
        
    catch e
        println("✗ X-distance control test failed: $e")
        return false
    end
end

function test_distance_type_options()
    println("\n=== Testing Distance Type Options ===")
    
    try
        # Create flags at diagonal positions
        flag1 = create_flag([1.0, 1.0], 1.0, 0.05)
        flag2 = create_flag([4.0, 5.0], 1.0, 0.05)
        
        # Test all distance types
        x_dist = compute_body_distance(flag1, flag2, :leading_edge, :leading_edge; distance_type=:x_distance)
        z_dist = compute_body_distance(flag1, flag2, :leading_edge, :leading_edge; distance_type=:z_distance)
        eucl_dist = compute_body_distance(flag1, flag2, :leading_edge, :leading_edge; distance_type=:euclidean)
        
        # Expected values
        expected_x = 3.0  # |4-1|
        expected_z = 4.0  # |5-1|
        expected_eucl = 5.0  # sqrt(3²+4²)
        
        @assert abs(x_dist - expected_x) < 0.1 "X-distance should be 3.0"
        @assert abs(z_dist - expected_z) < 0.1 "Z-distance should be 4.0"
        @assert abs(eucl_dist - expected_eucl) < 0.1 "Euclidean distance should be 5.0"
        
        println("✓ All distance types working:")
        println("  X-distance: $(x_dist) (expected: $(expected_x))")
        println("  Z-distance: $(z_dist) (expected: $(expected_z))")
        println("  Euclidean: $(eucl_dist) (expected: $(expected_eucl))")
        
        # Test controller with different distance types
        for dist_type in [:x_distance, :z_distance, :euclidean]
            controller = FlexibleBodyController([flag1, flag2]; distance_type=dist_type)
            @assert controller.distance_type == dist_type "Controller should use specified distance type"
            println("  ✓ Controller with $(dist_type) created successfully")
        end
        
        return true
        
    catch e
        println("✗ Distance type options test failed: $e")
        return false
    end
end

# Run all tests
function run_all_x_distance_tests()
    tests_passed = 0
    total_tests = 3
    
    if test_default_x_distance()
        tests_passed += 1
    end
    
    if test_x_distance_control()
        tests_passed += 1
    end
    
    if test_distance_type_options()
        tests_passed += 1
    end
    
    println("\n" * repeat("=", 50))
    println("X-Distance Test Results: $tests_passed/$total_tests tests passed")
    
    if tests_passed == total_tests
        println("All X-distance tests passed! ✓")
        println("\nKey changes verified:")
        println("- Default control point: :leading_edge")
        println("- Default distance type: :x_distance") 
        println("- PID controller uses x-distance by default")
        println("- All distance types (:x_distance, :z_distance, :euclidean) work")
        println("- Controller can be configured with different distance types")
        return true
    else
        println("Some X-distance tests failed.")
        return false
    end
end

# Run the tests
try
    success = run_all_x_distance_tests()
    exit(success ? 0 : 1)
catch e
    println("X-distance test execution failed with error: $e")
    exit(1)
end