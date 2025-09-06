# AMR Implementation Validation Test
# Tests the corrected AMR implementation for consistency and correctness

using LinearAlgebra

# Load BioFlows with AMR
include(joinpath(@__DIR__, "src", "BioFlows.jl"))
using .BioFlows

println("Testing AMR Implementation Fixes...")

function test_amr_coordinate_consistency()
    println("\n=== Testing AMR Coordinate System Consistency ===")
    
    # Test 2D XZ case
    println("Testing 2D XZ coordinate system...")
    try
        # Create 2D XZ plane AMR level
        amr_level_2d = AMRLevel(TwoDimensional, 0, 16, 16, 0.1, 0.1, 0.0, 0.0)
        
        # Verify coordinates
        @assert length(amr_level_2d.x_centers) == 16 "X centers should have 16 elements"
        @assert length(amr_level_2d.z_centers) == 16 "Z centers should have 16 elements" 
        @assert length(amr_level_2d.y_centers) == 0 "Y centers should be empty for 2D"
        @assert length(amr_level_2d.x_faces) == 17 "X faces should have 17 elements"
        @assert length(amr_level_2d.z_faces) == 17 "Z faces should have 17 elements"
        @assert length(amr_level_2d.y_faces) == 0 "Y faces should be empty for 2D"
        
        # Verify solution array dimensions
        @assert size(amr_level_2d.u) == (17, 16) "U velocity should be (17, 16)"
        @assert size(amr_level_2d.v) == (16, 17) "V velocity should be (16, 17)"
        @assert amr_level_2d.w === nothing "W velocity should be nothing for 2D"
        @assert size(amr_level_2d.p) == (16, 16) "Pressure should be (16, 16)"
        
        println("✓ 2D XZ coordinate system test passed")
        
    catch e
        println("✗ 2D XZ coordinate system test failed: $e")
        return false
    end
    
    # Test 3D case
    println("Testing 3D coordinate system...")
    try
        # Create 3D AMR level 
        amr_level_3d = AMRLevel(ThreeDimensional, 0, 16, 12, 8, 0.1, 0.1, 0.1, 
                               0.0, 0.0, 0.0)
        
        # Verify coordinates
        @assert length(amr_level_3d.x_centers) == 16 "X centers should have 16 elements"
        @assert length(amr_level_3d.y_centers) == 12 "Y centers should have 12 elements"
        @assert length(amr_level_3d.z_centers) == 8 "Z centers should have 8 elements"
        
        # Verify solution array dimensions
        @assert size(amr_level_3d.u) == (17, 12, 8) "U velocity should be (17, 12, 8)"
        @assert size(amr_level_3d.v) == (16, 13, 8) "V velocity should be (16, 13, 8)"
        @assert size(amr_level_3d.w) == (16, 12, 9) "W velocity should be (16, 12, 9)"
        @assert size(amr_level_3d.p) == (16, 12, 8) "Pressure should be (16, 12, 8)"
        
        println("✓ 3D coordinate system test passed")
        
    catch e
        println("✗ 3D coordinate system test failed: $e")
        return false
    end
    
    return true
end

function test_refinement_indicators()
    println("\n=== Testing Refinement Indicators ===")
    
    # Create test hierarchy
    try
        base_grid = StaggeredGrid(16, 16, 0.1, 0.1, TwoDimensional)
        hierarchy = AMRHierarchy(base_grid; max_level=2)
        
        # Create test solution state
        state = SolutionState2D(16, 16)
        
        # Fill with some test data
        for j = 1:16, i = 1:17
            state.u[i, j] = sin(2π * i / 17) * cos(2π * j / 16)
        end
        for j = 1:17, i = 1:16  
            state.v[i, j] = cos(2π * i / 16) * sin(2π * j / 17)
        end
        for j = 1:16, i = 1:16
            state.p[i, j] = sin(π * i / 16) * sin(π * j / 16)
        end
        
        # Test indicator computation
        indicators = compute_refinement_indicators_amr(hierarchy.base_level, state, nothing, hierarchy)
        
        # Verify indicators array dimensions
        @assert size(indicators) == (16, 16) "2D indicators should be (16, 16)"
        @assert all(indicators .>= 0.0) "All indicators should be non-negative"
        @assert all(indicators .<= 1.0) "All indicators should be ≤ 1.0"
        
        println("✓ 2D refinement indicators test passed")
        
        # Test 3D case (basic check)
        base_grid_3d = StaggeredGrid(16, 12, 8, 0.1, 0.1, 0.1, ThreeDimensional)
        hierarchy_3d = AMRHierarchy(base_grid_3d; max_level=2)
        state_3d = SolutionState3D(16, 12, 8)
        
        indicators_3d = compute_refinement_indicators_amr(hierarchy_3d.base_level, state_3d, nothing, hierarchy_3d)
        @assert size(indicators_3d) == (16, 12, 8) "3D indicators should be (16, 12, 8)"
        
        println("✓ 3D refinement indicators basic test passed")
        
    catch e
        println("✗ Refinement indicators test failed: $e")
        return false
    end
    
    return true
end

function test_velocity_interpolation()
    println("\n=== Testing Velocity Interpolation ===")
    
    try
        # Create test AMR level
        amr_level = AMRLevel(TwoDimensional, 1, 8, 8, 0.1, 0.1, 0.0, 0.0)
        
        # Create coarse and fine velocity arrays
        u_coarse = ones(5, 4)  # 4+1 x-faces, 4 z-cells
        v_coarse = ones(4, 5)  # 4 x-cells, 4+1 z-faces
        
        u_fine = zeros(9, 8)   # 8+1 x-faces, 8 z-cells
        v_fine = zeros(8, 9)   # 8 x-cells, 8+1 z-faces
        
        # Test conservative interpolation
        interpolate_velocity_conservative!(u_fine, v_fine, u_coarse, v_coarse, amr_level)
        
        # Check that interpolation preserves basic properties
        @assert all(u_fine .>= 0.0) "U velocity should be non-negative"
        @assert all(v_fine .>= 0.0) "V velocity should be non-negative"
        @assert maximum(abs.(u_fine)) <= 2.0 "U velocity should be reasonable"
        @assert maximum(abs.(v_fine)) <= 2.0 "V velocity should be reasonable"
        
        println("✓ Velocity interpolation test passed")
        
    catch e
        println("✗ Velocity interpolation test failed: $e")
        return false
    end
    
    return true
end

function test_conservative_restriction()
    println("\n=== Testing Conservative Restriction ===")
    
    try
        # Create test fine array
        fine_array = ones(16, 16)
        
        # Add some variation
        for j = 1:16, i = 1:16
            fine_array[i, j] = 1.0 + 0.1 * sin(2π * i / 16) * cos(2π * j / 16)
        end
        
        # Test conservative restriction
        coarse_array = conservative_restriction_2d(fine_array, 2)
        
        # Verify dimensions
        @assert size(coarse_array) == (8, 8) "Coarse array should be (8, 8)"
        
        # Check conservation (sum should be approximately preserved)
        fine_sum = sum(fine_array)
        coarse_sum = sum(coarse_array) * 4  # Scale by area ratio
        relative_error = abs(fine_sum - coarse_sum) / fine_sum
        
        @assert relative_error < 0.01 "Conservation error should be < 1%"
        
        println("✓ Conservative restriction test passed (error: $(100*relative_error)%)")
        
    catch e
        println("✗ Conservative restriction test failed: $e")
        return false
    end
    
    return true
end

# Run all tests
function run_all_amr_tests()
    println("🧪 Running AMR Implementation Validation Tests")
    println("=" ^ 50)
    
    tests_passed = 0
    total_tests = 4
    
    if test_amr_coordinate_consistency()
        tests_passed += 1
    end
    
    if test_refinement_indicators()  
        tests_passed += 1
    end
    
    if test_velocity_interpolation()
        tests_passed += 1
    end
    
    if test_conservative_restriction()
        tests_passed += 1
    end
    
    println("\n" * "=" * 50)
    println("📊 AMR Test Results: $tests_passed/$total_tests tests passed")
    
    if tests_passed == total_tests
        println("✅ All AMR tests passed! The implementation is working correctly.")
        return true
    else
        println("❌ Some AMR tests failed. Please review the implementation.")
        return false
    end
end

# Run the tests
try
    success = run_all_amr_tests()
    exit(success ? 0 : 1)
catch e
    println("❌ Test execution failed with error: $e")
    exit(1)
end