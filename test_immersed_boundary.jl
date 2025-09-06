# Test immersed boundary method implementation
using LinearAlgebra

# Load BioFlows
include(joinpath(@__DIR__, "src", "BioFlows.jl"))
using .BioFlows

println("Testing Immersed Boundary Method Implementation")
println("=" ^ 50)

function test_ibm_data_structure_consistency()
    println("\n=== Testing IBM Data Structure Consistency ===")
    
    try
        # Create test grid and bodies
        grid = StaggeredGrid2D(32, 24, 2.0, 1.5)
        
        # Create a simple circular body
        circle_body = RigidBody(
            Circle(0.2), 
            [1.0, 0.75], 
            [0.1, 0.0], 
            0.0, 
            0.0, 
            1.0, 
            1.0, 
            false, 
            1
        )
        bodies = RigidBodyCollection()
        add_body!(bodies, circle_body)
        
        # Create IBM data
        ib_data = ImmersedBoundaryData2D(bodies, grid)
        
        # Verify data structure consistency
        @assert size(ib_data.body_mask) == (grid.nx, grid.nz) "Body mask dimensions incorrect"
        @assert size(ib_data.distance_function) == (grid.nx, grid.nz) "Distance function dimensions incorrect"
        @assert size(ib_data.normal_vectors) == (grid.nx, grid.nz) "Normal vectors dimensions incorrect"
        @assert length(ib_data.boundary_points) > 0 "No boundary points generated"
        @assert length(ib_data.forcing_points) > 0 "No forcing points generated"
        
        println("✓ IBM data structure consistency verified")
        return true
        
    catch e
        println("✗ IBM data structure test failed: $e")
        return false
    end
end

function test_velocity_naming_consistency()
    println("\n=== Testing Velocity Naming Consistency ===")
    
    try
        # Create test grid and state
        grid = StaggeredGrid2D(16, 12, 1.0, 1.0)
        state = SolutionState2D(grid.nx, grid.nz)
        
        # Initialize test velocities
        for j = 1:grid.nz, i = 1:grid.nx+1
            state.u[i, j] = 1.0  # x-velocity
        end
        
        for j = 1:grid.nz+1, i = 1:grid.nx
            state.w[i, j] = 0.5  # z-velocity (NOT state.v)
        end
        
        # Verify we can access z-velocity correctly
        @assert size(state.w) == (grid.nx, grid.nz+1) "Z-velocity should use state.w in 2D XZ plane"
        @assert maximum(state.w) == 0.5 "Z-velocity values incorrect"
        
        println("✓ Velocity naming consistency verified (2D XZ uses state.w)")
        return true
        
    catch e
        println("✗ Velocity naming consistency test failed: $e")
        return false
    end
end

function test_body_mask_computation()
    println("\n=== Testing Body Mask Computation ===")
    
    try
        grid = StaggeredGrid2D(20, 20, 2.0, 2.0)
        
        # Create circle at center
        circle_body = RigidBody(
            Circle(0.3), 
            [1.0, 1.0], 
            [0.0, 0.0], 
            0.0, 
            0.0, 
            1.0, 
            1.0, 
            false, 
            1
        )
        bodies = RigidBodyCollection()
        add_body!(bodies, circle_body)
        
        # Compute body mask
        body_mask = bodies_mask_2d(bodies, grid)
        
        # Check that some cells are marked as inside the body
        inside_count = sum(body_mask)
        @assert inside_count > 0 "No cells marked as inside body"
        @assert inside_count < grid.nx * grid.nz "Not all cells should be inside body"
        
        # Check center point is inside
        center_i = Int(round(1.0 / grid.dx)) + 1
        center_j = Int(round(1.0 / grid.dz)) + 1
        @assert body_mask[center_i, center_j] "Center should be inside body"
        
        println("✓ Body mask computation verified ($inside_count cells inside)")
        return true
        
    catch e
        println("✗ Body mask computation test failed: $e")
        return false
    end
end

function test_regularized_delta_function()
    println("\n=== Testing Regularized Delta Function ===")
    
    try
        δh = 0.1
        grid_dx = 0.05
        grid_dz = 0.05
        
        # Test at origin (should be maximum)
        δ_origin = regularized_delta_2d(0.0, 0.0, δh, grid_dx, grid_dz)
        @assert δ_origin > 0 "Delta function should be positive at origin"
        
        # Test at distance > 2*δh (should be zero)
        δ_far = regularized_delta_2d(3*δh, 0.0, δh, grid_dx, grid_dz)
        @assert δ_far ≈ 0.0 atol=1e-10 "Delta function should be zero far from origin"
        
        # Test symmetry
        δ_pos = regularized_delta_2d(δh/2, δh/3, δh, grid_dx, grid_dz)
        δ_neg = regularized_delta_2d(-δh/2, -δh/3, δh, grid_dx, grid_dz)
        @assert δ_pos ≈ δ_neg atol=1e-10 "Delta function should be symmetric"
        
        println("✓ Regularized delta function properties verified")
        return true
        
    catch e
        println("✗ Regularized delta function test failed: $e")
        return false
    end
end

function test_force_spreading_conservation()
    println("\n=== Testing Force Spreading Conservation ===")
    
    try
        grid = StaggeredGrid2D(16, 16, 1.0, 1.0)
        nx, nz = grid.nx, grid.nz
        
        # Create force field
        force_field = Array{Vector{Float64},2}(undef, nx, nz)
        for j = 1:nz, i = 1:nx
            force_field[i, j] = [0.0, 0.0]
        end
        
        # Single Lagrangian point with unit force
        lag_positions = [[0.5, 0.5]]
        lag_forces = [[1.0, 2.0]]
        
        δh = 2.0 * max(grid.dx, grid.dz)
        
        # Apply force spreading
        apply_force_spreading_2d!(force_field, lag_forces, lag_positions, grid, δh)
        
        # Check conservation: total force should equal input force
        total_force = [0.0, 0.0]
        for j = 1:nz, i = 1:nx
            total_force[1] += force_field[i, j][1]
            total_force[2] += force_field[i, j][2]
        end
        
        @assert abs(total_force[1] - 1.0) < 0.1 "X-force not conserved in spreading"
        @assert abs(total_force[2] - 2.0) < 0.1 "Z-force not conserved in spreading"
        
        println("✓ Force spreading conservation verified")
        return true
        
    catch e
        println("✗ Force spreading conservation test failed: $e")
        return false
    end
end

# Run all tests
function run_all_ibm_tests()
    tests_passed = 0
    total_tests = 5
    
    if test_ibm_data_structure_consistency()
        tests_passed += 1
    end
    
    if test_velocity_naming_consistency()
        tests_passed += 1
    end
    
    if test_body_mask_computation()
        tests_passed += 1
    end
    
    if test_regularized_delta_function()
        tests_passed += 1
    end
    
    if test_force_spreading_conservation()
        tests_passed += 1
    end
    
    println("\n" * repeat("=", 50))
    println("IBM Test Results: $tests_passed/$total_tests tests passed")
    
    if tests_passed == total_tests
        println("All IBM tests passed! The immersed boundary method is working correctly.")
        return true
    else
        println("Some IBM tests failed. Please review the implementation.")
        return false
    end
end

# Run the tests
try
    success = run_all_ibm_tests()
    exit(success ? 0 : 1)
catch e
    println("Test execution failed with error: $e")
    exit(1)
end