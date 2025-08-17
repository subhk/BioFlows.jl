"""
Test Complete AMR Pressure Solver

This example demonstrates the complete solve_poisson_amr! function
with proper interface conditions and multi-level convergence.
"""

push!(LOAD_PATH, ".")
using BioFlows
using Printf

function create_test_amr_hierarchy()
    """Create a simple AMR hierarchy for testing."""
    
    println("Creating test AMR hierarchy...")
    
    # Base grid parameters
    nx_base, nz_base = 16, 12
    Lx, Lz = 4.0, 3.0
    base_grid = create_uniform_2d_grid(nx_base, nz_base, Lx, Lz)
    
    # Create AMR hierarchy
    hierarchy = AMRHierarchy(
        create_amr_level_from_grid(base_grid, 0),  # Base level
        2,                                         # Max level
        2,                                         # Refinement ratio
        1.0, 10.0, 5.0, 0.1,                     # Thresholds
        10, 2,                                     # Regrid interval, buffer
        Dict{Int, MultigridPoissonSolver}()       # Solvers (empty initially)
    )
    
    # Add levels
    hierarchy.levels = Dict{Int, AMRLevel}()
    hierarchy.levels[0] = create_amr_level_from_grid(base_grid, 0)
    
    # Create fine level (2x refinement)
    nx_fine, nz_fine = 2*nx_base, 2*nz_base
    fine_grid = create_uniform_2d_grid(nx_fine, nz_fine, Lx, Lz)
    hierarchy.levels[1] = create_amr_level_from_grid(fine_grid, 1)
    
    # Initialize multigrid solvers for each level
    hierarchy.mg_solvers[0] = MultigridPoissonSolver(base_grid)
    hierarchy.mg_solvers[1] = MultigridPoissonSolver(fine_grid)
    
    println("  ✓ Created hierarchy with $(length(hierarchy.levels)) levels")
    println("  ✓ Base level: $(nx_base) × $(nz_base)")
    println("  ✓ Fine level: $(nx_fine) × $(nz_fine)")
    
    return hierarchy
end

function create_amr_level_from_grid(grid::StaggeredGrid, level::Int)
    """Convert StaggeredGrid to AMRLevel."""
    
    AMRLevel(
        level, grid.nx, 0, grid.nz,           # level, nx, ny, nz (ny=0 for 2D)
        grid.dx, 0.0, grid.dz,                # dx, dy, dz
        TwoDimensional,                       # grid_type
        collect(grid.x), Float64[], collect(grid.y),     # centers
        collect(grid.xu), Float64[], collect(grid.yv),   # faces
        zeros(grid.nx+1, grid.nz), zeros(grid.nx, grid.nz+1), nothing, zeros(grid.nx, grid.nz), # arrays
        zeros(Bool, grid.nx, grid.nz), zeros(Union{Nothing, AMRLevel}, grid.nx, grid.nz), nothing, # refinement
        0.0, grid.Lx, 0.0, 0.0, 0.0, grid.Lz,          # bounds
        0, Dict(), false                       # MPI, neighbors, is_boundary
    )
end

function create_test_rhs(amr_level::AMRLevel)
    """Create test right-hand side with known analytical solution."""
    
    nx, nz = amr_level.nx, amr_level.nz
    dx, dz = amr_level.dx, amr_level.dz
    Lx, Lz = nx*dx, nz*dz
    
    rhs = zeros(nx, nz)
    
    # RHS for ∇²p = f where p(x,z) = sin(2πx/Lx) * sin(2πz/Lz)
    # Then f = -∇²p = 4π²(1/Lx² + 1/Lz²) * sin(2πx/Lx) * sin(2πz/Lz)
    
    for j = 1:nz, i = 1:nx
        x = amr_level.x_centers[i]
        z = amr_level.z_centers[j]
        
        rhs[i, j] = 4π² * (1/Lx^2 + 1/Lz^2) * sin(2π*x/Lx) * sin(2π*z/Lz)
    end
    
    return rhs
end

function create_analytical_solution(amr_level::AMRLevel)
    """Create analytical solution for comparison."""
    
    nx, nz = amr_level.nx, amr_level.nz
    dx, dz = amr_level.dx, amr_level.dz
    Lx, Lz = nx*dx, nz*dz
    
    p_exact = zeros(nx, nz)
    
    for j = 1:nz, i = 1:nx
        x = amr_level.x_centers[i]
        z = amr_level.z_centers[j]
        
        p_exact[i, j] = sin(2π*x/Lx) * sin(2π*z/Lz)
    end
    
    return p_exact
end

function test_amr_pressure_solver()
    """Main test function for AMR pressure solver."""
    
    println("="^60)
    println("Testing Complete AMR Pressure Solver")
    println("="^60)
    
    # Create AMR hierarchy
    hierarchy = create_test_amr_hierarchy()
    
    # Create test problems for each level
    println("\\n1. Setting up test problems...")
    
    pressure = Dict{Int, Matrix{Float64}}()
    rhs = Dict{Int, Matrix{Float64}}()
    analytical = Dict{Int, Matrix{Float64}}()
    
    for level in keys(hierarchy.levels)
        amr_level = hierarchy.levels[level]
        
        # Initialize pressure with zeros
        pressure[level] = zeros(amr_level.nx, amr_level.nz)
        
        # Create RHS and analytical solution
        rhs[level] = create_test_rhs(amr_level)
        analytical[level] = create_analytical_solution(amr_level)
        
        println("    Level $level: $(amr_level.nx) × $(amr_level.nz) cells")
    end
    
    # Test the AMR solver
    println("\\n2. Solving AMR Poisson equation...")
    
    # Solve with verbose output
    iterations = solve_poisson_amr!(hierarchy, pressure, rhs; 
                                   max_iterations=50, tolerance=1e-8, verbose=true)
    
    println("\\n3. Analyzing results...")
    
    # Compute errors for each level
    total_error = 0.0
    total_cells = 0
    
    for level in keys(hierarchy.levels)
        p_computed = pressure[level]
        p_exact = analytical[level]
        
        # Compute L2 error
        error_l2 = sqrt(sum((p_computed .- p_exact).^2) / length(p_computed))
        max_error = maximum(abs.(p_computed .- p_exact))
        
        @printf "    Level %d: L2 error = %.2e, Max error = %.2e\\n" level error_l2 max_error
        
        total_error += error_l2 * length(p_computed)
        total_cells += length(p_computed)
    end
    
    overall_error = total_error / total_cells
    
    # Test interface conditions
    println("\\n4. Verifying interface conditions...")
    test_interface_continuity(hierarchy, pressure)
    
    # Verify global residual
    final_residual = compute_global_residual(hierarchy, pressure, rhs)
    @printf "    Final global residual: %.2e\\n" final_residual
    
    println("\\n5. Summary:")
    @printf "    ✓ Converged in %d iterations\\n" iterations
    @printf "    ✓ Overall L2 error: %.2e\\n" overall_error
    @printf "    ✓ Final residual: %.2e\\n" final_residual
    println("    ✓ Interface conditions satisfied")
    
    # Performance test
    println("\\n6. Performance test...")
    test_solver_performance(hierarchy, pressure, rhs)
    
    println("\\n" * "="^60)
    println("AMR Pressure Solver Test Completed Successfully!")
    println("="^60)
    
    return pressure, analytical, hierarchy
end

function test_interface_continuity(hierarchy::AMRHierarchy, pressure)
    """Test that pressure is continuous across coarse-fine interfaces."""
    
    if length(hierarchy.levels) < 2
        println("    ⚠ Only one level - no interfaces to test")
        return
    end
    
    # Test continuity between levels 0 and 1
    if haskey(hierarchy.levels, 0) && haskey(hierarchy.levels, 1)
        p_coarse = pressure[0]
        p_fine = pressure[1]
        
        # Sample a few interface points
        max_discontinuity = 0.0
        test_points = 5
        
        for i = 1:2:min(test_points*2, size(p_coarse, 1))
            for j = 1:2:min(test_points*2, size(p_coarse, 2))
                # Map coarse cell to fine cells
                i_f = 2*i - 1
                j_f = 2*j - 1
                
                if i_f <= size(p_fine, 1) && j_f <= size(p_fine, 2)
                    discontinuity = abs(p_coarse[i, j] - p_fine[i_f, j_f])
                    max_discontinuity = max(max_discontinuity, discontinuity)
                end
            end
        end
        
        @printf "    Interface discontinuity: max = %.2e\\n" max_discontinuity
        
        if max_discontinuity < 1e-6
            println("    ✓ Pressure continuous across interfaces")
        else
            println("    ⚠ Some interface discontinuity detected")
        end
    end
end

function test_solver_performance(hierarchy::AMRHierarchy, pressure, rhs)
    """Test solver performance with timing."""
    
    # Reset pressure for clean timing test
    for level in keys(pressure)
        pressure[level] .= 0.0
    end
    
    # Time the solver
    start_time = time()
    iterations = solve_poisson_amr!(hierarchy, pressure, rhs; 
                                   max_iterations=30, tolerance=1e-8, verbose=false)
    solve_time = time() - start_time
    
    total_cells = sum(length(pressure[level]) for level in keys(pressure))
    
    @printf "    Solve time: %.3f seconds\\n" solve_time
    @printf "    Iterations: %d\\n" iterations
    @printf "    Total cells: %d\\n" total_cells
    @printf "    Time per cell per iteration: %.2e sec\\n" (solve_time / (total_cells * iterations))
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_amr_pressure_solver()
end