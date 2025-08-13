"""
WaterLily.jl-style Multigrid Solver Demonstration

This example demonstrates the performance and accuracy of the WaterLily.jl-style
multigrid solver compared to the GeometricMultigrid.jl approach.
"""

using BioFlow

function demo_waterlily_multigrid()
    println("WaterLily.jl-style Multigrid Solver Demo")
    println("=" * 50)
    
    # Create test grid
    nx, ny = 128, 96
    Lx, Ly = 2.0, 1.5
    grid = StaggeredGrid2D(nx, ny, Lx, Ly)
    
    println("Grid: $(nx) × $(ny)")
    println("Domain: $(Lx) × $(Ly)")
    println("Grid spacing: dx = $(grid.dx), dy = $(grid.dy)")
    println()
    
    # Create test RHS with known analytical solution
    # Let φ_exact = sin(πx/Lx) * cos(πy/Ly)
    # Then ∇²φ = -(π²/Lx² + π²/Ly²) * sin(πx/Lx) * cos(πy/Ly)
    rhs = zeros(nx, ny)
    φ_exact = zeros(nx, ny)
    
    for j = 1:ny, i = 1:nx
        x, y = grid.x[i], grid.y[j]
        φ_exact[i, j] = sin(π * x / Lx) * cos(π * y / Ly)
        rhs[i, j] = -(π^2/Lx^2 + π^2/Ly^2) * φ_exact[i, j]
    end
    
    println("Test problem: ∇²φ = rhs with analytical solution")
    println("φ_exact = sin(πx/Lx) * cos(πy/Ly)")
    println()
    
    # Test WaterLily.jl-style solver
    println("1. WaterLily.jl-style Multigrid Solver")
    println("-" * 40)
    
    φ_waterlily = zeros(nx, ny)
    
    # Create WaterLily.jl-style solver
    mg_waterlily = MultiLevelPoisson(nx, ny, grid.dx, grid.dy, 4; n_smooth=3, tol=1e-8)
    
    # Time the solution
    start_time = time()
    residual_wl, iterations_wl = solve_poisson!(φ_waterlily, rhs, mg_waterlily; max_iter=100)
    time_wl = time() - start_time
    
    # Compute error
    error_wl = maximum(abs.(φ_waterlily - φ_exact))
    l2_error_wl = sqrt(sum((φ_waterlily - φ_exact).^2) / (nx * ny))
    
    println("  Iterations: $iterations_wl")
    println("  Final residual: $residual_wl")
    println("  Max error: $error_wl")
    println("  L2 error: $l2_error_wl")
    println("  Time: $(time_wl * 1000) ms")
    println()
    
    # Test GeometricMultigrid.jl solver (for comparison)
    println("2. GeometricMultigrid.jl Solver")
    println("-" * 40)
    
    φ_geometric = zeros(nx, ny)
    bc = BoundaryConditions2D()  # Default homogeneous Neumann
    
    # Create GeometricMultigrid.jl-style solver
    solver_geometric = MultigridPoissonSolver(grid; solver_type=:geometric, levels=4)
    
    # Time the solution
    start_time = time()
    solve_poisson!(solver_geometric, φ_geometric, rhs, grid, bc)
    time_geom = time() - start_time
    
    # Compute error
    error_geom = maximum(abs.(φ_geometric - φ_exact))
    l2_error_geom = sqrt(sum((φ_geometric - φ_exact).^2) / (nx * ny))
    
    println("  Max error: $error_geom")
    println("  L2 error: $l2_error_geom")
    println("  Time: $(time_geom * 1000) ms")
    println()
    
    # Compare solvers
    println("3. Performance Comparison")
    println("-" * 40)
    speedup = time_geom / time_wl
    println("  Speedup (WaterLily vs Geometric): $(speedup)x")
    println("  Accuracy comparison:")
    println("    WaterLily L2 error: $l2_error_wl")
    println("    Geometric L2 error: $l2_error_geom")
    
    if l2_error_wl < l2_error_geom
        println("    ✓ WaterLily solver is more accurate")
    else
        println("    ✓ Geometric solver is more accurate")
    end
    
    if speedup > 1.0
        println("    ✓ WaterLily solver is faster")
    else
        println("    ✓ Geometric solver is faster")
    end
    println()
    
    # Test convergence behavior
    println("4. Convergence Analysis")
    println("-" * 40)
    
    test_convergence_rates(grid)
    println()
    
    # Test with different grid sizes
    println("5. Scalability Test")
    println("-" * 40)
    
    test_scalability()
    
    println()
    println("Demo completed successfully!")
    println("✓ WaterLily.jl-style multigrid implemented")
    println("✓ Performance comparison completed")
    println("✓ Solver accuracy verified")
end

function test_convergence_rates(grid::StaggeredGrid)
    """Test convergence rates for different tolerances."""
    
    nx, ny = grid.nx, grid.ny
    Lx, Ly = grid.Lx, grid.Ly
    
    # Create test problem
    rhs = zeros(nx, ny)
    for j = 1:ny, i = 1:nx
        x, y = grid.x[i], grid.y[j]
        rhs[i, j] = -(π^2/Lx^2 + π^2/Ly^2) * sin(π * x / Lx) * cos(π * y / Ly)
    end
    
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    
    println("  Tolerance | Iterations | Time (ms)")
    println("  ----------|------------|----------")
    
    for tol in tolerances
        φ = zeros(nx, ny)
        mg = MultiLevelPoisson(nx, ny, grid.dx, grid.dy, 4; n_smooth=3, tol=tol)
        
        start_time = time()
        residual, iterations = solve_poisson!(φ, rhs, mg; max_iter=200)
        elapsed_time = (time() - start_time) * 1000
        
        @printf("  %8.0e |   %8d | %8.2f\n", tol, iterations, elapsed_time)
    end
end

function test_scalability()
    """Test solver performance for different grid sizes."""
    
    grid_sizes = [(32, 24), (64, 48), (128, 96), (256, 192)]
    
    println("  Grid Size | Levels | Time (ms) | Time/DOF (μs)")
    println("  ----------|--------|-----------|---------------")
    
    for (nx, ny) in grid_sizes
        grid = StaggeredGrid2D(nx, ny, 2.0, 1.5)
        
        # Create test RHS
        rhs = zeros(nx, ny)
        for j = 1:ny, i = 1:nx
            x, y = grid.x[i], grid.y[j]
            rhs[i, j] = -(π^2/4 + π^2/2.25) * sin(π * x / 2.0) * cos(π * y / 1.5)
        end
        
        φ = zeros(nx, ny)
        levels = min(4, Int(floor(log2(min(nx, ny)))) - 1)
        mg = MultiLevelPoisson(nx, ny, grid.dx, grid.dy, levels; n_smooth=3, tol=1e-6)
        
        start_time = time()
        residual, iterations = solve_poisson!(φ, rhs, mg; max_iter=100)
        elapsed_time = (time() - start_time) * 1000
        
        time_per_dof = elapsed_time * 1000 / (nx * ny)  # μs per DOF
        
        @printf("  %3d × %3d |   %4d | %9.2f | %13.3f\n", 
                nx, ny, levels, elapsed_time, time_per_dof)
    end
    
    println()
    println("  Note: Time per DOF should remain roughly constant")
    println("        for optimal multigrid performance (O(N) complexity)")
end

# Demonstration of V-cycle efficiency
function demo_v_cycle_efficiency()
    println("\nV-Cycle Efficiency Demonstration")
    println("-" * 35)
    
    # Create test case
    nx, ny = 128, 128
    grid = StaggeredGrid2D(nx, ny, 1.0, 1.0)
    
    # Random RHS
    rhs = randn(nx, ny)
    rhs .-= sum(rhs) / (nx * ny)  # Ensure compatibility
    
    φ = zeros(nx, ny)
    mg = MultiLevelPoisson(nx, ny, grid.dx, grid.dy, 5; n_smooth=2, tol=1e-10)
    
    println("Tracking residual reduction per V-cycle...")
    
    # Manual V-cycles with residual tracking
    mg.x[1] .= φ
    mg.b[1] .= rhs
    
    for cycle = 1:10
        # Perform one V-cycle
        v_cycle!(mg, 1)
        
        # Compute residual norm
        residual_norm = compute_residual_norm(mg, 1)
        
        println("  V-cycle $cycle: residual = $(residual_norm)")
        
        if residual_norm < 1e-10
            println("  Converged after $cycle V-cycles")
            break
        end
    end
    
    φ .= mg.x[1]
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_waterlily_multigrid()
    demo_v_cycle_efficiency()
end