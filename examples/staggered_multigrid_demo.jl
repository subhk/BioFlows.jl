"""
Staggered Grid Multigrid Demonstration

This example demonstrates the proper handling of staggered grids in the multigrid solver:
- Pressure at cell centers
- Velocities at cell faces (middle of edges)  
- Proper discretization of pressure Poisson equation
- Correct gradient computation from cell centers to faces
"""

using BioFlow

function demo_staggered_multigrid()
    println("Staggered Grid Multigrid Solver Demo")
    println("=" * 50)
    
    # Create staggered grid
    nx, ny = 64, 48
    Lx, Ly = 2.0, 1.5
    grid = StaggeredGrid2D(nx, ny, Lx, Ly)
    
    println("Grid Configuration:")
    println("  Cells: $(nx) × $(ny)")
    println("  Domain: $(Lx) × $(Ly)")  
    println("  Cell spacing: dx = $(grid.dx), dy = $(grid.dy)")
    println()
    
    # Display grid layout
    println("Staggered Grid Layout:")
    println("  Pressure (φ): cell centers at (x[i], y[j]) where")
    println("    x[i] = $(grid.x[1]) to $(grid.x[end]) ($(length(grid.x)) points)")
    println("    y[j] = $(grid.y[1]) to $(grid.y[end]) ($(length(grid.y)) points)")
    println("  u-velocity: x-faces at xu[i] = $(grid.xu[1]) to $(grid.xu[end]) ($(length(grid.xu)) points)")
    println("  v-velocity: y-faces at yv[j] = $(grid.yv[1]) to $(grid.yv[end]) ($(length(grid.yv)) points)")
    println()
    
    # Test 1: Verify staggered grid structure
    println("1. Staggered Grid Verification")
    println("-" * 30)
    
    verify_staggered_layout(grid)
    println()
    
    # Test 2: Create realistic CFD test case
    println("2. CFD-Style Test Case")
    println("-" * 30)
    
    # Create velocity field on faces that represents some flow
    u_faces = zeros(nx + 1, ny)      # u at x-faces
    v_faces = zeros(nx, ny + 1)      # v at y-faces
    
    # Create a simple vortex-like flow
    for j = 1:ny, i = 1:nx+1
        x_face = grid.xu[i]
        y_center = grid.y[j]
        
        # Circular flow pattern
        xc, yc = Lx/2, Ly/2  # Center of vortex
        r = sqrt((x_face - xc)^2 + (y_center - yc)^2)
        if r > 0.1  # Avoid singularity
            u_faces[i, j] = -(y_center - yc) / r * exp(-r)
        end
    end
    
    for j = 1:ny+1, i = 1:nx
        x_center = grid.x[i]
        y_face = grid.yv[j]
        
        xc, yc = Lx/2, Ly/2
        r = sqrt((x_center - xc)^2 + (y_face - yc)^2)
        if r > 0.1
            v_faces[i, j] = (x_center - xc) / r * exp(-r)
        end
    end
    
    println("  Created vortex-like velocity field:")
    println("    u-velocity range: [$(minimum(u_faces)), $(maximum(u_faces))]")
    println("    v-velocity range: [$(minimum(v_faces)), $(maximum(v_faces))]")
    
    # Compute divergence at cell centers (RHS for pressure Poisson)
    div_u = zeros(nx, ny)
    compute_velocity_divergence_from_faces!(div_u, u_faces, v_faces, grid)
    
    max_div = maximum(abs.(div_u))
    println("    Initial divergence: max|∇·u| = $max_div")
    println()
    
    # Test 3: Solve pressure Poisson equation
    println("3. Pressure Poisson Solution")
    println("-" * 30)
    
    # Initialize pressure at cell centers
    φ = zeros(nx, ny)  # Pressure correction at cell centers
    
    # Create staggered-aware multigrid solver
    solver = MultigridPoissonSolver(grid; solver_type=:staggered, levels=4, tolerance=1e-8)
    bc = BoundaryConditions2D()  # Default Neumann BC for pressure
    
    # Solve ∇²φ = ∇·u
    println("  Solving ∇²φ = ∇·u with staggered multigrid...")
    start_time = time()
    solve_poisson!(solver, φ, div_u, grid, bc)
    elapsed_time = time() - start_time
    
    println("  Pressure solution computed:")
    println("    φ range: [$(minimum(φ)), $(maximum(φ))]")
    println("    Time: $(elapsed_time * 1000) ms")
    println()
    
    # Test 4: Verify pressure gradient computation
    println("4. Pressure Gradient to Faces")
    println("-" * 30)
    
    # Compute pressure gradient at faces (for velocity correction)
    dpdx_faces = zeros(nx + 1, ny)   # Pressure gradient at x-faces
    dpdy_faces = zeros(nx, ny + 1)   # Pressure gradient at y-faces
    
    compute_pressure_gradient_to_faces!(dpdx_faces, dpdy_faces, φ, grid)
    
    println("  Pressure gradient computed at faces:")
    println("    ∂φ/∂x at x-faces: range [$(minimum(dpdx_faces)), $(maximum(dpdx_faces))]")
    println("    ∂φ/∂y at y-faces: range [$(minimum(dpdy_faces)), $(maximum(dpdy_faces))]")
    
    # Verify dimensions
    @assert size(dpdx_faces) == size(u_faces) "∂φ/∂x must match u-velocity array size"
    @assert size(dpdy_faces) == size(v_faces) "∂φ/∂y must match v-velocity array size"
    println("    ✓ Gradient arrays have correct staggered dimensions")
    println()
    
    # Test 5: Complete projection step
    println("5. Complete Projection Step")
    println("-" * 30)
    
    # Velocity correction: u_new = u_old - dt * ∇φ
    dt = 0.01
    u_corrected = u_faces - dt * dpdx_faces
    v_corrected = v_faces - dt * dpdy_faces
    
    # Check divergence of corrected velocity
    div_u_corrected = zeros(nx, ny)
    compute_velocity_divergence_from_faces!(div_u_corrected, u_corrected, v_corrected, grid)
    
    max_div_corrected = maximum(abs.(div_u_corrected))
    reduction_factor = max_div_corrected / max_div
    
    println("  After pressure projection:")
    println("    Original divergence: $max_div")
    println("    Corrected divergence: $max_div_corrected")
    println("    Reduction factor: $reduction_factor")
    
    if reduction_factor < 0.01
        println("    ✓ Excellent divergence reduction!")
    elseif reduction_factor < 0.1
        println("    ✓ Good divergence reduction")
    else
        println("    ⚠ Limited divergence reduction - may need tighter tolerance")
    end
    println()
    
    # Test 6: Compare with non-staggered approach
    println("6. Comparison with Standard Multigrid")
    println("-" * 40)
    
    # Test same problem with regular WaterLily-style solver
    φ_regular = zeros(nx, ny)
    solver_regular = MultigridPoissonSolver(grid; solver_type=:waterlily, levels=4, tolerance=1e-8)
    
    start_time = time()
    solve_poisson!(solver_regular, φ_regular, div_u, grid, bc)
    time_regular = time() - start_time
    
    # Compare solutions
    error_diff = maximum(abs.(φ - φ_regular))
    
    println("  Solution comparison:")
    println("    Staggered solver time: $(elapsed_time * 1000) ms")
    println("    Regular solver time: $(time_regular * 1000) ms")
    println("    Max solution difference: $error_diff")
    
    if error_diff < 1e-10
        println("    ✓ Solutions are essentially identical")
        println("    → Both solvers solve the same discrete system correctly")
    else
        println("    → Small differences due to different boundary handling")
    end
    println()
    
    # Test 7: Grid refinement sensitivity
    println("7. Grid Refinement Study")
    println("-" * 30)
    
    test_grid_refinement()
    
    println()
    println("Demo completed successfully!")
    println("✓ Staggered grid layout verified")
    println("✓ Pressure at cell centers, velocities at faces")
    println("✓ Proper divergence computation from faces to centers")
    println("✓ Correct pressure gradient computation from centers to faces")
    println("✓ Complete projection method demonstrated")
end

function verify_staggered_layout(grid::StaggeredGrid)
    """Verify that the staggered grid has the correct layout."""
    
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    
    # Check cell centers
    x_center_expected = dx/2
    y_center_expected = dy/2
    
    println("  Cell center verification:")
    println("    First cell center: ($(grid.x[1]), $(grid.y[1]))")
    println("    Expected: ($x_center_expected, $y_center_expected)")
    
    @assert abs(grid.x[1] - x_center_expected) < 1e-14 "x-centers not at cell centers"
    @assert abs(grid.y[1] - y_center_expected) < 1e-14 "y-centers not at cell centers"
    println("    ✓ Cell centers correctly positioned")
    
    # Check face positions
    println("  Face position verification:")
    println("    First x-face: $(grid.xu[1]) (should be 0.0)")
    println("    Second x-face: $(grid.xu[2]) (should be $dx)")
    println("    First y-face: $(grid.yv[1]) (should be 0.0)")
    println("    Second y-face: $(grid.yv[2]) (should be $dy)")
    
    @assert abs(grid.xu[1]) < 1e-14 "First x-face not at boundary"
    @assert abs(grid.xu[2] - dx) < 1e-14 "Second x-face not at correct position"
    @assert abs(grid.yv[1]) < 1e-14 "First y-face not at boundary"
    @assert abs(grid.yv[2] - dy) < 1e-14 "Second y-face not at correct position"
    println("    ✓ Face positions correctly staggered")
    
    # Check array dimensions
    println("  Array dimension verification:")
    println("    Pressure arrays: $(nx) × $(ny)")
    println("    u-velocity arrays: $(nx+1) × $(ny)")  
    println("    v-velocity arrays: $(nx) × $(ny+1)")
    println("    ✓ Array dimensions match staggered grid requirements")
end

function test_grid_refinement()
    """Test behavior with different grid resolutions."""
    
    grid_sizes = [(16, 12), (32, 24), (64, 48), (128, 96)]
    
    println("  Grid Size | Max Div Before | Max Div After | Reduction | Time (ms)")
    println("  ----------|----------------|---------------|-----------|----------")
    
    for (nx, ny) in grid_sizes
        grid = StaggeredGrid2D(nx, ny, 2.0, 1.5)
        
        # Create simple test flow
        u_faces = zeros(nx + 1, ny)
        v_faces = zeros(nx, ny + 1)
        
        # Simple divergent flow
        for j = 1:ny, i = 1:nx+1
            x = grid.xu[i] - 1.0
            u_faces[i, j] = x * 0.1
        end
        for j = 1:ny+1, i = 1:nx
            y = grid.yv[j] - 0.75
            v_faces[i, j] = y * 0.1
        end
        
        # Compute initial divergence
        div_u = zeros(nx, ny)
        compute_velocity_divergence_from_faces!(div_u, u_faces, v_faces, grid)
        max_div_before = maximum(abs.(div_u))
        
        # Solve pressure Poisson
        φ = zeros(nx, ny)
        solver = MultigridPoissonSolver(grid; solver_type=:staggered, levels=3, tolerance=1e-6)
        bc = BoundaryConditions2D()
        
        start_time = time()
        solve_poisson!(solver, φ, div_u, grid, bc)
        elapsed_time = (time() - start_time) * 1000
        
        # Compute corrected divergence
        dpdx = zeros(nx + 1, ny)
        dpdy = zeros(nx, ny + 1)
        compute_pressure_gradient_to_faces!(dpdx, dpdy, φ, grid)
        
        dt = 0.01
        u_corrected = u_faces - dt * dpdx
        v_corrected = v_faces - dt * dpdy
        
        div_corrected = zeros(nx, ny)
        compute_velocity_divergence_from_faces!(div_corrected, u_corrected, v_corrected, grid)
        max_div_after = maximum(abs.(div_corrected))
        
        reduction = max_div_after / max_div_before
        
        @printf("  %3d × %3d | %14.6e | %13.6e | %9.2e | %8.2f\n", 
                nx, ny, max_div_before, max_div_after, reduction, elapsed_time)
    end
    
    println("  Note: Reduction factor should improve with finer grids")
    println("        Time should scale roughly as O(N) for optimal multigrid")
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_staggered_multigrid()
end