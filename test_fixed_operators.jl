#!/usr/bin/env julia

# Test script for fixed differential operators
println("Testing fixed differential operators...")

# Include the module in development mode
push!(LOAD_PATH, pwd())

try
    using BioFlows
    println("✓ BioFlows loaded successfully")
catch e
    println("✗ Failed to load BioFlows: $e")
    exit(1)
end

# Test our simplified differential operators
try
    println("\nTesting differential operators...")
    
    # Create a simple 2D grid
    grid = BioFlows.StaggeredGrid(
        nx=16, ny=8, nz=1,
        Lx=2.0, Ly=1.0, Lz=1.0,
        origin=[0.0, 0.0, 0.0]
    )
    
    # Create test field
    field = rand(Float64, 16, 8)
    
    # Test our optimized laplacian
    println("  Testing laplacian...")
    lap_result = BioFlows.laplacian(field, grid)
    println("  ✓ Laplacian computation completed without LLVM errors")
    
    # Test second derivatives
    println("  Testing second derivatives...")
    d2dx2_result = BioFlows.d2dx2(field, grid)
    d2dy2_result = BioFlows.d2dy2(field, grid)
    println("  ✓ Second derivative computations completed")
    
    println("\n✅ All differential operator tests passed!")
    println("   Fixed LLVM compilation issues successfully.")
    
catch e
    println("✗ Differential operator test failed: $e")
    println("Stack trace:")
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        println("  $i. $frame")
        if i > 10; break; end
    end
    exit(1)
end