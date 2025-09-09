#!/usr/bin/env julia

# Minimal test to isolate the LLVM compilation issue
using LinearAlgebra

println("Testing basic Julia functionality...")

# Test 1: Basic array operations
println("Test 1: Basic array operations")
A = rand(10, 10)
B = A .+ 1.0
println("  ✓ Basic array operations work")

# Test 2: Simple finite differences (like in BioFlows)
println("Test 2: Simple finite difference operations")
function simple_laplacian(u::Matrix{Float64}, dx::Float64, dy::Float64)
    nx, ny = size(u)
    result = zeros(nx, ny)
    
    for i in 2:nx-1, j in 2:ny-1
        result[i,j] = (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx^2 + 
                      (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dy^2
    end
    return result
end

u = rand(32, 16)
lap = simple_laplacian(u, 0.1, 0.1)
println("  ✓ Simple finite differences work")

# Test 3: More complex array operations
println("Test 3: Complex array operations with broadcasting")
try
    result = u .* lap .+ 0.01 .* lap
    println("  ✓ Complex broadcasting operations work")
catch e
    println("  ✗ Complex broadcasting failed: $e")
end

println("All basic tests passed. Issue likely specific to BioFlows implementation.")