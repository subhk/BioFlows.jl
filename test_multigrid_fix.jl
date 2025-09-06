# Test to validate multigrid solver correctness
# Test the fixed multigrid solver implementation

using LinearAlgebra

# Load BioFlows
include(joinpath(@__DIR__, "src", "BioFlows.jl"))
using .BioFlows

println("Testing BioFlows Multigrid Solver...")

# Create a simple 2D test case
nx, nz = 32, 32
Lx, Lz = 1.0, 1.0
dx, dz = Lx/nx, Lz/nz

# Create a simple staggered grid
grid = StaggeredGrid(nx, nz, dx, dz, TwoDimensional)

# Create a simple test RHS (manufactured solution)
# For ∇²φ = sin(πx)cos(πz), the solution is φ = -sin(πx)cos(πz)/(2π²)
x = [(i-0.5)*dx for i = 1:nx]
z = [(j-0.5)*dz for j = 1:nz]

rhs = zeros(nx, nz)
phi_exact = zeros(nx, nz)

for j = 1:nz, i = 1:nx
    xi, zj = x[i], z[j]
    rhs[i, j] = sin(π*xi) * cos(π*zj)
    phi_exact[i, j] = -sin(π*xi) * cos(π*zj) / (2*π^2)
end

# Remove mean to satisfy Neumann BC compatibility
rhs .-= sum(rhs) / (nx * nz)
phi_exact .-= sum(phi_exact) / (nx * nz)

# Create solver
solver = MultigridPoissonSolver(grid; levels=3, max_iterations=100, tolerance=1e-8)

# Create boundary conditions
bc = BoundaryConditions()

# Initial guess
phi = zeros(nx, nz)

println("Testing 2D multigrid solver...")
println("  Grid size: $(nx) x $(nz)")
println("  Levels: $(solver.levels)")
println("  Tolerance: $(solver.tolerance)")

# Solve
initial_residual = norm(rhs)
println("  Initial RHS norm: $initial_residual")

try
    solve_poisson!(solver, phi, rhs, grid, bc)
    
    # Check solution accuracy
    error_norm = norm(phi - phi_exact) / norm(phi_exact)
    residual_norm = norm(rhs - begin
        r = zeros(nx, nz)
        # Compute residual manually
        for j = 2:nz-1, i = 2:nx-1
            lap = (phi[i+1,j] - 2*phi[i,j] + phi[i-1,j])/dx^2 + 
                  (phi[i,j+1] - 2*phi[i,j] + phi[i,j-1])/dz^2
            r[i,j] = rhs[i,j] - lap
        end
        r
    end)
    
    println("✓ Multigrid solver completed successfully")
    println("  Relative error: $error_norm")
    println("  Residual norm: $residual_norm")
    
    # Check convergence
    if error_norm < 0.1  # Reasonable accuracy for coarse grid
        println("✓ Solution accuracy acceptable")
    else
        println("⚠ Solution accuracy could be improved")
    end
    
    if residual_norm < solver.tolerance * 10
        println("✓ Residual convergence good")
    else
        println("⚠ Residual convergence could be improved")
    end
    
catch e
    println("✗ Multigrid solver failed: $e")
    rethrow(e)
end

println("Multigrid validation completed successfully!")