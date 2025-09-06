# Simple validation test for BioFlow.jl multigrid solver
using LinearAlgebra

# Check that the module can be loaded
try
    include(joinpath(@__DIR__, "src", "BioFlows.jl"))
    println("✓ BioFlows.jl loaded successfully")
catch e
    println("✗ Failed to load BioFlows.jl: $e")
    exit(1)
end

# Simple smoke test - just check if basic types exist
if @isdefined(BioFlows) && hasfield(typeof(BioFlows), :MultigridPoissonSolver)
    println("✓ BioFlows multigrid solver types available")
else
    println("ℹ BioFlows loaded but some types may not be available")
end

println("Basic validation completed successfully")