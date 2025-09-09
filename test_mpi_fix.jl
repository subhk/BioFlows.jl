# Test if the MPI 2D fix works by creating state and accessing w field
include("src/core/types.jl")

# Create a minimal test MPI decomposition structure
struct TestDecomp <: AbstractMPIDecomposition
    nx_local_with_ghosts::Int
    nz_local_with_ghosts::Int
end

# Test creating an MPISolutionState2D
decomp = TestDecomp(10, 8)
state = MPISolutionState2D(decomp, Float64)

println("MPISolutionState2D created successfully!")
println("u field size: ", size(state.u))
println("w field size: ", size(state.w))
println("p field size: ", size(state.p))

# Test that we can access w field (this was the failing access)
state.w[1, 1] = 42.0
println("state.w[1,1] = ", state.w[1, 1])

# Test that there's no v field (should error if we try to access it)
try 
    state.v[1, 1] = 1.0
    println("ERROR: v field should not exist!")
catch e
    println("GOOD: v field correctly does not exist: ", e)
end

println("Test completed successfully!")