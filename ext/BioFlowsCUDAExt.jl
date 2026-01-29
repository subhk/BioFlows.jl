"""
    BioFlowsCUDAExt

Extension module that activates when CUDA.jl is loaded alongside BioFlows.jl.

This extension:
- Prints an info message confirming CUDA GPU support is available
- Ensures KernelAbstractions.jl can use the CUDA backend for `@loop` macros
- Provides GPU synchronization and array transfer utilities

# GPU Performance Notes
- All `@loop` macros automatically compile to CUDA kernels via KernelAbstractions.jl
- LinearAlgebra operations (dot products, norms) use cuBLAS for optimal performance
- Reductions (sum, maximum) are handled efficiently by CUDA.jl

Note: You must explicitly `using CUDA` to access `CuArray` for the `mem` parameter.

# Usage
```julia
using BioFlows
using CUDA  # Required to access CuArray

# Create a GPU-accelerated simulation
sim = Simulation((256, 128), (2.0, 1.0);
    inletBC = (1.0, 0.0),
    ν = 1e-6,
    mem = CuArray  # Use GPU memory
)
```

# Debugging GPU Issues
To catch scalar indexing issues (performance problems on GPU), run:
```julia
CUDA.allowscalar(false)
```
This will error on any scalar GPU access, helping identify code paths
that need GPU optimization.
"""
module BioFlowsCUDAExt

using BioFlows
using CUDA
using KernelAbstractions: synchronize, get_backend

function __init__()
    if CUDA.functional()
        @info "BioFlows: CUDA GPU backend enabled. Use `mem=CuArray` in Simulation constructor."
    else
        @warn "BioFlows: CUDA extension loaded but CUDA is not functional. GPU acceleration unavailable."
    end
end

"""
    gpu_sync!(arr)

Synchronize the GPU backend associated with the given array.
This ensures all GPU operations on the array have completed before
reading results back to CPU.

For CPU arrays, this is a no-op.
"""
function BioFlows.gpu_sync!(arr::CuArray)
    synchronize(get_backend(arr))
    return nothing
end

"""
    to_cpu(arr)

Convert a GPU array to a CPU Array. For CPU arrays, returns a copy.
This function ensures GPU operations are synchronized before transfer.
"""
function BioFlows.to_cpu(arr::CuArray{T}) where T
    synchronize(get_backend(arr))
    return Array(arr)
end

end # module
