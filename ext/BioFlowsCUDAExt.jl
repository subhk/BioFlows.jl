"""
    BioFlowsCUDAExt

Extension module to enable CUDA GPU support for BioFlows.jl.

When CUDA.jl is loaded, this extension provides:
- Re-exports `CuArray` for convenient GPU array creation
- Ensures KernelAbstractions.jl CUDA backend is available

The main module's `gpu_backend()` function detects CUDA availability by:
1. Checking if this extension is loaded via `Base.get_extension()`
2. Calling `CUDA.functional()` through the extension

# Usage
```julia
using BioFlows
using CUDA

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
using CUDA: CuArray

function __init__()
    if CUDA.functional()
        @info "BioFlows: CUDA GPU backend enabled. Use `mem=CuArray` in Simulation constructor."
    else
        @warn "BioFlows: CUDA extension loaded but CUDA is not functional. GPU acceleration unavailable."
    end
end

# Export CuArray for convenience
export CuArray

end # module
