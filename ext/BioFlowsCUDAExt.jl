"""
    BioFlowsCUDAExt

Extension module to enable CUDA GPU support for BioFlows.jl.

When CUDA.jl is loaded, this extension provides:
- Re-exports `CuArray` for convenient GPU array creation
- Ensures KernelAbstractions.jl CUDA backend is available
- Sets CUDA_LOADED flag for detection by gpu_backend()

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

# Flag to indicate CUDA is loaded and functional
# This is checked by gpu_backend() in the main module
const CUDA_LOADED = Ref(false)

function __init__()
    CUDA_LOADED[] = CUDA.functional()
    if CUDA_LOADED[]
        @info "BioFlows: CUDA GPU backend enabled. Use `mem=CuArray` in Simulation constructor."
    else
        @warn "BioFlows: CUDA extension loaded but CUDA is not functional. GPU acceleration unavailable."
    end
end

# Export CuArray for convenience
export CuArray

end # module
