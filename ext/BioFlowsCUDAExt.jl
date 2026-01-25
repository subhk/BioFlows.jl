"""
    BioFlowsCUDAExt

Extension module that activates when CUDA.jl is loaded alongside BioFlows.jl.

This extension:
- Prints an info message confirming CUDA GPU support is available
- Ensures KernelAbstractions.jl can use the CUDA backend for `@loop` macros

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

function __init__()
    if CUDA.functional()
        @info "BioFlows: CUDA GPU backend enabled. Use `mem=CuArray` in Simulation constructor."
    else
        @warn "BioFlows: CUDA extension loaded but CUDA is not functional. GPU acceleration unavailable."
    end
end

end # module
