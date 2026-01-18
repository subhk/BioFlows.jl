"""
    BioFlowsCUDAExt

Extension module to enable CUDA GPU support for BioFlows.jl.

When CUDA.jl is loaded, this extension provides:
- Re-exports `CuArray` for convenient GPU array creation
- Ensures KernelAbstractions.jl CUDA backend is available

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
"""
module BioFlowsCUDAExt

using BioFlows
using CUDA
using CUDA: CuArray

# Re-export CuArray through BioFlows when CUDA is loaded
function __init__()
    @info "BioFlows: CUDA GPU backend enabled. Use `mem=CuArray` in Simulation constructor."
end

# Export CuArray for convenience
export CuArray

end # module
