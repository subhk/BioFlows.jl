# [GPU Acceleration](@id gpu)

BioFlows.jl supports GPU acceleration via CUDA for NVIDIA GPUs. This can provide **10-50x speedup** over CPU execution, especially for larger simulations.

```@raw html
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;">
    <h3 style="margin-top: 0; color: white;">Key Point</h3>
    <p style="margin-bottom: 0;">Enabling GPU is as simple as adding <code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 4px;">mem = CuArray</code> to your Simulation constructor. Everything else stays the same!</p>
</div>
```

---

## Prerequisites

### Hardware Requirements

- NVIDIA GPU with CUDA Compute Capability 5.0+ (Maxwell or newer)
- Recommended: RTX 2000/3000/4000 series, Tesla V100/A100, or Quadro series
- Minimum 4GB GPU memory (8GB+ recommended for larger simulations)

### Software Requirements

- Julia 1.9 or later
- NVIDIA CUDA Toolkit (automatically installed with CUDA.jl)
- NVIDIA GPU drivers (version 450+ recommended)

---

## Installation

### Step 1: Install CUDA.jl

```julia
using Pkg
Pkg.add("CUDA")
```

### Step 2: Verify GPU Access

```julia
using CUDA

# Check if CUDA is functional
println("CUDA functional: ", CUDA.functional())

# List available GPUs
CUDA.devices()

# Check GPU memory
CUDA.memory_status()
```

Expected output:
```
CUDA functional: true
CUDA.DeviceIterator() for 1 devices:
0. NVIDIA GeForce RTX 3080
```

### Step 3: Install BioFlows.jl

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/BioFlows.jl")
```

### Step 4: Verify Backend Setting

BioFlows uses a compile-time backend preference. For GPU support, ensure it's set to "KernelAbstractions" (the default):

```julia
using BioFlows

# Check current backend (should be "KernelAbstractions" for GPU)
println("Backend: ", BioFlows.backend)

# If it shows "SIMD", switch to KernelAbstractions:
# BioFlows.set_backend("KernelAbstractions")
# Then restart Julia
```

!!! note
    If you previously set the backend to "SIMD" for serial CPU execution, you must
    switch back to "KernelAbstractions" for GPU support. The `Simulation` constructor
    will error if there's a mismatch between the backend and array type.

---

## Basic Usage

### CPU vs GPU Comparison

```julia
using BioFlows

# Define geometry
sdf(x, t) = sqrt((x[1] - 64)^2 + (x[2] - 64)^2) - 8

# CPU Simulation (default)
sim_cpu = Simulation((128, 128), (128.0, 128.0);
                     ν = 0.01,
                     body = AutoBody(sdf))

# GPU Simulation - just add mem=CuArray
using CUDA
sim_gpu = Simulation((128, 128), (128.0, 128.0);
                     ν = 0.01,
                     body = AutoBody(sdf),
                     mem = CuArray)  # ← This is the only change!
```

### Running a GPU Simulation

```julia
using CUDA
using BioFlows

# Setup
radius, center = 16, 128
sdf(x, t) = sqrt((x[1] - center)^2 + (x[2] - center)^2) - radius

# Create GPU simulation
sim = Simulation((512, 256), (512.0, 256.0);
                 inletBC = (1.0, 0.0),
                 ν = 2radius / 200,  # Re = 200
                 body = AutoBody(sdf),
                 mem = CuArray)

# Time integration (runs on GPU)
sim_step!(sim, 10.0)

println("Simulation complete at t* = ", sim_time(sim))
```

---

## Performance Tips

### 1. Use Larger Grids

GPU acceleration is most beneficial for larger problems:

| Grid Size | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| 64×64 | 1.0s | 0.5s | 2× |
| 256×256 | 15s | 0.8s | 19× |
| 512×512 | 120s | 2.5s | 48× |
| 1024×512 | 480s | 8s | 60× |

*Times are illustrative - actual performance depends on hardware.*

### 2. Batch Operations

The GPU is most efficient when doing many operations at once. Avoid reading results too frequently:

```julia
# BAD: Inefficient - transfers data every step
for step in 1:1000
    sim_step!(sim)
    ω = vorticity_component(sim, 3)  # Transfers from GPU
    println(maximum(ω))
end

# GOOD: Efficient - transfer only when needed
for step in 1:1000
    sim_step!(sim)
    if step % 100 == 0  # Only every 100 steps
        ω = vorticity_component(sim, 3)
        println("Step $step: max ω = ", maximum(ω))
    end
end
```

### 3. Single Precision (Float32)

BioFlows uses Float32 by default, which is optimal for GPU:

```julia
# Default: Float32 (recommended for GPU)
sim = Simulation((512, 256), (512.0, 256.0); mem=CuArray, ...)

# Float64 if needed (2x slower on most GPUs)
sim = Simulation((512, 256), (512.0, 256.0); mem=CuArray, T=Float64, ...)
```

!!! tip "Float32 vs Float64"
    Most NVIDIA consumer GPUs have ~32× more Float32 throughput than Float64.
    Tesla/A100 GPUs have ~2× more. Always use Float32 unless you need extra precision.

### 4. Avoid Scalar Indexing

Never access individual GPU array elements directly:

```julia
# BAD: scalar indexing (extremely slow)
for i in 1:size(sim.flow.u, 1)
    sim.flow.u[i, 1, 1] = 0.0  # Each access is a GPU transfer!
end

# GOOD: use broadcast or kernel operations
sim.flow.u .= 0  # Single GPU operation
```

---

## GPU with AMR

Adaptive Mesh Refinement works on GPU:

```julia
using CUDA
using BioFlows

sdf(x, t) = sqrt((x[1] - 64)^2 + (x[2] - 64)^2) - 8

config = AMRConfig(
    max_level = 2,
    body_distance_threshold = 4.0,
    regrid_interval = 10
)

sim = AMRSimulation((256, 256), (256.0, 256.0);
                    inletBC = (1.0, 0.0),
                    ν = 0.01,
                    body = AutoBody(sdf),
                    amr_config = config,
                    mem = CuArray)

for step in 1:500
    sim_step!(sim; remeasure=true)
end

amr_info(sim)
```

---

## GPU with Moving Bodies

Moving bodies work seamlessly on GPU:

```julia
using CUDA
using BioFlows

# Oscillating cylinder
sdf(x, t) = sqrt(x[1]^2 + x[2]^2) - 8
map(x, t) = x .- [0, 4*sin(0.5*t)]

body = AutoBody(sdf, map)

sim = Simulation((256, 128), (256.0, 128.0);
                 inletBC = (1.0, 0.0),
                 ν = 0.08,
                 body = body,
                 mem = CuArray)

# remeasure=true updates body position each step
for step in 1:1000
    sim_step!(sim; remeasure=true)
end
```

---

## Reading Results from GPU

Results are automatically transferred from GPU when accessed:

```julia
using CUDA
using BioFlows

sim = Simulation((256, 128), (256.0, 128.0);
                 ν = 0.01, body = AutoBody(sdf), mem = CuArray)

sim_step!(sim, 5.0)

# These functions handle GPU→CPU transfer automatically
ω = vorticity_component(sim, 3)       # Returns CPU Array
vel = cell_center_velocity(sim)        # Returns CPU Array
p = cell_center_pressure(sim)          # Returns CPU Array

# Force computation also works
forces = force_components(sim)
println("Drag: ", forces.total[1])
println("Lift: ", forces.total[2])
```

### Manual Transfer

If you need direct array access:

```julia
# Transfer GPU array to CPU
u_cpu = Array(sim.flow.u)

# Transfer CPU array to GPU
u_gpu = CuArray(u_cpu)
```

---

## Saving GPU Simulation Data

Output writers work with GPU simulations:

```julia
using CUDA
using BioFlows

sim = Simulation((256, 128), (256.0, 128.0);
                 ν = 0.01, body = AutoBody(sdf), mem = CuArray)

# Create writer (saves to JLD2)
writer = CenterFieldWriter("output.jld2"; interval = 0.5)

for step in 1:1000
    sim_step!(sim)
    write!(writer, sim)  # Handles GPU→CPU transfer internally
end

close!(writer)
```

---

## 3D GPU Simulations

3D simulations benefit even more from GPU acceleration:

```julia
using CUDA
using BioFlows

# 3D sphere
center = [64, 64, 64]
sdf(x, t) = sqrt(sum((x .- center).^2)) - 12

sim = Simulation((128, 128, 128), (128.0, 128.0, 128.0);
                 inletBC = (1.0, 0.0, 0.0),
                 ν = 0.01,
                 body = AutoBody(sdf),
                 mem = CuArray)

# 3D simulations are compute-intensive - GPU is essential
sim_step!(sim, 5.0)
```

---

## Troubleshooting

### CUDA Not Found

```julia
julia> using CUDA
ERROR: CUDA.jl could not find an appropriate CUDA runtime
```

**Solution:** Install CUDA drivers from NVIDIA website, or let Julia install them:
```julia
using CUDA
CUDA.set_runtime_version!(v"12.0")  # Or your preferred version
```

### Out of GPU Memory

```
ERROR: Out of GPU memory trying to allocate X bytes
```

**Solutions:**
1. Reduce grid size
2. Use smaller `max_level` for AMR
3. Close other GPU applications
4. Check memory with `CUDA.memory_status()`

### Scalar Indexing Warning

```
Warning: Performing scalar indexing on task...
```

**Solution:** This indicates slow GPU access. Review code to avoid direct array indexing.
You can disable the warning (not recommended) with:
```julia
CUDA.allowscalar(false)  # Will error instead of warn
```

### NaN Values

If simulation produces NaN on GPU but not CPU:

1. Check for division by zero in SDF
2. Reduce time step with `fixed_Δt`
3. Ensure SDF is smooth and continuous

### Backend Mismatch Error

```
ERROR: Backend mismatch: The @loop backend is set to "SIMD"...
```

**Cause:** The compile-time backend preference is set to "SIMD" (serial CPU), but you're trying to use GPU arrays.

**Solution:**
```julia
using BioFlows
BioFlows.set_backend("KernelAbstractions")
# Restart Julia, then try again
```

---

## Benchmarking

Compare CPU vs GPU performance:

```julia
using CUDA
using BioFlows
using BenchmarkTools

sdf(x, t) = sqrt((x[1] - 64)^2 + (x[2] - 64)^2) - 8

# CPU benchmark
sim_cpu = Simulation((256, 256), (256.0, 256.0); ν=0.01, body=AutoBody(sdf))
@btime sim_step!($sim_cpu) setup=(sim_cpu.flow.Δt[end]=0.1)

# GPU benchmark
sim_gpu = Simulation((256, 256), (256.0, 256.0); ν=0.01, body=AutoBody(sdf), mem=CuArray)
CUDA.@sync @btime sim_step!($sim_gpu) setup=(sim_gpu.flow.Δt[end]=0.1)
```

!!! note
    Use `CUDA.@sync` when benchmarking GPU code to ensure operations complete.

---

## Multi-GPU (Advanced)

For very large simulations, BioFlows supports MPI for multi-GPU setups:

```julia
# Run with: mpiexecjl -n 4 julia script.jl
using MPI
using CUDA
using BioFlows

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Each rank uses one GPU
device!(rank % length(devices()))

# Create distributed simulation
sim = Simulation((1024, 512), (1024.0, 512.0);
                 ν = 0.01,
                 body = AutoBody(sdf),
                 mem = CuArray)

# Simulation automatically handles domain decomposition
sim_step!(sim, 10.0)

MPI.Finalize()
```

---

## Summary

```@raw html
<div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
<h3 style="margin-top: 0;">Quick Reference</h3>

<table style="width: 100%;">
<tr>
    <td><strong>Enable GPU</strong></td>
    <td><code>Simulation(...; mem = CuArray)</code></td>
</tr>
<tr>
    <td><strong>Check GPU</strong></td>
    <td><code>CUDA.functional()</code></td>
</tr>
<tr>
    <td><strong>Memory status</strong></td>
    <td><code>CUDA.memory_status()</code></td>
</tr>
<tr>
    <td><strong>Transfer to CPU</strong></td>
    <td><code>Array(gpu_array)</code></td>
</tr>
<tr>
    <td><strong>Best precision</strong></td>
    <td>Float32 (default)</td>
</tr>
</table>
</div>
```

**Remember:** GPU acceleration shines with larger grids (256×256+). For small test cases, CPU may be faster due to transfer overhead.
