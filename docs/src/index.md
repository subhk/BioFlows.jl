# BioFlows.jl

*High-Performance Computational Fluid Dynamics in Julia*

BioFlows.jl is a pure Julia package for simulating incompressible viscous flows around complex geometries using the **Boundary Data Immersion Method (BDIM)**. It's designed for studying bio-inspired locomotion, fluid-structure interaction, and bluff body flows.

---

## Highlights

```@raw html
<div class="feature-grid">
    <div class="feature-card gradient-card gradient-primary">
        <h3>High Performance</h3>
        <p>GPU acceleration via CUDA with single-precision (Float32) for maximum throughput. Run simulations 10-50x faster on NVIDIA GPUs.</p>
    </div>
    <div class="feature-card gradient-card gradient-accent">
        <h3>Bio-Inspired</h3>
        <p>Built-in support for swimming fish, flapping wings, and flexible structures with Euler-Bernoulli beam FSI coupling.</p>
    </div>
    <div class="feature-card gradient-card gradient-info">
        <h3>Adaptive Mesh</h3>
        <p>Automatic mesh refinement (AMR) near bodies and flow features. Patches follow moving bodies in real-time.</p>
    </div>
    <div class="feature-card gradient-card gradient-success">
        <h3>Easy to Use</h3>
        <p>Define geometry with simple signed distance functions. No mesh generation required - just code and run.</p>
    </div>
</div>
```

---

## Key Features

| Feature | Description |
|:--------|:------------|
| **Navier-Stokes Solver** | Incompressible flow with projection method on staggered grids |
| **BDIM** | Boundary Data Immersion Method for smooth immersed boundaries |
| **GPU Support** | CUDA acceleration with `mem=CuArray` - single line change |
| **AMR** | Adaptive mesh refinement with automatic body tracking |
| **FSI** | Fluid-structure interaction via Euler-Bernoulli beam |
| **Moving Bodies** | Time-dependent SDF for rigid and flexible motion |
| **2D/3D** | Full support for both 2D and 3D simulations |
| **Output** | JLD2 snapshots and VTK export for ParaView |

---

## Quick Start

```julia
using BioFlows

# Define a cylinder using signed distance function
radius, center = 8, 32
sdf(x, t) = sqrt((x[1] - center)^2 + (x[2] - center)^2) - radius

# Create simulation at Re = 100
sim = Simulation((128, 64), (128.0, 64.0);
                 inletBC = (1.0, 0.0),
                 ν = 2radius / 100,  # Re = U*D/ν = 100
                 body = AutoBody(sdf))

# Run to t* = 5 (convective time units)
sim_step!(sim, 5.0)

# Check results
println("Time: t* = ", sim_time(sim))
```

**That's it!** No mesh generation, no complex setup - just define your geometry and simulate.

---

## GPU Acceleration

Run on NVIDIA GPU with a single change:

```julia
using CUDA
using BioFlows

sim = Simulation((512, 256), (512.0, 256.0);
                 ν = 0.01,
                 body = AutoBody(sdf),
                 mem = CuArray)  # <- GPU enabled!

sim_step!(sim, 10.0)
```

See the [GPU Guide](@ref gpu) for detailed instructions and performance tips.

---

## Installation

### From GitHub

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/BioFlows.jl")
```

### For GPU Support

```julia
using Pkg
Pkg.add("CUDA")
Pkg.add(url="https://github.com/subhk/BioFlows.jl")
```

### Local Development

```bash
git clone https://github.com/subhk/BioFlows.jl.git
cd BioFlows.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

---

## Documentation

```@raw html
<div class="quick-links">
    <a href="getting_started/" class="quick-link">
        <strong>Getting Started</strong>
        <span>Installation and first simulation</span>
    </a>
    <a href="gpu/" class="quick-link">
        <strong>GPU Guide</strong>
        <span>CUDA acceleration setup</span>
    </a>
    <a href="core_types/" class="quick-link">
        <strong>Core Types</strong>
        <span>Simulation, Flow, AutoBody</span>
    </a>
    <a href="amr/" class="quick-link">
        <strong>AMR</strong>
        <span>Adaptive mesh refinement</span>
    </a>
    <a href="examples/" class="quick-link">
        <strong>Examples</strong>
        <span>Cylinders, fish, FSI</span>
    </a>
    <a href="api/" class="quick-link">
        <strong>API Reference</strong>
        <span>Complete function reference</span>
    </a>
</div>
```

---

## Example Gallery

### Flow Past Cylinder (Re = 100)

```julia
using BioFlows

sdf(x, t) = sqrt((x[1]-32)^2 + (x[2]-32)^2) - 8
sim = Simulation((128, 64), (128.0, 64.0);
                 inletBC = (1.0, 0.0),
                 ν = 0.16,
                 body = AutoBody(sdf))
sim_step!(sim, 10.0)

# Visualize vorticity
ω = vorticity_component(sim, 3)
```

### Swimming Fish (Carangiform)

```julia
using BioFlows
include("examples/swimming_fish.jl")

sim, history = run_swimming_fish(
    fish_length = 0.2,
    amplitude = 0.1,
    frequency = 1.0,
    amplitude_envelope = :carangiform
)
```

### Oscillating Cylinder

```julia
using BioFlows

sdf(x, t) = sqrt(x[1]^2 + x[2]^2) - 8
map(x, t) = x .- [0, 4*sin(0.5*t)]  # Vertical oscillation
body = AutoBody(sdf, map)

sim = Simulation((128, 64), (128.0, 64.0);
                 inletBC = (1.0, 0.0),
                 ν = 0.16,
                 body = body)
sim_step!(sim, 20.0; remeasure=true)
```

---

## Citation

If you use BioFlows.jl in your research, please cite:

```bibtex
@software{bioflows2024,
  author = {Kar, Subhajit and Ghosh, Dibyendu},
  title = {BioFlows.jl: Computational Fluid Dynamics with Immersed Boundary Methods},
  year = {2024},
  url = {https://github.com/subhk/BioFlows.jl}
}
```

---

## License

BioFlows.jl is released under the MIT License.

---

```@raw html
<div style="text-align: center; margin-top: 3rem; padding-top: 2rem; border-top: 1px solid var(--bio-gray-200);">
    <p style="color: var(--bio-gray-500); font-size: 0.9rem;">
        Developed by <strong>Subhajit Kar</strong> and <strong>Dibyendu Ghosh</strong>
    </p>
</div>
```
