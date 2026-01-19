# BioFlows.jl

```@raw html
<div style="text-align: center; margin: 2rem 0;">
    <p style="font-size: 1.4em; color: #555; margin-bottom: 1.5rem;">
        <strong>High-Performance Computational Fluid Dynamics in Julia</strong>
    </p>
    <p style="font-size: 1.1em; color: #666;">
        Immersed boundary methods for bio-inspired flows with GPU acceleration
    </p>
</div>
```

---

## Overview

BioFlows.jl is a pure Julia package for simulating incompressible viscous flows around complex geometries using the **Boundary Data Immersion Method (BDIM)**. It's designed for studying bio-inspired locomotion, fluid-structure interaction, and bluff body flows.

```@raw html
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin: 2rem 0;">

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 12px;">
    <h3 style="margin-top: 0; color: white;">High Performance</h3>
    <p style="margin-bottom: 0;">GPU acceleration via CUDA with single-precision (Float32) for maximum throughput. Run simulations 10-50x faster on NVIDIA GPUs.</p>
</div>

<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1.5rem; border-radius: 12px;">
    <h3 style="margin-top: 0; color: white;">Bio-Inspired</h3>
    <p style="margin-bottom: 0;">Built-in support for swimming fish, flapping wings, and flexible structures with Euler-Bernoulli beam FSI coupling.</p>
</div>

<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 1.5rem; border-radius: 12px;">
    <h3 style="margin-top: 0; color: white;">Adaptive Mesh</h3>
    <p style="margin-bottom: 0;">Automatic mesh refinement (AMR) near bodies and flow features. Patches follow moving bodies in real-time.</p>
</div>

<div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; padding: 1.5rem; border-radius: 12px;">
    <h3 style="margin-top: 0; color: white;">Easy to Use</h3>
    <p style="margin-bottom: 0;">Define geometry with simple signed distance functions. No mesh generation required - just code and run.</p>
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
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">

<a href="getting_started/" style="text-decoration: none;">
<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea;">
    <strong style="color: #333;">Getting Started</strong>
    <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 0.9em;">Installation and first simulation</p>
</div>
</a>

<a href="gpu/" style="text-decoration: none;">
<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #f5576c;">
    <strong style="color: #333;">GPU Guide</strong>
    <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 0.9em;">CUDA acceleration setup</p>
</div>
</a>

<a href="core_types/" style="text-decoration: none;">
<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #4facfe;">
    <strong style="color: #333;">Core Types</strong>
    <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 0.9em;">Simulation, Flow, AutoBody</p>
</div>
</a>

<a href="amr/" style="text-decoration: none;">
<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #43e97b;">
    <strong style="color: #333;">AMR</strong>
    <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 0.9em;">Adaptive mesh refinement</p>
</div>
</a>

<a href="examples/" style="text-decoration: none;">
<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #f093fb;">
    <strong style="color: #333;">Examples</strong>
    <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 0.9em;">Cylinders, fish, FSI</p>
</div>
</a>

<a href="api/" style="text-decoration: none;">
<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffa726;">
    <strong style="color: #333;">API Reference</strong>
    <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 0.9em;">Complete function reference</p>
</div>
</a>

</div>
```

---

## Example Gallery

### Flow Past Cylinder (Re = 100)

```julia
using BioFlows

sdf(x, t) = sqrt((x[1]-32)^2 + (x[2]-32)^2) - 8
sim = Simulation((128, 64), (1, 0), 16.0; ν=0.16, body=AutoBody(sdf))
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

sim = Simulation((128, 64), (1, 0), 16.0; ν=0.16, body=body)
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
<div style="text-align: center; margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #eee;">
    <p style="color: #888;">
        Developed by <strong>Subhajit Kar</strong> and <strong>Dibyendu Ghosh</strong>
    </p>
</div>
```
