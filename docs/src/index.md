# BioFlows.jl

*A Julia package for computational fluid dynamics with immersed boundary methods*

BioFlows.jl provides a complete solver for incompressible viscous flow on Cartesian
grids using the Boundary Data Immersion Method (BDIM).

## Features

- Pure Julia solver for incompressible Navier-Stokes equations
- Immersed boundary method via BDIM (Boundary Data Immersion Method)
- Implicit geometry definition through signed distance functions
- Adaptive Mesh Refinement (AMR) near bodies and flow features
- **Fluid-Structure Interaction (FSI)** via Euler-Bernoulli beam with Hermite FEM
- CPU and GPU execution via KernelAbstractions.jl
- MPI support for distributed computing
- Built-in diagnostics: forces, vorticity, cell-centered fields
- JLD2 and VTK output (via extensions)

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/subhk/BioFlows.jl")
```

Or activate the project locally:

```bash
julia --project
julia> ]instantiate
```

## Quick Example

```julia
using BioFlows

# Define cylinder geometry via signed distance function
radius = 8
center = 32
sdf(x, t) = sqrt((x[1] - center)^2 + (x[2] - center)^2) - radius

# Create simulation: domain (nx, nz), boundary velocity, length scale
sim = Simulation((128, 64), (1, 0), 2radius;
                 ν = 2radius / 100,    # Re = 100
                 body = AutoBody(sdf))

# Advance to t*U/L = 1.0 (convective time units)
sim_step!(sim, 1.0; remeasure=false)

# Check simulation state
println("Time: ", sim_time(sim))
```

## Documentation

```@contents
Pages = [
    "getting_started.md",
    "codebase_structure.md",
    "numerical_methods.md",
    "core_types.md",
    "amr.md",
    "diagnostics.md",
    "examples.md",
    "api.md",
]
Depth = 2
```

## Authors

- Subhajit Kar
- Dibyendu Ghosh
