# BioFlows.jl

[![CI](https://github.com/subhk/BioFlows.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/subhk/BioFlows.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/subhk/BioFlows.jl/actions/workflows/documentation.yml/badge.svg)](https://subhk.github.io/BioFlows.jl/dev/)

A Julia package for computational fluid dynamics (CFD) simulations with immersed
boundary methods. BioFlows provides a complete solver for incompressible viscous
flow on Cartesian grids using the Boundary Data Immersion Method (BDIM).

## Features

- Pure Julia solver for incompressible Navier-Stokes equations
- Immersed boundary method via BDIM (Boundary Data Immersion Method)
- Implicit geometry definition through signed distance functions
- Adaptive Mesh Refinement (AMR) near bodies and flow features

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/subhk/BioFlows.jl")
```

Or activate the project locally:

```bash
git clone https://github.com/subhk/BioFlows.jl.git
cd BioFlows.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start

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

## Examples

Run the example scripts:

```bash
julia --project examples/flow_past_cylinder_2d.jl
julia --project examples/circle_benchmark.jl
julia --project examples/oscillating_cylinder.jl
julia --project examples/torus_3d.jl
julia --project examples/sphere_3d.jl
```

| Example | Description |
|---------|-------------|
| `flow_past_cylinder_2d.jl` | Full 2D cylinder simulation with configurable grid, Re, boundary conditions, and JLD2 output |
| `circle_benchmark.jl` | Simple 2D cylinder benchmark with force logging |
| `oscillating_cylinder.jl` | Cylinder with sinusoidal cross-flow motion |
| `torus_3d.jl` | 3D torus in periodic inflow |
| `sphere_3d.jl` | 3D sphere wake simulation |

## Adaptive Mesh Refinement

```julia
using BioFlows

# Define geometry
sdf(x, t) = sqrt((x[1] - 64)^2 + (x[2] - 64)^2) - 8

# Configure AMR
config = AMRConfig(
    max_level = 2,                  # Refinement levels (2x, 4x resolution)
    body_distance_threshold = 3.0,  # Refine within 3 cells of body
    regrid_interval = 10            # Check every 10 steps
)

# Create AMR-enabled simulation
sim = AMRSimulation((128, 128), (1.0, 0.0), 16.0;
                    ν = 16.0/200,
                    body = AutoBody(sdf),
                    amr_config = config)

# AMR regridding happens automatically during time stepping
for _ in 1:1000
    sim_step!(sim; remeasure=true)
end

println("Refined cells: ", num_refined_cells(sim.refined_grid))
```

## Diagnostics & Output

```julia
# Force coefficients (drag, lift)
coeffs = force_coefficients(sim)

# Record force history over time
history = NamedTuple[]
for step in 1:100
    sim_step!(sim)
    record_force!(history, sim)
end

# Vorticity fields
ω = vorticity_component(sim, 3)   # Out-of-plane for 2D
ω_mag = vorticity_magnitude(sim)

# Cell-centered velocity/vorticity snapshots to JLD2
writer = CenterFieldWriter("fields.jld2"; interval=0.1)
for step in 1:100
    sim_step!(sim)
    file_save!(writer, sim)
end
```

## Contributing

1. Fork and branch from `main`
2. Add or modify functionality
3. Run `Pkg.test()` locally
4. Open a pull request

Bug reports and feature requests welcome via [GitHub Issues](https://github.com/subhk/BioFlows.jl/issues).

