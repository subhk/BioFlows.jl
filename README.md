# BioFlows.jl

A Julia package for computational fluid dynamics (CFD) simulations with immersed
boundary methods. BioFlows provides a complete solver for incompressible viscous
flow on Cartesian grids using the Boundary Data Immersion Method (BDIM).

## Features

- Pure Julia solver for incompressible Navier-Stokes equations
- Immersed boundary method via BDIM (Boundary Data Immersion Method)
- Implicit geometry definition through signed distance functions
- Adaptive Mesh Refinement (AMR) near bodies and flow features
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

## Quick Start

```julia
using BioFlows

# Define cylinder geometry via signed distance function
radius = 8
center = 32
sdf(x, t) = √((x[1] - center)^2 + (x[2] - center)^2) - radius

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
| `flow_past_cylinder_2d.jl` | 2D cylinder wake with configurable grid, Re, and output |
| `circle_benchmark.jl` | Simple 2D cylinder benchmark |
| `oscillating_cylinder.jl` | Cylinder with sinusoidal motion |
| `torus_3d.jl` | 3D torus in periodic inflow |
| `sphere_3d.jl` | 3D sphere wake |
| `flexible_body_pid_control.jl` | Flexible body with PID controller |

## Adaptive Mesh Refinement

```julia
config = AMRConfig(max_level=2, body_distance_threshold=3.0)
sim = AMRSimulation((128, 128), (1.0, 0.0), 16.0;
                    ν=16.0/200, body=AutoBody(sdf), amr_config=config)

# AMR regridding happens automatically
for _ in 1:1000
    sim_step!(sim; remeasure=true)
end
```

## Diagnostics & Output

```julia
# Force coefficients
coeffs = force_coefficients(sim)  # [Cd, Cl]

# Record force history
history = NamedTuple[]
record_force!(history, sim)

# Vorticity fields
ω = vorticity_component(sim, 3)  # out-of-plane for 2D
ω_mag = vorticity_magnitude(sim)

# Cell-centered snapshots to JLD2
writer = CenterFieldWriter("fields.jld2"; interval=0.1)
maybe_save!(writer, sim)
```

## Documentation

See `docs/overview.md` for comprehensive documentation including:
- Core types: `Simulation`, `Flow`, `AutoBody`
- AMR system: `AMRSimulation`, `AMRConfig`
- Diagnostics API
- Output options

## Testing

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## Contributing

1. Fork and branch from `main`
2. Add or modify functionality
3. Run `Pkg.test()` locally
4. Open a pull request

Bug reports and feature requests welcome via issues.
