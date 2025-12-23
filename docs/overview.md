# BioFlows.jl Documentation

BioFlows.jl is a Julia package for computational fluid dynamics (CFD) simulations
with immersed boundary methods. It provides a complete solver for incompressible
viscous flow on Cartesian grids using the Boundary Data Immersion Method (BDIM).

## Quickstart

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

# Query simulation state
println("Current time: ", sim_time(sim))
```

Key concepts:
- `Simulation` wraps `Flow` (velocity/pressure fields) + `AbstractPoisson` (pressure solver) + `AbstractBody` (geometry)
- `sim_step!(sim, t_end)` integrates until dimensionless time `t*U/L = t_end`
- `sim_step!(sim)` advances a single time step
- `AutoBody(sdf)` defines geometry implicitly via a signed distance function

## Core Types

### Simulation

```julia
Simulation(dims::NTuple, inletBC, L::Number;
           U=norm(inletBC), Δt=0.25, ν=0., ϵ=1, g=nothing,
           perdir=(), outletBC=false,
           body::AbstractBody=NoBody(),
           T=Float32, mem=Array)
```

- `dims`: Grid dimensions `(nx, nz)` for 2D or `(nx, ny, nz)` for 3D
- `inletBC`: Inlet boundary velocity:
  - `Tuple`: Constant velocity, e.g., `(1.0, 0.0)` for uniform flow
  - `Function(i,x,t)`: Spatially/temporally varying, where `i`=component, `x`=position, `t`=time
- `L`: Length scale for non-dimensionalization
- `U`: Velocity scale (auto-computed from `inletBC` if constant, **required** if function)
- `ν`: Kinematic viscosity (`Re = U*L/ν`)
- `ϵ`: BDIM kernel width
- `perdir`: Periodic directions, e.g. `(2,)` for z-periodic
- `outletBC`: Enable convective outlet boundary in x-direction
- `body`: Immersed geometry (`AutoBody`, `NoBody`, etc.)
- `T`: Float type (`Float32` or `Float64`)
- `mem`: Array backend (`Array` for CPU, `CuArray` for GPU)

**Inlet BC Examples:**
```julia
# Uniform inlet
sim = Simulation(dims, (1.0, 0.0), L; ν=0.01)

# Parabolic profile (varies with z in 2D)
H = Lz / 2
inletBC(i, x, t) = i == 1 ? 1.5 * (1 - ((x[2] - H) / H)^2) : 0.0
sim = Simulation(dims, inletBC, L; U=1.5, ν=0.01)
```

### Flow

The `Flow` struct holds all fluid fields:
- `u`: Velocity vector field
- `p`: Pressure scalar field
- `σ`: Divergence field
- `V`, `μ₀`, `μ₁`: BDIM moment fields for immersed boundaries

### AutoBody

```julia
AutoBody(sdf, map=(x,t)->x; compose=true)
```

Define geometry implicitly:
- `sdf(x,t)`: Signed distance function (negative inside body)
- `map(x,t)`: Coordinate mapping for moving/deforming bodies
- `compose`: Auto-compose `sdf∘map` when true

Example — oscillating cylinder:
```julia
sdf(x, t) = √(x[1]^2 + x[2]^2) - radius
map(x, t) = x .- [0, A*sin(ω*t)]  # vertical oscillation
body = AutoBody(sdf, map)
```

## Adaptive Mesh Refinement (AMR)

BioFlows includes AMR support for efficient resolution near bodies and flow features.

### AMRSimulation

```julia
config = AMRConfig(
    max_level = 2,                    # Max refinement (2x, 4x, ...)
    body_distance_threshold = 3.0,    # Refine within 3 cells of body
    velocity_gradient_threshold = 1.0,
    vorticity_threshold = 1.0,
    regrid_interval = 10,             # Steps between regrid checks
    buffer_size = 1,                  # Buffer cells around refined regions
    body_weight = 0.5,                # Weight for body proximity indicator
    gradient_weight = 0.3,
    vorticity_weight = 0.2
)

sim = AMRSimulation((128, 128), (1.0, 0.0), 16.0;
                    Re=200, body=AutoBody(sdf), amr_config=config)

# AMR regridding happens automatically during sim_step!
for _ in 1:1000
    sim_step!(sim; remeasure=true)
end

# Check refinement status
println("Refined cells: ", num_refined_cells(sim.refined_grid))
```

### AMR Types

- `StaggeredGrid`: MAC grid with face-centered velocities, cell-centered pressure
- `SolutionState`: Container for `u`, `v`, `w`, `p` on staggered grid
- `RefinedGrid`: Tracks refined cells and local sub-grids

## Diagnostics

### Force Computation

```julia
# Get force components
components = force_components(sim; ρ=1.0, reference_area=sim.L)
# Returns: (pressure, viscous, total, coefficients)

# Get dimensionless force coefficients
coeffs = force_coefficients(sim)  # [Cd, Cl] or [Cd, Cl, Cside]

# Record force history
history = NamedTuple[]
for step in 1:1000
    sim_step!(sim)
    record_force!(history, sim)
end
```

### Vorticity

```julia
# Single component (use 3 for out-of-plane in 2D)
ω3 = vorticity_component(sim, 3)

# Magnitude
ω_mag = vorticity_magnitude(sim)

# Cell-centered fields for visualization
vel = cell_center_velocity(sim)
vort = cell_center_vorticity(sim)
pres = cell_center_pressure(sim)
```

### Diagnostic Summary

```julia
diag = compute_diagnostics(sim)
# Returns: (max_u, max_w, max_v, CFL, Δt, length_scale, grid)
```

## Output

### CenterFieldWriter (JLD2)

Save cell-centered velocity and vorticity snapshots at regular intervals:

```julia
writer = CenterFieldWriter("fields.jld2"; interval=0.1)

for step in 1:1000
    sim_step!(sim)
    file_save!(writer, sim)  # Saves when interval elapses
end

println("Saved $(writer.samples) snapshots")
```

### VTK Output (Extension)

Load `WriteVTK` to enable VTK output:

```julia
using WriteVTK
using BioFlows

# VTK writer becomes available
vtkWriter(sim, "output")
```

## Example Gallery

| Script | Description |
|--------|-------------|
| `flow_past_cylinder_2d.jl` | Full 2D cylinder simulation with configurable grid, Re, boundary conditions, and JLD2 snapshot output |
| `circle_benchmark.jl` | Simple 2D cylinder benchmark with force logging |
| `oscillating_cylinder.jl` | Cylinder with sinusoidal cross-flow motion |
| `torus_3d.jl` | 3D torus in periodic inflow |
| `sphere_3d.jl` | 3D sphere wake simulation |

Run examples:
```bash
julia --project examples/flow_past_cylinder_2d.jl
julia --project examples/circle_benchmark.jl
julia --project examples/oscillating_cylinder.jl
julia --project examples/torus_3d.jl
julia --project examples/sphere_3d.jl
```

## Testing

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## API Reference

### Simulation Control
- `sim_step!(sim, t_end)` — Integrate to dimensionless time
- `sim_step!(sim)` — Single time step
- `sim_time(sim)` — Current dimensionless time `t*U/L`
- `time(sim.flow)` — Current simulation time `t`
- `measure!(sim)` — Update body coefficients
- `perturb!(sim; noise=0.1)` — Add velocity perturbations
- `sim_info(sim)` — Print simulation status

### AMR Control
- `amr_regrid!(sim)` — Force regridding
- `set_amr_active!(sim, bool)` — Enable/disable AMR
- `get_refinement_indicator(sim)` — Current indicator field
- `num_refined_cells(grid)` — Count refined cells
- `refinement_level(grid, i, j)` — Query cell refinement
- `amr_info(sim)` — Print AMR status
- `check_divergence(sim)` — Check velocity divergence

### Diagnostics
- `pressure_force(sim)` — Pressure force vector
- `viscous_force(sim)` — Viscous force vector
- `total_force(sim)` — Total force vector
- `force_components(sim)` — All forces + coefficients
- `force_coefficients(sim)` — Dimensionless coefficients
- `record_force!(history, sim)` — Append to history
- `vorticity_component(sim, i)` — i-th vorticity component
- `vorticity_magnitude(sim)` — Vorticity magnitude field
- `cell_center_velocity(sim)` — Interpolated velocity
- `cell_center_vorticity(sim)` — Interpolated vorticity
- `cell_center_pressure(sim)` — Cell-centered pressure
- `compute_diagnostics(sim)` — Summary statistics
- `summarize_force_history(history)` — Force statistics

### Output
- `CenterFieldWriter(filename; interval)` — Create writer
- `file_save!(writer, sim)` — Conditional snapshot
- `save!(sim, fname)` — Save simulation state (requires JLD2)
- `load!(sim; fname)` — Load simulation state
- `vtkWriter(sim, path)` — VTK output (requires WriteVTK)
