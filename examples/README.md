# BioFlows.jl Examples

This directory contains example scripts demonstrating BioFlows capabilities.

## 2D Examples

- `circle_benchmark.jl`: 2D Kármán street over a fixed cylinder. Captures drag
  and lift coefficients using `pressure_force`.
- `flow_past_cylinder_2d.jl`: Comprehensive 2D cylinder example with configurable
  domain `(Lx,Lz)`, grid `(nx,nz)`, Reynolds number, fixed time step, boundary
  controls (`uBC`, `perdir`, `exitBC`), built-in force diagnostics, and optional
  cell-centered velocity/vorticity snapshots to JLD2.
- `oscillating_cylinder.jl`: Cross-flow oscillation of a cylinder with prescribed
  sinusoidal motion (Strouhal `St = 0.2`). Tracks total force coefficients and
  body displacement over time.

## 3D Examples

- `torus_3d.jl`: 3D torus (donut) immersed in a periodic stream with force logging.
- `sphere_3d.jl`: 3D sphere wake in uniform inflow. Supports custom floating-point
  types (`Float32`, `Float64`) and memory backends (CPU/GPU).

## Running Examples

Run any script directly from the project root:

```sh
julia --project examples/circle_benchmark.jl
julia --project examples/flow_past_cylinder_2d.jl
julia --project examples/oscillating_cylinder.jl
julia --project examples/torus_3d.jl
julia --project examples/sphere_3d.jl
```

Each script exposes a `*_sim` constructor and a `run_*` helper so you can import
and reuse them from notebooks or other scripts:

```julia
include("examples/circle_benchmark.jl")
sim = circle_sim(; Re=200)
sim_step!(sim, 1.0)
```

## Example Output

The `flow_past_cylinder_2d.jl` example produces:
- Force history with drag/lift coefficients
- Cell-centered velocity and vorticity snapshots (JLD2 format)
- Diagnostic statistics (mean Cd, lift RMS, CFL)
