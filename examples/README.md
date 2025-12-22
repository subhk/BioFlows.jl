# BioFlows.jl Examples

This directory contains example scripts demonstrating BioFlows capabilities.

## 2D Examples

- `circle_benchmark.jl`: 2D Kármán street over a fixed cylinder. Captures drag
  and lift coefficients.
- `flow_past_cylinder_2d.jl`: Comprehensive 2D cylinder example with configurable
  domain `(Lx,Lz)`, grid `(nx,nz)`, fixed time step `dt`, optional `final_time`,
  boundary controls (`uBC`, `perdir`, `exitBC`), built-in diagnostics, and
  optional cell-centered velocity/vorticity snapshots.
- `oscillating_cylinder.jl`: Cross-flow oscillation of a cylinder with prescribed
  sinusoidal motion (Strouhal `St = 0.2`). Tracks total force coefficients and
  body displacement.

## 3D Examples

- `torus_3d.jl`: 3D torus immersed in a periodic stream with force logging.
- `sphere_3d.jl`: 3D sphere wake in uniform inflow. Supports custom floating-point
  types and memory backends.

## Running Examples

Run any script directly:
```sh
julia --project examples/circle_benchmark.jl
julia --project examples/flow_past_cylinder_2d.jl
julia --project examples/oscillating_cylinder.jl
julia --project examples/torus_3d.jl
julia --project examples/sphere_3d.jl
```

Each script exposes a `*_sim` constructor and a `run_*` helper so you can reuse
them from notebooks without triggering the default run.

## Visualization & Utilities

- `flexible_body_pid_control.jl`: Flexible body array with PID coordination.
- `animate_vorticity_cylinder.jl` / `plot_vorticity_cylinder.jl`: Vorticity visualization.
- `plot.jl`: Convenience plotting helpers.
