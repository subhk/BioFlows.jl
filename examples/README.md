# BioFlows.jl Examples

The `examples/` directory now mirrors the official WaterLily show‑case cases and
keeps a few legacy BioFlows demonstrations for comparison.

## WaterLily Parity
- `waterlily_circle.jl`: 2D Kármán street over a fixed cylinder. Captures drag
  and lift coefficients exactly as the WaterLily quickstart.
- `flow_past_cylinder_2d.jl`: Convenience wrapper around the circle benchmark
  with configurable domain `(Lx,Lz)`, grid `(nx,nz)`, fixed time step `dt`,
  optional `final_time`, and boundary controls (`uBC`, `perdir`, `exitBC`)
  plus built-in diagnostics and optional cell-centered velocity/vorticity
  snapshots.
- `waterlily_oscillating_cylinder.jl`: Cross-flow oscillation of a cylinder
  with prescribed sinusoidal motion (Strouhal `St = 0.2`). Tracks total force
  coefficients and body displacement.
- `waterlily_donut.jl`: 3D torus immersed in a periodic stream; reproduces the
  donut benchmark from WaterLily-Examples with force logging.
- `waterlily_3d_sphere.jl`: 3D sphere wake in uniform inflow. Matches the
  README sample and supports custom floating-point types and memory backends.

Run any script directly, e.g.
```sh
julia --project examples/waterlily_circle.jl
```

Each script exposes a `*_sim` constructor and a `run_*` helper so you can reuse
them from notebooks without triggering the default run.

## Legacy Utilities
- `final_fixed_cylinder.jl`: Original BioFlows rigid-body driver.
- `flexible_body_pid_control.jl`: Flexible body array with PID coordination.
- `animate_vorticity_cylinder.jl` / `plot_vorticity_cylinder.jl`: Post-processing tools for NetCDF outputs.
- `plot.jl`: Convenience plotting helpers shared by historical scripts.

These files remain for reference while the WaterLily port matures; they are not
covered by the new test suite.
