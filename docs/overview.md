# BioFlows + WaterLily Overview

BioFlows vendors the WaterLily solver so that existing BioFlows projects can
migrate to WaterLily APIs without leaving this repository. This document mirrors
the structure of the WaterLily quickstart and highlights places where the
BioFlows wrapper adds context.

## Quickstart

```julia
using BioFlows
const WL = BioFlows.WaterLily

radius = 24 # pick a radius compatible with powers of two
center = 4radius - 1
sdf(x, t) = √((x[1] - center)^2 + (x[2] - center)^2) - radius

sim = WL.Simulation((6radius, 8radius), (1, 0), 2radius;
                    ν = 2radius / 120,
                    body = WL.AutoBody(sdf))
WL.sim_step!(sim, 1.0; remeasure = false)
```

Key takeaways:
- The constructor is identical to WaterLily’s – you are calling the vendored
  module directly.
- `WL.sim_step!` advances the simulation either a single step or to a target
  convective time depending on the arguments.
- Additional helpers (`measure!`, `perturb!`, `flood`, plotting utilities) live
  on `BioFlows.WaterLily` too.

## Example Gallery

The `examples/` directory mirrors WaterLily’s showcase scripts:

| Script                               | Description |
|-------------------------------------|-------------|
| `waterlily_circle.jl`               | 2D cylinder wake with drag/lift logging. |
| `flow_past_cylinder_2d.jl`          | Wrapper with configurable `(nx,nz)`/`(Lx,Lz)`, fixed `dt`/`final_time`, boundary controls, and centre-field snapshots. |
| `waterlily_oscillating_cylinder.jl` | Cylinder with sinusoidal cross-flow motion. |
| `waterlily_donut.jl`                | 3D torus immersed in periodic inflow. |
| `waterlily_3d_sphere.jl`            | 3D sphere wake; supports alternative element types. |

Run them with `julia --project FILE.jl`. Each script exposes a `*_sim` and
`run_*` function so you can reuse the setup programmatically.

## Diagnostics & Output

- `record_force!`, `force_components`, and `force_coefficients` provide the
  WaterLily drag/lift workflow at the BioFlows level.
- Use `vorticity_component` or `vorticity_magnitude` together with `flood` to
  mirror the WaterLily plotting notebooks.
- WaterLily’s own `save!`, `load!`, and `vtkWriter` functions remain available
  via extensions; legacy BioFlows writers have been removed.
- `CenterFieldWriter` captures cell-centred velocity and vorticity snapshots at
  fixed time intervals in JLD2 format.

## Testing

BioFlows integrates WaterLily’s upstream `test/runtests.jl`. From the project
root execute:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Missing direct dependencies in the manifest (for example `EllipsisNotation`)
can be fixed with `Pkg.resolve()` before re-running the tests.

## Compatibility Notes

- Legacy BioFlows APIs (`create_2d_simulation_config`, flexible bodies, AMR,
  etc.) are no longer shipped. They will reappear as WaterLily-friendly wrappers
  when ported.
- The `BioFlows` module re-exports common WaterLily identifiers (e.g.
  `Simulation`, `sim_step!`) for convenience.
- WaterLily’s plotting extensions (`Plots`, `Makie`, `WriteVTK`) remain optional.
  Load the corresponding packages to enable them exactly as described in the
  upstream documentation.

For more background, consult `docs/archive/waterlily_migration_map.md` or refer
to the official WaterLily documentation.
