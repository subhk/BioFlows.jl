# API Reference

This page provides a quick reference to all exported functions and types.
See the linked pages for detailed documentation.

## Simulation Types

See [Core Types](@ref) for details.

| Type | Description |
|------|-------------|
| `Simulation` | Main simulation container |
| `AMRSimulation` | Simulation with adaptive mesh refinement |
| `AMRConfig` | Configuration for AMR |
| `Flow` | Fluid field storage (velocity, pressure) |
| `AutoBody` | Implicit geometry via signed distance function |

## Simulation Control

See [Core Types](@ref) for details.

| Function | Description |
|----------|-------------|
| `sim_step!(sim)` | Advance one time step |
| `sim_step!(sim, t_end)` | Advance to target time |
| `sim_time(sim)` | Get dimensionless time (t*U/L) |
| `measure!(sim)` | Update body coefficients |
| `perturb!(sim)` | Add velocity perturbations |
| `sim_info(sim)` | Print simulation status |

## AMR Functions

See [Adaptive Mesh Refinement](@ref) for details.

| Function | Description |
|----------|-------------|
| `amr_regrid!(sim)` | Force regridding |
| `set_amr_active!(sim, bool)` | Enable/disable AMR |
| `get_refinement_indicator(sim)` | Get current indicator field |
| `num_refined_cells(grid)` | Count refined cells |
| `refinement_level(grid, i, j)` | Query cell refinement level |

## Force Diagnostics

See [Diagnostics](@ref) for details.

| Function | Description |
|----------|-------------|
| `pressure_force(sim)` | Pressure force vector |
| `viscous_force(sim)` | Viscous force vector |
| `total_force(sim)` | Total force vector |
| `force_components(sim)` | All forces + coefficients |
| `force_coefficients(sim)` | Dimensionless coefficients |
| `record_force!(history, sim)` | Append to force history |

## Vorticity Functions

See [Diagnostics](@ref) for details.

| Function | Description |
|----------|-------------|
| `vorticity_component(sim, i)` | i-th vorticity component |
| `vorticity_magnitude(sim)` | Vorticity magnitude field |
| `cell_center_velocity(sim)` | Interpolated velocity at cell centers |
| `cell_center_vorticity(sim)` | Interpolated vorticity at cell centers |

## Output

| Function | Description |
|----------|-------------|
| `CenterFieldWriter(file; interval)` | Create JLD2 snapshot writer |
| `maybe_save!(writer, sim)` | Save snapshot if interval elapsed |

## Exported Symbols

### From BioFlows.jl

```julia
# Simulation
Simulation, AbstractSimulation, sim_step!, sim_time, measure!, sim_info, perturb!

# AMR
AMRSimulation, AMRConfig, amr_regrid!, set_amr_active!, get_refinement_indicator

# Flow
Flow, mom_step!, quick, cds

# Pressure
AbstractPoisson, Poisson, MultiLevelPoisson, solver!, mult!

# Bodies
AbstractBody, AutoBody, measure_sdf!, sdf, measure

# Diagnostics
pressure_force, viscous_force, total_force, force_components, force_coefficients, record_force!
vorticity_component, vorticity_magnitude, cell_center_velocity, cell_center_vorticity
curl, ω, ω_mag

# Output
CenterFieldWriter, maybe_save!

# AMR Types
StaggeredGrid, SolutionState, RefinedGrid
FlowToGridAdapter, flow_to_staggered_grid, flow_to_solution_state, create_refined_grid
compute_body_refinement_indicator, compute_velocity_gradient_indicator
compute_vorticity_indicator, compute_combined_indicator
mark_cells_for_refinement, apply_buffer_zone!

# Utilities
L₂, BC!, @inside, inside, δ, apply!, loc, @log, set_backend, backend

# Statistics
MeanFlow, update!, uu!, uu
```
