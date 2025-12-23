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
| `amr_info(sim)` | Print AMR status and statistics |
| `check_divergence(sim)` | Check velocity divergence on all levels |
| `amr_cfl(flow, cp)` | Compute CFL considering refined patches |
| `synchronize_base_and_patches!(flow, cp)` | Sync data between base and patches |

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
| `cell_center_pressure(sim)` | Pressure at cell centers |

## Output

| Function | Description |
|----------|-------------|
| `CenterFieldWriter(file; interval)` | Create JLD2 snapshot writer |
| `ForceWriter(file; interval)` | Create CSV force coefficient writer |
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
vorticity_component, vorticity_magnitude, cell_center_velocity, cell_center_vorticity, cell_center_pressure
curl, ω, ω_mag
compute_diagnostics, summarize_force_history

# Output
CenterFieldWriter, ForceWriter, maybe_save!

# AMR Types
StaggeredGrid, SolutionState, RefinedGrid, GridType, TwoDimensional, ThreeDimensional
is_2d, is_3d, num_refined_cells, refinement_level, domain_size, cell_volume
FlowToGridAdapter, flow_to_staggered_grid, flow_to_solution_state, create_refined_grid
compute_body_refinement_indicator, compute_velocity_gradient_indicator
compute_vorticity_indicator, compute_combined_indicator
mark_cells_for_refinement, apply_buffer_zone!

# AMR Composite Solver
CompositePoisson, PatchPoisson, RefinedVelocityField, RefinedVelocityPatch
add_patch!, remove_patch!, get_patch, clear_patches!, has_patches, num_patches
create_patches!, update_patches!, ensure_proper_nesting!
amr_project!, amr_mom_step!, check_amr_divergence, regrid_amr!
amr_cfl, synchronize_base_and_patches!, interpolate_velocity_to_patches!
amr_info, check_divergence

# Utilities
L₂, BC!, @inside, inside, δ, apply!, loc, @log, set_backend, backend

# Statistics
MeanFlow, update!, uu!, uu
```
