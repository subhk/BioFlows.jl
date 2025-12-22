# API Reference

## Simulation Types

```@docs
Simulation
AMRSimulation
AMRConfig
```

## Flow and Body Types

```@docs
Flow
AutoBody
```

## Grid Types (AMR)

```@docs
StaggeredGrid
SolutionState
RefinedGrid
GridType
TwoDimensional
ThreeDimensional
```

## Simulation Control

```@docs
sim_step!
sim_time
measure!
perturb!
sim_info
```

## AMR Functions

```@docs
amr_regrid!
set_amr_active!
get_refinement_indicator
num_refined_cells
refinement_level
is_2d
is_3d
domain_size
cell_volume
```

## AMR Adapters

```@docs
FlowToGridAdapter
flow_to_staggered_grid
flow_to_solution_state
create_refined_grid
```

## Refinement Indicators

```@docs
compute_body_refinement_indicator
compute_velocity_gradient_indicator
compute_vorticity_indicator
compute_combined_indicator
mark_cells_for_refinement
apply_buffer_zone!
```

## Force Diagnostics

```@docs
pressure_force
viscous_force
total_force
force_components
force_coefficients
record_force!
```

## Vorticity Functions

```@docs
vorticity_component
vorticity_magnitude
curl
ω
ω_mag
```

## Cell-Centered Fields

```@docs
cell_center_velocity
cell_center_vorticity
compute_diagnostics
```

## Output

```@docs
CenterFieldWriter
maybe_save!
save!
load!
vtkWriter
default_attrib
```

## Pressure Solvers

```@docs
AbstractPoisson
Poisson
MultiLevelPoisson
solver!
mult!
```

## Flow Solver

```@docs
mom_step!
quick
cds
```

## Utilities

```@docs
L₂
BC!
@inside
inside
δ
apply!
loc
@log
set_backend
backend
```

## Body Functions

```@docs
AbstractBody
measure_sdf!
sdf
measure
```

## Statistics

```@docs
MeanFlow
update!
uu!
uu
```

## Index

```@index
```
