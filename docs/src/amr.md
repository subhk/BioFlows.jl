# Adaptive Mesh Refinement

BioFlows includes an Adaptive Mesh Refinement (AMR) system that automatically
refines the computational grid near immersed bodies and regions of high flow
gradients.

## AMRSimulation

The `AMRSimulation` type wraps a standard `Simulation` and adds AMR capability.

```@docs
AMRSimulation
AMRConfig
```

## Basic Usage

```julia
using BioFlows

# Define geometry
radius = 8
center = 64
sdf(x, t) = sqrt((x[1] - center)^2 + (x[2] - center)^2) - radius

# Configure AMR
config = AMRConfig(
    max_level = 2,                     # Max refinement (2x, 4x resolution)
    body_distance_threshold = 3.0,     # Refine within 3 cells of body
    velocity_gradient_threshold = 1.0,
    vorticity_threshold = 1.0,
    regrid_interval = 10,              # Check regridding every 10 steps
    buffer_size = 1                    # Buffer cells around refined regions
)

# Create AMR simulation
sim = AMRSimulation((128, 128), (128.0, 128.0);
                    inletBC = (1.0, 0.0),
                    ν = 2radius / 200,
                    body = AutoBody(sdf),
                    L_char = 2radius,
                    amr_config = config)

# Time integration (regridding happens automatically)
for step in 1:1000
    sim_step!(sim; remeasure=true)
end

# Check refinement
println("Refined cells: ", num_refined_cells(sim.refined_grid))
```

## AMRConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_level` | 2 | Maximum refinement level (1=2x, 2=4x, etc.) |
| `body_distance_threshold` | 3.0 | Refine within this distance from body (cells) |
| `velocity_gradient_threshold` | 1.0 | Threshold for velocity gradient indicator |
| `vorticity_threshold` | 1.0 | Threshold for vorticity-based refinement |
| `regrid_interval` | 10 | Steps between regridding checks |
| `buffer_size` | 1 | Buffer cells around refined regions |
| `body_weight` | 0.5 | Weight for body proximity in combined indicator |
| `gradient_weight` | 0.3 | Weight for velocity gradient |
| `vorticity_weight` | 0.2 | Weight for vorticity |

## Refinement Indicators

The AMR system uses multiple indicators to decide where to refine:

### Body Distance Indicator

Refines cells near the immersed body surface:

```julia
indicator = compute_body_refinement_indicator(flow, body;
    threshold=config.body_distance_threshold, t=t)
```

### Velocity Gradient Indicator

Refines regions of high velocity gradients:

```julia
indicator = compute_velocity_gradient_indicator(flow;
    threshold=config.velocity_gradient_threshold)
```

### Vorticity Indicator

Refines regions of high vorticity:

```julia
indicator = compute_vorticity_indicator(flow;
    threshold=config.vorticity_threshold)
```

### Combined Indicator

Combines all indicators with configurable weights:

```julia
indicator = compute_combined_indicator(flow, body;
    body_threshold=3.0,
    gradient_threshold=1.0,
    vorticity_threshold=1.0,
    t=0.0,
    body_weight=0.5,
    gradient_weight=0.3,
    vorticity_weight=0.2
)
```

## AMR Control

### Enable/Disable AMR

```julia
# Disable regridding (keep current mesh)
set_amr_active!(sim, false)

# Re-enable
set_amr_active!(sim, true)
```

### Force Regridding

```julia
amr_regrid!(sim)
```

### Query Refinement

```julia
# Number of refined cells
n = num_refined_cells(sim.refined_grid)

# Refinement level at specific cell (0 = base, 1+ = refined)
level = refinement_level(sim.refined_grid, i, j)

# Get current indicator field (for visualization)
indicator = get_refinement_indicator(sim)
```

## Grid Types

### StaggeredGrid

The base grid type using MAC (Marker-And-Cell) layout:

```@docs
StaggeredGrid
```

- Velocities at face centers
- Pressure at cell centers
- Supports 2D (XZ plane) and 3D

### RefinedGrid

Container for AMR data:

```@docs
RefinedGrid
```

Tracks:
- Base (coarse) grid
- Refined cell locations and levels
- Local refined sub-grids
- Interpolation weights

## Performance Considerations

1. **Regrid Interval**: Larger intervals reduce overhead but may miss features
2. **Buffer Size**: Prevents refinement boundaries from affecting solution
3. **Max Level**: Higher levels increase accuracy but add cost
4. **Indicator Weights**: Tune based on flow physics

## Example: Wake Refinement

```julia
# Emphasize wake region (high vorticity) over body proximity
config = AMRConfig(
    max_level = 3,
    body_weight = 0.2,        # Less emphasis on body
    gradient_weight = 0.3,
    vorticity_weight = 0.5,   # More emphasis on wake
    regrid_interval = 20
)
```

## Composite Solver (Advanced)

For advanced users, BioFlows exposes the internal composite solver types used for
AMR pressure projection.

### CompositePoisson

The composite Poisson solver combines a base multigrid solver with refined patches:

```julia
# CompositePoisson manages:
# - Base grid: MultiLevelPoisson for coarse solution
# - Patches: PatchPoisson solvers for refined regions
# - Velocity: RefinedVelocityField for patch velocities
```

### Patch Types

| Type | Description |
|------|-------------|
| `PatchPoisson` | Local Poisson solver for a refined patch |
| `RefinedVelocityPatch` | Velocity storage at refined resolution |
| `RefinedVelocityField` | Collection of velocity patches |

### Patch Operations

```julia
# Add/remove patches
add_patch!(field, anchor, patch)
remove_patch!(field, anchor)
get_patch(field, anchor)
clear_patches!(field)

# Query patches
has_patches(cp)
num_patches(cp)
```

### AMR Projection

The `amr_project!` function performs divergence-free projection on all levels:

```julia
# Full AMR projection workflow:
# 1. Set divergence on base grid
# 2. Set divergence on refined patches
# 3. Interpolate velocity to patches
# 4. Solve composite Poisson system
# 5. Correct velocities at all levels
# 6. Enforce interface consistency

amr_project!(flow, cp)
```

### Utility Functions

| Function | Description |
|----------|-------------|
| `amr_cfl(flow, cp)` | CFL considering refined patches |
| `check_amr_divergence(flow, cp)` | Divergence at all levels |
| `synchronize_base_and_patches!(flow, cp)` | Sync after regridding |
