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

## Moving Bodies with AMR

BioFlows supports AMR for both rigid and flexible moving bodies. The mesh
automatically tracks body motion and regrids when necessary.

### Body Types

| Type | Description | Example |
|------|-------------|---------|
| **Rigid** | Shape unchanged, position/orientation varies | Oscillating cylinder, rotating ellipse |
| **Flexible** | Shape deforms over time | Swimming fish, flapping wing |

### Convenience Configurations

BioFlows provides pre-configured settings for common use cases:

#### FlexibleBodyAMRConfig

Optimized for deforming bodies like swimming fish:

```julia
config = FlexibleBodyAMRConfig(
    max_level = 2,                    # 4x refinement
    body_distance_threshold = 4.0,    # Larger region for moving bodies
    indicator_change_threshold = 0.05, # 5% change triggers regrid
    min_regrid_interval = 2           # Allow frequent regridding
)
```

**Default settings:**
- `flexible_body = true` - Enable motion-adaptive regridding
- `indicator_change_threshold = 0.05` - 5% cell change triggers regrid
- `min_regrid_interval = 2` - Allow regridding every 2 steps
- `regrid_interval = 5` - Check regridding at least every 5 steps
- `body_distance_threshold = 4.0` - Larger refinement region
- `body_weight = 0.6` - Higher weight for body proximity

#### RigidBodyAMRConfig

Optimized for moving rigid bodies:

```julia
config = RigidBodyAMRConfig(
    max_level = 2,
    body_distance_threshold = 3.0,
    indicator_change_threshold = 0.08,  # Less sensitive (8%)
    min_regrid_interval = 3             # Less frequent
)
```

**Default settings:**
- `flexible_body = true` - Enable motion-adaptive regridding
- `indicator_change_threshold = 0.08` - 8% cell change triggers regrid
- `min_regrid_interval = 3` - Allow regridding every 3 steps
- `regrid_interval = 8` - Check regridding every 8 steps
- `body_distance_threshold = 3.0` - Standard refinement region
- `body_weight = 0.5` - Standard weight for body proximity

### Motion Detection

The AMR system automatically detects body motion by comparing refinement
indicators between time steps:

```julia
# Get motion statistics
stats = get_body_motion_stats(sim)
println("Indicator stored: ", stats.indicator_stored)
println("Patches: ", stats.n_patches)
println("Steps since regrid: ", stats.steps_since_regrid)
```

When the indicator change exceeds `indicator_change_threshold`, regridding
is triggered (subject to `min_regrid_interval`).

### Helper Functions

```julia
# Force immediate regridding
force_regrid!(sim)

# Reset body tracking (after sudden position changes)
reset_body_tracking!(sim)

# Get AMR status
info = amr_info(sim)
println("Active: ", info.active)
println("Flexible body: ", info.flexible_body)
println("Refined cells: ", info.refined_cells)
println("Patches: ", info.num_patches)
```

## Swimming Fish with AMR

BioFlows includes comprehensive support for flexible swimming bodies.

### Single Fish

```julia
using BioFlows
include("examples/swimming_fish.jl")

# Create AMR-enabled swimming fish simulation
sim = swimming_fish_amr_sim(
    nx = 256, nz = 128,          # Grid dimensions
    fish_length = 0.2,           # Fish body length
    amplitude = 0.1,             # Tail amplitude (relative to length)
    frequency = 1.0,             # Oscillation frequency (Hz)
    amplitude_envelope = :carangiform,  # Swimming mode
    amr_max_level = 2            # Refinement level
)

# Run simulation - patches follow the fish automatically
for step in 1:1000
    sim_step!(sim; remeasure=true)

    if step % 100 == 0
        info = amr_info(sim; verbose=false)
        println("Step $step: patches=$(info.num_patches)")
    end
end
```

### Swimming Modes

BioFlows supports multiple swimming modes with different amplitude envelopes:

| Mode | Envelope | Description |
|------|----------|-------------|
| `:carangiform` | `A(s) = A_tail * (s/L)²` | Tail-dominated (tuna, mackerel) |
| `:anguilliform` | `A(s) = A_head + (A_tail - A_head) * s/L` | Whole-body (eel, lamprey) |
| `:subcarangiform` | `A(s) = A_tail * (s/L)^1.5` | Intermediate (trout, carp) |
| `:uniform` | `A(s) = A_uniform` | Constant amplitude |

```julia
# Anguilliform swimming (eel-like, whole-body motion)
sim = swimming_fish_amr_sim(
    amplitude_envelope = :anguilliform,
    head_amplitude = 0.05,    # Non-zero head motion
    amplitude = 0.12,         # Tail amplitude
    wavelength = 0.8          # Shorter wavelength
)
```

### Leading Edge Motion

Fish can have additional heave (vertical oscillation) and pitch (angular
oscillation) at the leading edge:

```julia
# Fish with combined heave + pitch motion
sim = swimming_fish_amr_sim(
    heave_amplitude = 0.05,   # Vertical oscillation at head
    heave_phase = 0.0,        # In phase with body wave
    pitch_amplitude = 0.15,   # Angular oscillation (radians)
    pitch_phase = π/2         # 90° phase lead for optimal thrust
)
```

The body centerline follows:

```
y(x,t) = y_head(t) + A(x) * sin(k*x - ω*t + φ) + pitch_contribution(x,t)
```

where:
- `y_head(t) = heave_amplitude * sin(ω*t + heave_phase)` - leading edge heave
- `A(x)` = amplitude envelope (varies based on swimming mode)
- `k = 2π/λ` = wave number
- `ω = 2π*f` = angular frequency

## Fish School with AMR

Simulate multiple swimming fish with individual phase offsets:

```julia
using BioFlows
include("examples/swimming_fish.jl")

# Create fish school with AMR
sim, fish_configs = fish_school_amr_sim(
    nx = 512, nz = 256,
    Lx = 2.0, Lz = 1.0,
    n_fish = 3,                    # Number of fish
    formation = :staggered,        # Formation type
    phase_offset = π/3,            # Phase difference between fish
    fish_length = 0.15,
    amplitude = 0.1,
    amr_max_level = 2
)

# Run simulation
for step in 1:500
    sim_step!(sim; remeasure=true)
end

# Check AMR status
amr_info(sim)
```

### School Formations

| Formation | Description |
|-----------|-------------|
| `:staggered` | Diagonal arrangement (default) |
| `:inline` | Tandem (one behind another) |
| `:side_by_side` | Lateral arrangement |
| `:diamond` | Diamond pattern (4+ fish) |
| `:custom` | User-defined positions |

```julia
# Diamond formation with 4 fish
sim, configs = fish_school_amr_sim(
    n_fish = 4,
    formation = :diamond,
    spacing = 0.15          # Lateral spacing
)
```

### Custom Fish Positions

```julia
# Define custom positions
custom = [
    FishConfig(0.2, 0.4, 0.0),      # x_pos, z_pos, phase
    FishConfig(0.2, 0.6, π/4),
    FishConfig(0.4, 0.5, π/2),
]

sim, _ = fish_school_amr_sim(
    custom_positions = custom,
    formation = :custom
)
```

### Phase Synchronization

Control phase relationships between fish:

```julia
# Synchronized school (all same phase)
sim, _ = fish_school_amr_sim(
    n_fish = 3,
    phase_offset = 0.0,        # No phase difference
    formation = :side_by_side
)

# Wave-like phase progression
sim, _ = fish_school_amr_sim(
    n_fish = 5,
    phase_offset = π/4,        # Progressive phase delay
    formation = :inline
)
```

## Oscillating Cylinder with AMR

For rigid body motion like oscillating cylinders:

```julia
using BioFlows
include("examples/oscillating_cylinder.jl")

# Create oscillating cylinder with AMR
sim = oscillating_cylinder_amr_sim(
    n = 128, m = 64,
    St = 0.2,              # Strouhal number
    amplitude = 0.3,       # Oscillation amplitude (relative to diameter)
    max_level = 2
)

# Run simulation
for step in 1:500
    sim_step!(sim; remeasure=true)
end

amr_info(sim)
```

### Other Rigid Body Examples

```julia
# Rotating cylinder
sim = rotating_cylinder_amr_sim(
    n = 64, m = 64,
    ω = 0.5,               # Angular velocity (rad/time)
    max_level = 2
)

# Orbiting cylinder (large motion)
sim = orbiting_cylinder_amr_sim(
    n = 96, m = 96,
    orbit_radius = 12,     # Orbit radius (grid cells)
    orbit_period = 80,     # Orbit period (time units)
    max_level = 2
)
```

## Best Practices for Moving Bodies

### 1. Always Use `remeasure=true`

```julia
# Moving bodies require remeasuring the SDF each step
sim_step!(sim; remeasure=true)
```

### 2. Choose Appropriate Configuration

```julia
# Flexible body (swimming fish, flapping wings)
config = FlexibleBodyAMRConfig(max_level=2)

# Rigid body (oscillating cylinder, rotating ellipse)
config = RigidBodyAMRConfig(max_level=2)
```

### 3. Tune Indicator Threshold

For fast-moving bodies, use lower thresholds:

```julia
config = FlexibleBodyAMRConfig(
    indicator_change_threshold = 0.03,  # More sensitive
    min_regrid_interval = 1             # Allow every step
)
```

For slower motion, higher thresholds reduce overhead:

```julia
config = RigidBodyAMRConfig(
    indicator_change_threshold = 0.15,  # Less sensitive
    min_regrid_interval = 5             # Less frequent
)
```

### 4. Monitor Regridding

```julia
for step in 1:1000
    sim_step!(sim; remeasure=true)

    # Check regridding frequency
    stats = get_body_motion_stats(sim)
    if stats.steps_since_regrid == 0
        println("Regrid at step $step")
    end
end
```

### 5. Force Coefficient Computation

```julia
# Compute forces on moving body
forces = total_force(sim)  # Returns (Fx, Fz) or (Fx, Fy, Fz)

# Normalize to coefficients
Cd = forces[1] / (0.5 * sim.L * sim.U^2)
Cl = forces[2] / (0.5 * sim.L * sim.U^2)
```

## Troubleshooting

### NaN Values

If velocity becomes NaN:
1. Reduce time step (decrease CFL number)
2. Increase `min_regrid_interval` if regridding too frequently
3. Check body SDF is smooth and continuous

### Excessive Regridding

If regridding every step:
1. Increase `indicator_change_threshold`
2. Increase `min_regrid_interval`
3. Use `RigidBodyAMRConfig` for rigid bodies

### Missing Refinement

If patches don't follow body:
1. Increase `body_distance_threshold`
2. Decrease `indicator_change_threshold`
3. Ensure `remeasure=true` is set

### Memory Issues

For large simulations:
1. Reduce `max_level`
2. Increase `regrid_interval`
3. Use smaller `buffer_size`
