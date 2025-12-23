# Diagnostics

BioFlows provides built-in diagnostics for forces, vorticity, and flow statistics.

## Force Computation

### Force Components

```@docs
force_components
force_coefficients
record_force!
```

### Usage

```julia
using BioFlows

# Get all force components
components = force_components(sim; ρ=1.0, reference_area=sim.L)

# Access individual forces
pressure_force = components.pressure   # [Fx, Fz] or [Fx, Fy, Fz]
viscous_force = components.viscous
total_force = components.total

# Dimensionless coefficients
Cd, Cl = components.coefficients[3]  # Total force coefficients
```

### Force History

Record forces over time for analysis:

```julia
history = NamedTuple[]

for step in 1:1000
    sim_step!(sim)
    record_force!(history, sim; ρ=1.0, reference_area=sim.L)
end

# Access recorded data
times = [h.time for h in history]
drag_coeffs = [h.total_coeff[1] for h in history]
lift_coeffs = [h.total_coeff[2] for h in history]
```

### Force Statistics

```julia
# Compute mean and RMS after discarding transient
stats = summarize_force_history(history; discard=0.2)  # Discard first 20%

println("Mean drag: ", stats.drag_mean)
println("Drag std:  ", stats.drag_std)
println("Mean lift: ", stats.lift_mean)
println("Lift std:  ", stats.lift_std)
```

### Automatic Force File Output

Use `ForceWriter` to automatically save lift and drag coefficients to a JLD2 file
at specified time intervals:

```julia
using BioFlows

# Create simulation
sim = Simulation((128, 128), (1.0, 0.0), 1.0; ν=0.001, body=AutoBody(sdf))

# Create force writer - saves every 0.1 time units
force_writer = ForceWriter("forces.jld2"; interval=0.1, reference_area=sim.L)

# Time stepping loop
for _ in 1:5000
    sim_step!(sim)
    maybe_save!(force_writer, sim)  # Writes to JLD2 when interval elapsed
end
```

The JLD2 file contains arrays: `time`, `Cd`, `Cl`, `Cd_pressure`, `Cd_viscous`,
`Cl_pressure`, `Cl_viscous`.

```julia
# Read the output file
using JLD2
data = load("forces.jld2")
time = data["time"]
Cd = data["Cd"]
Cl = data["Cl"]
```

## Vorticity

### Vorticity Fields

```@docs
vorticity_component
vorticity_magnitude
```

### Usage

```julia
# Out-of-plane vorticity for 2D (ω₃ = ∂v/∂x - ∂u/∂z)
ω3 = vorticity_component(sim, 3)

# Vorticity magnitude
ω_mag = vorticity_magnitude(sim)

# For 3D simulations
ωx = vorticity_component(sim, 1)
ωy = vorticity_component(sim, 2)
ωz = vorticity_component(sim, 3)
```

### Ghost Layer Handling

By default, ghost layers are stripped from output:

```julia
# With ghost layers stripped (default)
ω = vorticity_component(sim, 3; strip_ghosts=true)

# Keep ghost layers
ω_with_ghosts = vorticity_component(sim, 3; strip_ghosts=false)
```

## Cell-Centered Fields

For visualization and output, interpolate to cell centers:

```@docs
cell_center_velocity
cell_center_vorticity
cell_center_pressure
```

### Usage

```julia
# Cell-centered velocity [nx, nz, 2] or [nx, ny, nz, 3]
vel = cell_center_velocity(sim)

# Cell-centered vorticity
# 2D: scalar field [nx, nz]
# 3D: vector field [nx, ny, nz, 3]
vort = cell_center_vorticity(sim)

# Cell-centered pressure [nx, nz] or [nx, ny, nz]
pres = cell_center_pressure(sim)
```

## Simulation Diagnostics

The `compute_diagnostics` function returns summary statistics for the current simulation state.

### Usage

```julia
diag = compute_diagnostics(sim)

println("Max u-velocity: ", diag.max_u)
println("Max w-velocity: ", diag.max_w)
println("CFL number:     ", diag.CFL)
println("Time step:      ", diag.Δt)
println("Grid size:      ", diag.grid)
```

## Example: Complete Diagnostics Loop

```julia
using BioFlows
using Statistics

# Setup simulation
sim = Simulation((128, 64), (1, 0), 16.0;
                 ν = 16.0 / 100,
                 body = AutoBody(sdf))

# Storage
force_history = NamedTuple[]
max_cfl = 0.0

# Main loop
for step in 1:5000
    sim_step!(sim; remeasure=false)
    record_force!(force_history, sim)

    # Track CFL
    diag = compute_diagnostics(sim)
    max_cfl = max(max_cfl, diag.CFL)

    # Print every 500 steps
    if step % 500 == 0
        stats = summarize_force_history(force_history; discard=0.3)
        println("Step $step: t=$(round(sim_time(sim), digits=2)), ",
                "Cd=$(round(stats.drag_mean, digits=3)), ",
                "CFL=$(round(diag.CFL, digits=3))")
    end
end

# Final statistics
final_stats = summarize_force_history(force_history; discard=0.3)
println("\nFinal Results:")
println("  Mean Cd = ", round(final_stats.drag_mean, digits=4))
println("  Std Cd  = ", round(final_stats.drag_std, digits=4))
println("  Mean Cl = ", round(final_stats.lift_mean, digits=4))
println("  Max CFL = ", round(max_cfl, digits=3))
```

## Visualization

BioFlows integrates with Plots.jl via extensions:

```julia
using Plots
using BioFlows

# Get vorticity field
ω = vorticity_component(sim, 3)

# Plot with flood
heatmap(ω', c=:RdBu, clim=(-2, 2),
        xlabel="x", ylabel="z",
        title="Vorticity field")
```

For VTK output (ParaView compatible):

```julia
using WriteVTK
using BioFlows

# VTK writer becomes available
vtkWriter(sim, "output_dir")
```
