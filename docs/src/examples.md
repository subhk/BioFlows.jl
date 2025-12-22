# Examples

BioFlows includes several example scripts in the `examples/` directory.

## Flow Past Cylinder (2D)

**File:** `examples/flow_past_cylinder_2d.jl`

The most comprehensive example with configurable parameters:

```julia
using BioFlows

include("examples/flow_past_cylinder_2d.jl")

# Run with custom parameters
sim, history, stats, writer, diagnostics = run_flow_past_cylinder(
    nx = 256,           # Grid points in x
    nz = 64,            # Grid points in z
    Lx = 8.0,           # Domain length
    Lz = 2.0,           # Domain height
    Re = 150,           # Reynolds number
    final_time = 10.0,  # Convective time units
    save_center_fields = true,
    center_filename = "output.jld2",
    diagnostic_interval = 100
)

println("Mean Cd: ", stats.drag_mean)
println("Lift RMS: ", stats.lift_rms)
```

### Command Line

```bash
julia --project examples/flow_past_cylinder_2d.jl
```

### Output

- Force history with Cd/Cl coefficients
- Cell-centered velocity/vorticity snapshots (JLD2)
- Diagnostic statistics

## Circle Benchmark

**File:** `examples/circle_benchmark.jl`

Simple 2D cylinder for benchmarking:

```julia
using BioFlows

radius = 8
center = 4 * radius - 1
sdf(x, t) = sqrt((x[1] - center)^2 + (x[2] - center)^2) - radius

sim = Simulation((6radius, 8radius), (1, 0), 2radius;
                 ν = 2radius / 120,
                 body = AutoBody(sdf))

sim_step!(sim, 1.0; remeasure=false)
```

### Command Line

```bash
julia --project examples/circle_benchmark.jl
```

## Oscillating Cylinder

**File:** `examples/oscillating_cylinder.jl`

Cylinder with sinusoidal cross-flow motion:

```julia
using BioFlows

radius = 8
A = radius / 2    # Oscillation amplitude
St = 0.2          # Strouhal number
ω = 2π * St       # Angular frequency

sdf(x, t) = sqrt(x[1]^2 + x[2]^2) - radius
map(x, t) = x .- [0, A * sin(ω * t)]

body = AutoBody(sdf, map)

sim = Simulation((128, 64), (1, 0), 2radius;
                 ν = 2radius / 100,
                 body = body)

# Must remeasure for moving bodies
sim_step!(sim, 5.0; remeasure=true)
```

### Command Line

```bash
julia --project examples/oscillating_cylinder.jl
```

## 3D Torus

**File:** `examples/torus_3d.jl`

3D torus in periodic inflow:

```julia
using BioFlows

R = 16  # Major radius
r = 4   # Minor radius

function sdf_torus(x, t)
    # Distance from torus centerline
    d_ring = sqrt(x[1]^2 + x[2]^2) - R
    sqrt(d_ring^2 + x[3]^2) - r
end

sim = Simulation((64, 64, 64), (1, 0, 0), 2r;
                 ν = 2r / 50,
                 body = AutoBody(sdf_torus),
                 perdir = (1, 2))  # Periodic in x and y

sim_step!(sim, 2.0; remeasure=false)
```

### Command Line

```bash
julia --project examples/torus_3d.jl
```

## 3D Sphere

**File:** `examples/sphere_3d.jl`

3D sphere wake simulation:

```julia
using BioFlows

radius = 8
center = [32, 32, 32]

sdf(x, t) = sqrt(sum((x .- center).^2)) - radius

sim = Simulation((128, 64, 64), (1, 0, 0), 2radius;
                 ν = 2radius / 100,
                 body = AutoBody(sdf))

sim_step!(sim, 1.0; remeasure=false)
```

### Command Line

```bash
julia --project examples/sphere_3d.jl
```

## Visualization Examples

### Plot Vorticity

```julia
using BioFlows
using Plots

# Setup simulation
radius = 8
sdf(x, t) = sqrt((x[1] - 32)^2 + (x[2] - 32)^2) - radius
sim = Simulation((128, 64), (1, 0), 2radius;
                 ν = 2radius / 100,
                 body = AutoBody(sdf))

# Run simulation
sim_step!(sim, 5.0; remeasure=false)

# Get vorticity
ω = vorticity_component(sim, 3)

# Create plot
heatmap(ω', c=:RdBu, clim=(-2, 2),
        aspect_ratio=:equal,
        xlabel="x", ylabel="z",
        title="Vorticity ω_z")
savefig("vorticity.png")
```

### Animate Vorticity

```julia
using BioFlows
using Plots

# Setup
sdf(x, t) = sqrt((x[1] - 32)^2 + (x[2] - 32)^2) - 8
sim = Simulation((128, 64), (1, 0), 16.0;
                 ν = 16.0 / 100,
                 body = AutoBody(sdf))

anim = @animate for step in 1:500
    sim_step!(sim)

    if step % 10 == 0
        ω = vorticity_component(sim, 3)
        heatmap(ω', c=:RdBu, clim=(-2, 2),
                title="t = $(round(sim_time(sim), digits=2))")
    end
end

gif(anim, "vorticity.gif", fps=15)
```

## Loading Saved Data

Read JLD2 snapshots from `CenterFieldWriter`:

```julia
using JLD2

# Load snapshot
jldopen("center_fields.jld2", "r") do file
    # List snapshots
    for key in keys(file)
        println(key)
    end

    # Read specific snapshot
    t = file["snapshot_10/time"]
    vel = file["snapshot_10/velocity"]
    vort = file["snapshot_10/vorticity"]
end
```

## GPU Execution

Run on NVIDIA GPU:

```julia
using CUDA
using BioFlows

sim = Simulation((256, 128), (1, 0), 16.0;
                 ν = 16.0 / 200,
                 body = AutoBody(sdf),
                 mem = CuArray)  # Use GPU arrays

sim_step!(sim, 10.0)
```
