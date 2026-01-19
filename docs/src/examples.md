# Examples

```@raw html
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
    <h3 style="margin-top: 0; color: white;">Ready-to-Run Examples</h3>
    <p style="margin-bottom: 0;">BioFlows includes comprehensive examples in the <code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 4px;">examples/</code> directory. From simple cylinders to swimming fish schools with FSI coupling.</p>
</div>
```

```@raw html
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
    <strong>2D Cylinder</strong><br>
    <small style="color: #666;">Classic benchmark</small>
</div>
<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
    <strong>Moving Bodies</strong><br>
    <small style="color: #666;">Oscillating & rotating</small>
</div>
<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
    <strong>Swimming Fish</strong><br>
    <small style="color: #666;">Flexible bodies</small>
</div>
<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
    <strong>3D Simulations</strong><br>
    <small style="color: #666;">Sphere & torus</small>
</div>
</div>
```

---

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

## Swimming Fish (Flexible Body)

**File:** `examples/swimming_fish.jl`

Flexible swimming fish with traveling wave motion:

```julia
using BioFlows
include("examples/swimming_fish.jl")

# Single carangiform swimmer (tail-dominated, like tuna)
sim, history = run_swimming_fish(
    steps = 500,
    nx = 256, nz = 128,
    fish_length = 0.2,
    amplitude = 0.1,           # Tail amplitude
    frequency = 1.0,           # Hz
    amplitude_envelope = :carangiform
)

# Get force statistics
stats = summarize_force_history(history; discard=0.2)
println("Mean Cd: ", stats.drag_mean)
println("Mean Cl: ", stats.lift_mean)
```

### Swimming Modes

```julia
# Anguilliform (eel-like, whole-body motion)
sim = swimming_fish_sim(
    amplitude_envelope = :anguilliform,
    head_amplitude = 0.05,
    wavelength = 0.8
)

# Subcarangiform (trout-like)
sim = swimming_fish_sim(
    amplitude_envelope = :subcarangiform
)
```

### Leading Edge Motion

```julia
# Fish with heave motion at head
sim = swimming_fish_sim(
    heave_amplitude = 0.05,
    heave_phase = 0.0
)

# Fish with pitch motion at head
sim = swimming_fish_sim(
    pitch_amplitude = 0.15,   # radians
    pitch_phase = π/2         # 90° phase lead
)

# Combined heave + pitch
sim = swimming_fish_sim(
    heave_amplitude = 0.03,
    pitch_amplitude = 0.1
)
```

### Command Line

```bash
julia --project examples/swimming_fish.jl
```

## Fish School

**File:** `examples/swimming_fish.jl`

Multiple swimming fish with configurable formations:

```julia
using BioFlows
include("examples/swimming_fish.jl")

# Staggered formation with 3 fish
sim, history, configs = run_fish_school(
    steps = 500,
    n_fish = 3,
    formation = :staggered,
    phase_offset = π/3,        # Phase difference between fish
    spacing = 0.15             # Lateral spacing
)
```

### Formations

```julia
# Inline (tandem)
sim, _ = fish_school_sim(n_fish=4, formation=:inline)

# Side by side
sim, _ = fish_school_sim(n_fish=3, formation=:side_by_side)

# Diamond (4+ fish)
sim, _ = fish_school_sim(n_fish=4, formation=:diamond)
```

### Synchronized vs Wave Motion

```julia
# Synchronized (all same phase)
sim, _ = synchronized_school_sim(n_fish=3)

# Wave-like phase progression
sim, _ = wave_school_sim(n_fish=5, phase_offset=π/4)
```

## Swimming Fish with AMR

**File:** `examples/swimming_fish.jl`

Adaptive mesh refinement for swimming fish:

```julia
using BioFlows
include("examples/swimming_fish.jl")

# Single fish with AMR
sim = swimming_fish_amr_sim(
    nx = 256, nz = 128,
    amplitude_envelope = :carangiform,
    amr_max_level = 2
)

for step in 1:500
    sim_step!(sim; remeasure=true)
end

amr_info(sim)
```

### Fish School with AMR

```julia
# Multiple fish with AMR
sim, configs = fish_school_amr_sim(
    n_fish = 3,
    formation = :staggered,
    amr_max_level = 2
)

for step in 1:500
    sim_step!(sim; remeasure=true)
end
```

## Fluid-Structure Interaction (FSI)

### Flexible Beam (Euler-Bernoulli)

**File:** `test_fsi.jl`

The Euler-Bernoulli beam solver uses Hermite finite elements for accurate structural dynamics:

```julia
using BioFlows

# Material properties
material = BeamMaterial(ρ=1100.0, E=1e6, ν_poisson=0.45)

# Geometry: 0.2m beam with 51 nodes
geometry = BeamGeometry(0.2, 51; thickness=0.01, width=0.05)

# Create beam with clamped-free boundary conditions
beam = EulerBernoulliBeam(geometry, material;
                          bc_left=CLAMPED,
                          bc_right=FREE,
                          damping=0.1)

# Apply uniform load
fill!(beam.q, 10.0)  # 10 N/m

# Time step the beam
for i in 1:100
    step!(beam, 1e-4)
end

# Check results
println("Max displacement: ", maximum(abs.(beam.w)) * 1000, " mm")
println("Kinetic energy: ", kinetic_energy(beam), " J")
println("Potential energy: ", potential_energy(beam), " J")
```

### Active Swimming with Traveling Wave

```julia
using BioFlows

# Create fish-like beam
L = 0.2  # Fish length
material = BeamMaterial(ρ=1050.0, E=5e5)
h_func = fish_thickness_profile(L, 0.02)  # Max thickness 0.02m
geometry = BeamGeometry(L, 51; thickness=h_func, width=0.02)

beam = EulerBernoulliBeam(geometry, material;
                          bc_left=CLAMPED, bc_right=FREE,
                          damping=0.5)

# Create traveling wave muscle activation (carangiform)
f_wave = traveling_wave_forcing(
    amplitude=100.0,
    frequency=2.0,
    wavelength=1.0,
    envelope=:carangiform,
    L=L
)

# Simulate swimming
dt = 1e-4
for i in 1:5000
    t = i * dt
    set_active_forcing!(beam, f_wave, t)
    step!(beam, dt)
end

# Get deformation
κ = get_curvature(beam)
M = get_bending_moment(beam)
println("Max curvature: ", maximum(abs.(κ)), " 1/m")
println("Max moment: ", maximum(abs.(M)), " N·m")
```

### Beam Properties

Access displacement and rotation fields:

```julia
# Displacement (w) and rotation (θ) at each node
w = beam.w        # Returns view into state vector
θ = beam.θ        # Rotation = ∂w/∂x
w_dot = beam.w_dot  # Velocity

# Set initial conditions (views allow in-place modification)
for i in 1:beam.geometry.n
    s = beam.s[i]
    beam.w[i] = 0.001 * (s/L)^2  # Parabolic shape
    beam.θ[i] = 0.002 * s/L      # Corresponding rotation
end

# Reset to zero
reset!(beam)
```

### Saving Beam State to Files

Save flexible body positions to JLD2 files with configurable save rates:

```julia
using BioFlows

# Create beam
material = BeamMaterial(ρ=1100.0, E=1e6)
geometry = BeamGeometry(0.2, 51; thickness=0.01, width=0.05)
beam = EulerBernoulliBeam(geometry, material; bc_left=CLAMPED, bc_right=FREE)

# Create writer - saves every 0.01 time units
writer = BeamStateWriter("flag_1.jld2"; interval=0.01)

# Simulation loop
dt = 1e-4
for step in 1:10000
    t = step * dt
    fill!(beam.q, 50.0)  # Apply load
    step!(beam, dt)
    file_save!(writer, beam, t)
end

# IMPORTANT: Close writer to save data to file
close!(writer, beam)
```

### Multiple Flags (Separate Files)

For multiple flexible bodies, each gets its own file:

```julia
# Create 5 flags with different properties
n_flags = 5
beams = [
    EulerBernoulliBeam(
        BeamGeometry(0.2, 51; thickness=0.01, width=0.05),
        BeamMaterial(ρ=1100.0, E=1e6 * i);  # Different stiffness
        bc_left=CLAMPED, bc_right=FREE
    ) for i in 1:n_flags
]

# Create writer group - creates flag_1.jld2, flag_2.jld2, etc.
writers = BeamStateWriterGroup("flag", n_flags; interval=0.01)

# Simulation loop
for step in 1:10000
    t = step * dt
    for beam in beams
        step!(beam, dt)
    end
    file_save!(writers, beams, t)  # Save all beams
end

# Close all writers
close!(writers, beams)
```

### Reading Beam State Files

```julia
using JLD2

jldopen("flag_1.jld2", "r") do file
    # Metadata
    n_snapshots = file["metadata/n_snapshots"]
    n_nodes = file["metadata/n_nodes"]
    L = file["metadata/length"]

    # Time series
    t = file["time"]
    KE = file["kinetic_energy"]
    PE = file["potential_energy"]

    # Field matrices (n_nodes × n_snapshots)
    w = file["fields/displacement"]
    θ = file["fields/rotation"]
    κ = file["fields/curvature"]

    # Individual snapshots
    w_10 = file["snapshots/10/displacement"]
    θ_10 = file["snapshots/10/rotation"]
end
```

### Static Analysis (Point Loads)

For static problems, solve K*u = F directly:

```julia
using LinearAlgebra

# Create beam
beam = EulerBernoulliBeam(geometry, material;
                          bc_left=CLAMPED, bc_right=FREE)

# Apply point load at tip
n_dof = 2 * beam.geometry.n
F = zeros(n_dof)
tip_dof = n_dof - 1  # Tip displacement DOF
F[tip_dof] = 5.0     # 5 N point load

# Static solve
K = Matrix(beam.K_mat)
u = K \ F

# Tip deflection
w_tip = u[tip_dof]
println("Tip deflection: ", w_tip * 1000, " mm")

# Analytical: w = PL³/(3EI)
E, I, L = material.E, beam.I_vec[1], beam.geometry.L
w_analytical = 5.0 * L^3 / (3 * E * I)
println("Analytical: ", w_analytical * 1000, " mm")
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
