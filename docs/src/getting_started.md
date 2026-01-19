# Getting Started

This guide will walk you through installing BioFlows.jl and running your first simulation.

---

## Installation

### Option 1: From GitHub (Recommended)

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/BioFlows.jl")
```

### Option 2: Local Development

```bash
git clone https://github.com/subhk/BioFlows.jl.git
cd BioFlows.jl
```

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Optional: GPU Support

For NVIDIA GPU acceleration (10-50× speedup):

```julia
using Pkg
Pkg.add("CUDA")
```

See the [GPU Guide](@ref gpu) for detailed setup instructions.

---

## Your First Simulation

Let's simulate flow past a 2D cylinder - a classic CFD benchmark.

### Step 1: Import BioFlows

```julia
using BioFlows
```

### Step 2: Define Parameters

```julia
# Physical parameters
Re = 100        # Reynolds number
U = 1.0         # Inlet velocity (m/s)
D = 16.0        # Cylinder diameter (length scale)

# Grid dimensions
nx, nz = 128, 64            # Grid points
Lx, Lz = Float32(nx), Float32(nz)  # Domain size (using Δx = 1)
```

### Step 3: Define Geometry

BioFlows uses **signed distance functions (SDF)** to define geometry:

```julia
# Cylinder center position
center_x = nx / 4   # Place at 1/4 of domain
center_z = nz / 2   # Center vertically
radius = D / 2

# SDF: negative inside, positive outside, zero on surface
sdf(x, t) = sqrt((x[1] - center_x)^2 + (x[2] - center_z)^2) - radius
```

```@raw html
<div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196f3; margin: 1rem 0;">
<strong>What is an SDF?</strong><br>
A signed distance function returns the distance from point <code>x</code> to the nearest surface:
<ul style="margin-bottom: 0;">
<li><code>sdf(x, t) &lt; 0</code> → Inside the body</li>
<li><code>sdf(x, t) = 0</code> → On the surface</li>
<li><code>sdf(x, t) &gt; 0</code> → Outside the body</li>
</ul>
</div>
```

### Step 4: Create Simulation

```julia
sim = Simulation((nx, nz), (Lx, Lz);
                 inletBC = (U, 0f0),       # Inlet velocity (u, w)
                 ν = U * D / Re,           # Kinematic viscosity
                 body = AutoBody(sdf),     # Immersed body
                 L_char = D)               # Length scale for coefficients
```

### Step 5: Run Simulation

```julia
# Run until t* = 10 (convective time units)
final_time = 10.0

while sim_time(sim) < final_time
    sim_step!(sim; remeasure=false)

    # Print progress every 100 steps
    if length(sim.flow.Δt) % 100 == 0
        println("t* = ", round(sim_time(sim), digits=2))
    end
end

println("Simulation complete!")
```

### Complete Code

```julia
using BioFlows

# Parameters
Re, U, D = 100, 1f0, 16f0
nx, nz = 128, 64
Lx, Lz = Float32(nx), Float32(nz)

# Geometry
center_x, center_z = nx/4, nz/2
radius = D/2
sdf(x, t) = sqrt((x[1] - center_x)^2 + (x[2] - center_z)^2) - radius

# Create and run
sim = Simulation((nx, nz), (Lx, Lz);
                 inletBC = (U, 0f0),
                 ν = U * D / Re,
                 body = AutoBody(sdf),
                 L_char = D)

sim_step!(sim, 10.0)  # Run to t* = 10

println("Final time: t* = ", sim_time(sim))
```

---

## Understanding the Output

### Dimensionless Time

BioFlows reports **dimensionless (convective) time**:

```math
t^* = \frac{t \cdot U}{L}
```

- `sim_time(sim)` → Dimensionless time `t*`
- `time(sim.flow)` → Raw simulation time in seconds

### Checking Simulation State

```julia
# Print simulation info
sim_info(sim)

# Get diagnostics
diag = compute_diagnostics(sim)
println("Max velocity: ", diag.max_u)
println("CFL number: ", diag.CFL)
println("Time step: ", diag.Δt)
```

---

## Visualizing Results

### With Plots.jl

```julia
using Plots

# Get vorticity field
ω = vorticity_component(sim, 3)  # Component 3 = out-of-plane (ω_z)

# Create heatmap
heatmap(ω',
        c = :RdBu,           # Red-blue colormap
        clim = (-2, 2),      # Color limits
        aspect_ratio = :equal,
        xlabel = "x", ylabel = "z",
        title = "Vorticity ωz at t* = $(round(sim_time(sim), digits=2))")
```

### With Makie.jl

```julia
using CairoMakie

ω = vorticity_component(sim, 3)

fig = Figure(size = (800, 400))
ax = Axis(fig[1,1], xlabel="x", ylabel="z", title="Vorticity")
hm = heatmap!(ax, ω', colormap=:RdBu, colorrange=(-2, 2))
Colorbar(fig[1,2], hm, label="ωz")
fig
```

---

## Key Concepts

### Signed Distance Functions (SDF)

Geometry is defined implicitly - no mesh generation required!

```julia
# Circle/Cylinder
sdf_circle(x, t) = sqrt(x[1]^2 + x[2]^2) - radius

# Rectangle
sdf_rect(x, t) = max(abs(x[1]) - width/2, abs(x[2]) - height/2)

# Sphere (3D)
sdf_sphere(x, t) = sqrt(x[1]^2 + x[2]^2 + x[3]^2) - radius

# Ellipse
sdf_ellipse(x, t) = sqrt((x[1]/a)^2 + (x[2]/b)^2) - 1
```

### Moving Bodies

Use a `map` function for time-dependent motion:

```julia
# Oscillating cylinder (vertical)
sdf(x, t) = sqrt(x[1]^2 + x[2]^2) - radius
map(x, t) = x .- [0, A * sin(ω * t)]

body = AutoBody(sdf, map)
sim = Simulation(...; body = body)

# Important: use remeasure=true for moving bodies!
sim_step!(sim, 10.0; remeasure=true)
```

### Boundary Conditions

```julia
# Constant inlet velocity
sim = Simulation(...; inletBC = (1.0, 0.0))

# Parabolic profile (Poiseuille)
H = Lz / 2
inletBC(i, x, t) = i == 1 ? 1.5 * (1 - ((x[2] - H) / H)^2) : 0.0
sim = Simulation(...; inletBC = inletBC, U = 1.5)

# Time-varying inlet
inletBC(i, x, t) = i == 1 ? 1.0 + 0.1*sin(2π*t) : 0.0
sim = Simulation(...; inletBC = inletBC, U = 1.0)

# Periodic in z-direction
sim = Simulation(...; perdir = (2,))

# Convective outlet
sim = Simulation(...; outletBC = true)
```

---

## Next Steps

```@raw html
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1.5rem 0;">

<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea;">
    <strong>GPU Acceleration</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.95em;">Speed up your simulations 10-50x with CUDA.</p>
    <a href="../gpu/">Read GPU Guide</a>
</div>

<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #43e97b;">
    <strong>Adaptive Mesh</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.95em;">Automatic refinement near bodies and features.</p>
    <a href="../amr/">Read AMR Guide</a>
</div>

<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #f093fb;">
    <strong>Examples</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.95em;">Cylinders, fish, FSI, and more.</p>
    <a href="../examples/">Browse Examples</a>
</div>

<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #4facfe;">
    <strong>Core Types</strong>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.95em;">Deep dive into Simulation, Flow, AutoBody.</p>
    <a href="../core_types/">Read Core Types</a>
</div>

</div>
```

---

## Running Examples

The `examples/` directory contains ready-to-run scripts:

```bash
# Flow past cylinder (2D)
julia --project examples/flow_past_cylinder_2d.jl

# Oscillating cylinder
julia --project examples/oscillating_cylinder.jl

# Swimming fish
julia --project examples/swimming_fish.jl

# 3D sphere
julia --project examples/sphere_3d.jl
```

---

## Troubleshooting

### Simulation Explodes (NaN values)

**Cause:** Time step too large or unstable SDF.

**Solutions:**
1. Reduce initial time step: `Δt = 0.1f0`
2. Use fixed time step: `fixed_Δt = 0.05f0`
3. Check SDF is smooth and continuous

### Slow Performance

**Solutions:**
1. Enable GPU: `mem = CuArray`
2. Use Float32 (default) instead of Float64
3. Reduce grid size for testing

### Memory Error

**Solutions:**
1. Reduce grid dimensions
2. Reduce AMR `max_level`
3. For GPU: check `CUDA.memory_status()`

---

## Getting Help

- **Documentation:** You're here! Check the sidebar for more topics.
- **Issues:** [GitHub Issues](https://github.com/subhk/BioFlows.jl/issues)
- **Examples:** `examples/` directory in the repository
