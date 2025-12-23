# Getting Started

## Installation

### From GitHub

```julia
using Pkg
Pkg.add(url = "https://github.com/subhk/BioFlows.jl")
```

### Local Development

Clone the repository and activate:

```bash
git clone https://github.com/subhk/BioFlows.jl.git
cd BioFlows.jl
julia --project
```

Then instantiate dependencies:

```julia
using Pkg
Pkg.instantiate()
```

## Your First Simulation

Here's a complete example of simulating flow past a 2D cylinder:

```julia
using BioFlows

# Physical parameters
Re = 100        # Reynolds number
U = 1.0         # Inlet velocity
D = 16.0        # Cylinder diameter (length scale)

# Grid parameters
nx, nz = 128, 64

# Define cylinder geometry
center_x = nx / 4
center_z = nz / 2
radius = D / 2

# Signed distance function (negative inside body)
sdf(x, t) = sqrt((x[1] - center_x)^2 + (x[2] - center_z)^2) - radius

# Create simulation
sim = Simulation((nx, nz), (U, 0), D;
                 ν = U * D / Re,
                 body = AutoBody(sdf))

# Time integration
final_time = 10.0  # Convective time units

while sim_time(sim) < final_time
    sim_step!(sim; remeasure=false)

    # Print progress every 100 steps
    if length(sim.flow.Δt) % 100 == 0
        println("t = ", round(sim_time(sim), digits=2))
    end
end

println("Simulation complete!")
```

## Key Concepts

### Dimensionless Time

BioFlows uses dimensionless (convective) time:

```math
t^* = \frac{t \cdot U}{L}
```

where `U` is the velocity scale and `L` is the length scale.

- `sim_time(sim)` returns the current dimensionless time `t*`
- `time(sim.flow)` returns the raw simulation time `t`

### Signed Distance Functions

Geometry is defined implicitly via signed distance functions (SDF):
- `sdf(x, t) < 0`: Inside the body
- `sdf(x, t) = 0`: On the body surface
- `sdf(x, t) > 0`: Outside the body

Example SDFs:

```julia
# Circle/Cylinder
sdf_circle(x, t) = sqrt(x[1]^2 + x[2]^2) - radius

# Rectangle
sdf_rect(x, t) = max(abs(x[1]) - width/2, abs(x[2]) - height/2)

# Sphere (3D)
sdf_sphere(x, t) = sqrt(x[1]^2 + x[2]^2 + x[3]^2) - radius
```

### Boundary Conditions

Specify inlet boundary conditions via `inletBC`:

```julia
# Constant inlet velocity (uniform flow)
sim = Simulation(dims, (1.0, 0.0), L; ...)

# Spatially-varying inlet (parabolic profile in z)
# Function signature: inletBC(i, x, t) where i=component, x=position, t=time
H = Lz / 2  # channel half-height
U_max = 1.5
inletBC(i, x, t) = i == 1 ? U_max * (1 - ((x[2] - H) / H)^2) : 0.0
sim = Simulation(dims, inletBC, L; U=U_max, ...)  # Must specify U for functions

# Time-varying inlet (oscillating)
inletBC(i, x, t) = i == 1 ? 1.0 + 0.1*sin(2π*t) : 0.0
sim = Simulation(dims, inletBC, L; U=1.0, ...)
```

Additional boundary options:
- `perdir=(2,)`: Make direction 2 (z) periodic
- `outletBC=true`: Convective outlet in x-direction

## Running Examples

The `examples/` directory contains ready-to-run scripts:

```bash
julia --project examples/flow_past_cylinder_2d.jl
julia --project examples/circle_benchmark.jl
julia --project examples/oscillating_cylinder.jl
```

See [Examples](@ref) for detailed descriptions.
