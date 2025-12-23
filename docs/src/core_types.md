# Core Types

## Simulation

The main container for a BioFlows simulation.

```@docs
Simulation
```

### Constructor

```julia
Simulation(dims::NTuple, inletBC, L::Number;
           U=norm(inletBC), Δt=0.25, ν=0., ϵ=1, g=nothing,
           perdir=(), outletBC=false,
           body::AbstractBody=NoBody(),
           T=Float32, mem=Array)
```

**Arguments:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dims` | `NTuple{N,Int}` | Grid dimensions `(nx, nz)` or `(nx, ny, nz)` |
| `inletBC` | `Tuple` or `Function` | Inlet boundary velocity (see below) |
| `L` | `Number` | Length scale for non-dimensionalization |
| `U` | `Number` | Velocity scale (auto-computed if `inletBC` is constant, **required** if function) |
| `Δt` | `Number` | Initial time step (default: 0.25) |
| `ν` | `Number` | Kinematic viscosity (`Re = U*L/ν`) |
| `ϵ` | `Number` | BDIM kernel width (default: 1) |
| `g` | `Function` or `Nothing` | Acceleration field `g(i,x,t)` |
| `perdir` | `Tuple` | Periodic directions, e.g. `(2,)` |
| `outletBC` | `Bool` | Convective outlet in x-direction |
| `body` | `AbstractBody` | Immersed geometry |
| `T` | `Type` | Float type (`Float32` or `Float64`) |
| `mem` | `Type` | Array backend (`Array` for CPU) |

### Inlet Boundary Conditions

The `inletBC` parameter specifies the inlet velocity and can be:

1. **Constant (Tuple):** Uniform velocity across the inlet
   ```julia
   inletBC = (1.0, 0.0)  # U=1 in x, 0 in z
   ```

2. **Spatially-varying (Function):** Velocity varies with position
   ```julia
   # Parabolic profile: u(z) = U_max * (1 - (z-H)²/H²)
   H = Lz / 2
   inletBC(i, x, t) = i == 1 ? 1.5 * (1 - ((x[2] - H) / H)^2) : 0.0
   ```

3. **Time-varying (Function):** Velocity varies with time
   ```julia
   # Oscillating inlet
   inletBC(i, x, t) = i == 1 ? 1.0 + 0.1*sin(2π*t) : 0.0
   ```

Function signature: `inletBC(i, x, t)` where:
- `i`: Velocity component (1=x, 2=z in 2D; 1=x, 2=y, 3=z in 3D)
- `x`: Position vector `[x, z]` or `[x, y, z]`
- `t`: Time

### Fields

| Field | Description |
|-------|-------------|
| `U` | Velocity scale |
| `L` | Length scale |
| `ϵ` | BDIM kernel width |
| `flow` | `Flow` struct with velocity/pressure fields |
| `body` | Immersed body geometry |
| `pois` | Pressure Poisson solver |

## Flow

The `Flow` struct holds all fluid fields for the simulation.

```@docs
Flow
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `u` | `Array{T,D+1}` | Velocity vector field |
| `u⁰` | `Array{T,D+1}` | Previous velocity (for time stepping) |
| `f` | `Array{T,D+1}` | Force vector field |
| `p` | `Array{T,D}` | Pressure scalar field |
| `σ` | `Array{T,D}` | Divergence scalar field |
| `V` | `Array{T,D+1}` | Body velocity vector (BDIM) |
| `μ₀` | `Array{T,D+1}` | Zeroth moment (BDIM) |
| `μ₁` | `Array{T,D+2}` | First moment tensor (BDIM) |
| `Δt` | `Vector{T}` | Time step history |
| `ν` | `T` | Kinematic viscosity |

## AutoBody

Define geometry implicitly via signed distance functions.

```@docs
AutoBody
```

### Constructor

```julia
AutoBody(sdf, map=(x,t)->x; compose=true)
```

**Arguments:**

| Parameter | Description |
|-----------|-------------|
| `sdf` | Signed distance function `sdf(x, t)` |
| `map` | Coordinate mapping function `map(x, t)` for moving bodies |
| `compose` | Auto-compose `sdf∘map` when true (default) |

### Example: Static Cylinder

```julia
radius = 8
center = [32, 32]
sdf(x, t) = sqrt((x[1] - center[1])^2 + (x[2] - center[2])^2) - radius
body = AutoBody(sdf)
```

### Example: Oscillating Cylinder

```julia
radius = 8
A = 5.0  # Amplitude
ω = 0.5  # Angular frequency

sdf(x, t) = sqrt(x[1]^2 + x[2]^2) - radius
map(x, t) = x .- [0, A * sin(ω * t)]  # Vertical oscillation

body = AutoBody(sdf, map)
```

### Example: Rotating Ellipse

```julia
a, b = 10, 5  # Semi-axes
ω = 0.2       # Angular velocity

sdf(x, t) = sqrt((x[1]/a)^2 + (x[2]/b)^2) - 1

function map(x, t)
    θ = ω * t
    c, s = cos(θ), sin(θ)
    [c*x[1] + s*x[2], -s*x[1] + c*x[2]]
end

body = AutoBody(sdf, map)
```

## Simulation Control Functions

### Time Stepping

```@docs
sim_step!
sim_time
measure!
perturb!
sim_info
```

### Usage

```julia
# Single time step
sim_step!(sim; remeasure=true)

# Integrate to target time
sim_step!(sim, 10.0; remeasure=false, verbose=true)

# Query time
t_star = sim_time(sim)  # Dimensionless time t*U/L
t_raw = time(sim.flow)  # Raw simulation time

# Update body for moving geometry
measure!(sim)

# Add perturbations for flow instability
perturb!(sim; noise=0.1)

# Print status
sim_info(sim)  # Prints: tU/L=..., Δt=...
```
