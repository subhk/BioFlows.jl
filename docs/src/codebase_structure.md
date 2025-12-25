# Codebase Structure

This document provides an overview of the BioFlows.jl source code organization, module dependencies, and key components.

## Directory Layout

```
BioFlows.jl/
├── src/                          # Source code
│   ├── BioFlows.jl               # Main module file
│   ├── Flow.jl                   # Flow solver, BDIM, time stepping
│   ├── Poisson.jl                # Base Poisson pressure solver
│   ├── MultiLevelPoisson.jl      # Multigrid pressure solver
│   ├── Body.jl                   # Abstract body types and SDF utilities
│   ├── AutoBody.jl               # Implicit geometry via SDFs
│   ├── util.jl                   # Utilities, macros, boundary conditions
│   ├── Diagnostics.jl            # Force computation, vorticity
│   ├── Metrics.jl                # Flow statistics (mean flow, Reynolds stress)
│   ├── Output.jl                 # I/O utilities (VTK, JLD2 writers)
│   ├── amr/                      # Adaptive Mesh Refinement
│   │   ├── amr_types.jl          # Core AMR data structures
│   │   ├── amr_project.jl        # AMR projection and momentum step
│   │   ├── bioflows_amr_adapter.jl  # Flow-to-grid adapters
│   │   ├── body_refinement_indicator.jl  # Refinement criteria
│   │   ├── composite_poisson.jl  # Multi-level pressure solver
│   │   ├── composite_solver.jl   # Composite grid solver
│   │   ├── interface_operators.jl # Coarse-fine interface operators
│   │   ├── patch_creation.jl     # AMR patch management
│   │   ├── patch_poisson.jl      # Patch-level pressure solver
│   │   └── refined_fields.jl     # Refined velocity field storage
│   └── fsi/                      # Fluid-Structure Interaction
│       ├── EulerBernoulliBeam.jl # Beam dynamics solver
│       ├── FluidStructureCoupling.jl  # Beam-fluid coupling
│       └── BeamAMR.jl            # Beam-aware AMR
├── ext/                          # Package extensions
│   ├── BioFlowsJLD2Ext.jl        # JLD2 I/O extension
│   ├── BioFlowsReadVTKExt.jl     # VTK reading extension
│   ├── BioFlowsWriteVTKExt.jl    # VTK writing extension
│   ├── BioFlowsPlotsExt.jl       # Plots.jl integration
│   └── BioFlowsMakieExt.jl       # Makie.jl integration
├── examples/                     # Example scripts
├── test/                         # Test suite
└── docs/                         # Documentation
```

## Core Modules

### Main Module (`BioFlows.jl`)

The entry point that:
- Imports all dependencies (KernelAbstractions, MPI, ForwardDiff, etc.)
- Includes all source files in dependency order
- Defines `Simulation` and `AMRSimulation` types
- Exports public API

### Flow Solver (`Flow.jl`)

The heart of the CFD solver implementing:

| Function | Description |
|----------|-------------|
| `Flow()` | Constructor for flow state (velocity, pressure, coefficients) |
| `mom_step!()` | Full momentum step: predictor + pressure projection + corrector |
| `BDIM!()` | Boundary Data Immersion Method for immersed boundaries |
| `conv_diff!()` | Convection-diffusion operator (advection + viscous terms) |
| `accelerate!()` | Apply body forces (gravity, prescribed accelerations) |
| `exitBC!()` | Convective outlet boundary condition |
| `project!()` | Pressure projection for divergence-free velocity |
| `scale_u!()` | Velocity scaling for time stepping |

**Key data structures in `Flow`:**
```julia
struct Flow{T,D}
    u::Array{T,D+1}     # Velocity field (staggered)
    u⁰::Array{T,D+1}    # Previous velocity
    p::Array{T,D}       # Pressure
    f::Array{T,D+1}     # RHS / forcing
    σ::Array{T,D}       # Divergence
    μ₀::Array{T,D+1}    # BDIM volume fraction
    μ₁::Array{T,D+2}    # BDIM first moment (gradient correction)
    V::Array{T,D+1}     # Body velocity
    Δx::NTuple{D,T}     # Grid spacing
    Δt::Vector{T}       # Time step history
    # ... additional fields
end
```

### Pressure Solvers

#### `Poisson.jl` - Base Solver
- Conjugate gradient (CG) solver with geometric multigrid preconditioner
- Handles periodic and Neumann boundary conditions
- Key functions: `solver!()`, `mult!()`, `residual!()`

#### `MultiLevelPoisson.jl` - Multigrid Solver
- V-cycle multigrid preconditioner
- Automatic coarsening up to 4x4 base grid
- Restriction and prolongation operators

### Boundary Conditions (`util.jl`)

| Function | Description |
|----------|-------------|
| `BC!()` | Apply velocity boundary conditions |
| `exitBC!()` | Convective outlet BC (1D upwind advection) |
| `perBC!()` | Periodic boundary conditions for scalars |

**BC Types:**
- **Dirichlet**: Normal velocity component at domain boundaries
- **Neumann**: Zero-gradient for tangential components (stress-free)
- **Periodic**: Ghost cell copying for cyclic boundaries
- **BDIM**: No-slip on immersed bodies via μ₀/μ₁ weighting

### Body Definitions

#### `Body.jl`
- `AbstractBody` type hierarchy
- `NoBody()` for flows without immersed objects
- `measure_sdf!()` for computing signed distance fields

#### `AutoBody.jl`
- Implicit geometry via signed distance functions (SDFs)
- Supports CSG operations: union (`+`), subtraction (`-`)
- Time-dependent geometry via `map(x, t)` transformations

```julia
# Example: Cylinder at (cx, cz) with radius R
sdf(x, t) = sqrt((x[1] - cx)^2 + (x[2] - cz)^2) - R
body = AutoBody(sdf)

# Moving cylinder with velocity (Vx, Vz)
map(x, t) = x .- (Vx*t, Vz*t)
body = AutoBody(sdf, map)
```

## Adaptive Mesh Refinement (AMR)

### Architecture

```
AMRSimulation
    ├── Simulation (base grid)
    ├── AMRConfig (refinement parameters)
    ├── RefinedGrid (cell tracking)
    ├── CompositePoisson (multi-level pressure solver)
    │   ├── base_pois (MultiLevelPoisson)
    │   └── patches (Dict of PatchPoisson)
    └── FlowToGridAdapter
```

### AMR Files

| File | Purpose |
|------|---------|
| `amr_types.jl` | `StaggeredGrid`, `RefinedGrid`, `SolutionState` types |
| `amr_project.jl` | `amr_project!()`, `amr_mom_step!()` |
| `body_refinement_indicator.jl` | Refinement criteria based on body distance, gradients, vorticity |
| `composite_poisson.jl` | `CompositePoisson` multi-level solver |
| `composite_solver.jl` | Composite grid iteration |
| `interface_operators.jl` | Coarse-fine interpolation, restriction |
| `patch_creation.jl` | Create/manage `PatchPoisson` instances |
| `patch_poisson.jl` | Individual patch pressure solver |
| `refined_fields.jl` | `RefinedVelocityField` for storing fine-level data |

### Key AMR Functions

```julia
# Create AMR-enabled simulation
sim = AMRSimulation((nx, nz), (Lx, Lz);
    body=body,
    amr_config=AMRConfig(max_level=2))

# Time stepping with automatic regridding
sim_step!(sim; remeasure=true)

# Force immediate regrid
force_regrid!(sim)

# Check divergence at all levels
check_divergence(sim; verbose=true)
```

## Fluid-Structure Interaction (FSI)

### FSI Files

| File | Purpose |
|------|---------|
| `EulerBernoulliBeam.jl` | Beam dynamics with Hermite finite elements |
| `FluidStructureCoupling.jl` | `FSISimulation`, fluid-beam coupling |
| `BeamAMR.jl` | `BeamAMRSimulation`, AMR for flexible bodies |

### Beam Model

- Euler-Bernoulli beam theory (small deflections)
- Hermite cubic shape functions (C1 continuity)
- Newmark-beta time integration
- Boundary conditions: CLAMPED, FREE, PINNED, PRESCRIBED

```julia
# Create a flexible beam
beam = EulerBernoulliBeam(
    n_elements = 20,
    length = 1.0,
    material = BeamMaterial(E=1e6, rho=1000),
    geometry = BeamGeometry(width=0.1, thickness_func=fish_thickness_profile),
    left_bc = CLAMPED,
    right_bc = FREE
)

# FSI simulation
fsi_sim = FSISimulation((256, 128), (2.0, 1.0);
    beam = beam,
    ν = 1e-3)
```

## Diagnostics (`Diagnostics.jl`)

### Force Computation

| Function | Description |
|----------|-------------|
| `pressure_force(sim)` | Pressure contribution to force |
| `viscous_force(sim)` | Viscous contribution to force |
| `total_force(sim)` | Sum of pressure and viscous forces |
| `force_coefficients(sim)` | Cd, Cl (drag and lift coefficients) |

### Flow Quantities

| Function | Description |
|----------|-------------|
| `ω(flow)` | Vorticity field (2D: scalar, 3D: vector) |
| `ω_mag(flow)` | Vorticity magnitude |
| `cell_center_velocity(flow)` | Interpolate staggered velocity to cell centers |
| `cell_center_pressure(flow)` | Cell-centered pressure |

## Memory and Backends

BioFlows.jl supports multiple compute backends via KernelAbstractions.jl:

```julia
# CPU (default)
sim = Simulation(dims, L; mem=Array, ...)

# GPU (requires CUDA.jl)
using CUDA
sim = Simulation(dims, L; mem=CuArray, ...)
```

### Backend Selection

```julia
# Check current backend
BioFlows.backend

# Set backend
BioFlows.set_backend("KernelAbstractions")  # Default, multi-threaded
BioFlows.set_backend("SIMD")                # Single-threaded, optimized
```

## Module Dependency Graph

```
BioFlows.jl
    │
    ├── util.jl (macros, BC functions)
    │
    ├── Poisson.jl
    │   └── MultiLevelPoisson.jl
    │
    ├── Body.jl
    │   └── AutoBody.jl
    │
    ├── Flow.jl (depends on Poisson, Body, util)
    │
    ├── amr/ (depends on Flow, Poisson)
    │   ├── amr_types.jl
    │   ├── bioflows_amr_adapter.jl
    │   ├── body_refinement_indicator.jl
    │   ├── refined_fields.jl
    │   ├── patch_poisson.jl
    │   ├── interface_operators.jl
    │   ├── composite_poisson.jl
    │   ├── composite_solver.jl
    │   ├── patch_creation.jl
    │   └── amr_project.jl
    │
    ├── fsi/ (depends on Flow, amr/)
    │   ├── EulerBernoulliBeam.jl
    │   ├── FluidStructureCoupling.jl
    │   └── BeamAMR.jl
    │
    ├── Metrics.jl (depends on Flow)
    ├── Diagnostics.jl (depends on Flow, Body)
    └── Output.jl (depends on Flow, Diagnostics)
```

## Key Conventions

### Indexing

- **Staggered grid**: Velocity stored at face centers
- **Ghost cells**: 1-cell layer at domain boundaries
- **Interior indices**: `2:N+1` for N interior cells
- **`@inside` macro**: Generates loops over interior cells

### Coordinate System

- **2D**: `(x, z)` where `x` is streamwise, `z` is vertical
- **3D**: `(x, y, z)` standard Cartesian
- **SDF convention**: Negative inside body, positive outside

### Naming Conventions

| Suffix | Meaning |
|--------|---------|
| `!` | Mutating function (modifies arguments) |
| `_u` | Operates on velocity fields |
| `_p` | Operates on pressure fields |
| `⁰` | Previous time step value |
| `μ₀`, `μ₁` | BDIM moments |

## Extending BioFlows

### Adding New Body Types

```julia
# Define custom body
struct MyBody <: AbstractBody
    # fields
end

# Implement required methods
BioFlows.measure!(flow::Flow, body::MyBody; kwargs...) = ...
```

### Adding New Diagnostics

```julia
# Add to Diagnostics.jl or create extension
function my_diagnostic(sim::Simulation)
    flow = sim.flow
    # Compute diagnostic from flow fields
end
```

### Creating Extensions

Use Julia's package extension mechanism:

```julia
# In ext/MyExtension.jl
module MyExtension

using BioFlows
using MyDependency  # Triggers extension loading

# Extend BioFlows functions
BioFlows.my_function(args...) = ...

end
```

## Performance Tips

1. **Use `remeasure=false`** for static bodies to skip SDF recomputation
2. **Set `JULIA_NUM_THREADS=auto`** for multi-threaded CPU execution
3. **Use GPU arrays** (`CuArray`) for large 3D simulations
4. **Tune AMR parameters**: Lower `regrid_interval` for fast-moving bodies
5. **Use `store_fluxes=false`** (default) unless debugging conservation
