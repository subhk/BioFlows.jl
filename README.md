# BioFlows.jl

A comprehensive Julia package for simulating biological flows using computational fluid dynamics with immersed boundary methods.

## Features

BioFlows.jl provides a complete framework for biological flow simulation with the following capabilities:

### Flow Simulation
- **2D and 3D Navier-Stokes equations** with dimensionalized formulation
- **Multiple grid types**: Standard 2D (xy), 2D in xz-plane, and full 3D
- **Finite volume discretization** with 2nd order accuracy using staggered velocity-pressure grids
- **Time-stepping schemes**: Low-storage Adams-Bashforth, RK3, and RK4
- **Boundary conditions**: Inlet, outlet (pressure/velocity), no-slip, free-slip, periodic walls

### Immersed Boundary Method
- **Boundary data immersion method** for fluid-structure interaction
- **Rigid bodies**: Circles, squares, rectangles with multiple body support
- **Flexible bodies**: Lagrangian formulation for fish-like swimming and flag-like motion
- Support for prescribed motion and constraint-based dynamics

### Advanced Numerical Methods
- **Geometric multigrid solver** for pressure Poisson equation
- **Adaptive mesh refinement** with multiple refinement criteria
- **MPI parallelization** using PencilArrays.jl for distributed computing

### Flexible Body Dynamics
Based on the mathematical formulation in `flexible_bodies.pdf`, supporting:
- Sinusoidal motion at the front with free rear end
- Fixed front end with free motion
- Rotational constraints with customizable parameters
- Variable thickness, rigidity, and density along the body

### Output and Visualization
- **NetCDF format** output with configurable snapshots
- Time interval and iteration-based saving options
- Support for velocity, pressure, vorticity, and body motion data

## Repository Structure

- `src/core`: Core types, grids, and differential operators
- `src/boundary`: Boundary conditions and boundary-layer AMR utilities
- `src/amr`: Adaptive mesh refinement implementations (base, v2, MPI)
- `src/mg`: Multigrid solvers (staggered, WaterLily-style, MPI)
- `src/immersed`: Immersed boundary method utilities
- `src/api`: High-level simulation API
- `src/2D`, `src/3D`: Dimension-specific grids, discretizations, and solvers
- `src/bodies`: Rigid and flexible bodies and controllers
- `src/output`: NetCDF writer and I/O helpers

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/BioFlows.jl")
```

## Dependencies

BioFlows.jl utilizes several specialized Julia packages:
- `GeometricMultigrid.jl` - Multigrid pressure solver
- `ParametricBodies.jl` - Body geometry generation
- `PencilArrays.jl` - MPI parallelization
- `ForwardDiff.jl` - Automatic differentiation
- `NetCDF.jl` - Scientific data output
- `MPI.jl` - Parallel computing

## Quick Start

### Simple 2D Channel Flow

```julia
using BioFlows

# Run a simple 2D channel flow simulation
final_state = run_bioflow_2d(
    nx = 64,           # Grid points in x
    ny = 32,           # Grid points in y  
    Lx = 4.0,          # Domain length
    Ly = 1.0,          # Domain height
    Reynolds = 100.0,   # Reynolds number
    final_time = 5.0,   # Simulation time
    output_file = "channel_flow"
)
```

### Flow Around a Cylinder

```julia
using BioFlows

# Create simulation configuration
config = create_2d_simulation_config(
    nx = 128, ny = 64,
    Lx = 6.0, Ly = 2.0,
    Reynolds = 200.0,
    dt = 0.005,
    final_time = 10.0,
    adaptive_refinement = true
)

# Add circular cylinder
add_rigid_circle!(config, [2.0, 1.0], 0.2)

# Run simulation
solver = create_solver(config)
initial_state = initialize_simulation(config, initial_conditions=:uniform_flow)
final_state = run_simulation(config, solver, initial_state)
```

### Flexible Body Swimming

```julia
using BioFlows

# Configure simulation
config = create_2d_simulation_config(
    nx = 128, ny = 64,
    Lx = 6.0, Ly = 2.0,
    Reynolds = 100.0,
    dt = 0.002,
    final_time = 5.0
)

# Add flexible body with sinusoidal motion
add_flexible_body!(config, 
    [1.5, 1.0],          # Front position
    1.0,                  # Body length
    20,                   # Number of Lagrangian points
    thickness = 0.05,
    rigidity = 10.0,
    front_constraint = :sinusoidal,
    motion_amplitude = 0.2,
    motion_frequency = 2.0
)

# Run simulation
solver = create_solver(config)
initial_state = initialize_simulation(config)
final_state = run_simulation(config, solver, initial_state)
```

### 3D Flow Simulation

```julia
using BioFlows

# 3D flow around a sphere
final_state = run_bioflow_3d(
    nx = 48, ny = 32, nz = 32,
    Lx = 6.0, Ly = 2.0, Lz = 2.0,
    Reynolds = 200.0,
    dt = 0.005,
    final_time = 8.0,
    use_mpi = false,  # Set to true for parallel execution
    output_file = "sphere_flow_3d"
)
```

## Grid Types

BioFlows.jl supports multiple grid configurations:

```julia
# Standard 2D grid (xy-plane)
config = create_2d_simulation_config(..., grid_type = TwoDimensional)

# 2D grid in xz-plane (z is vertical)  
config = create_2d_simulation_config(..., grid_type = TwoDimensionalXZ)

# Full 3D grid
config = create_3d_simulation_config(...)
```

## Boundary Conditions

### Inlet/Outlet Conditions
- **Inlet**: Prescribed velocity `InletBC(u_inlet, v_inlet, w_inlet)`
- **Pressure outlet**: `PressureOutletBC(p_outlet)`
- **Velocity outlet**: `VelocityOutletBC(u_outlet, v_outlet, w_outlet)`

### Wall Conditions
- **No-slip**: `NoSlipBC()` - Zero velocity at walls
- **Free-slip**: `FreeSlipBC()` - Zero normal velocity, free tangential
- **Periodic**: `PeriodicBC()` - Periodic boundaries

## Time-Stepping Schemes

```julia
# Available schemes
time_scheme = :adams_bashforth  # Low-storage Adams-Bashforth
time_scheme = :rk3             # 3rd-order Runge-Kutta
time_scheme = :rk4             # 4th-order Runge-Kutta
```

## Adaptive Mesh Refinement

```julia
# Enable adaptive refinement
config = create_2d_simulation_config(..., adaptive_refinement = true)

# Customize refinement criteria
config.refinement_criteria = AdaptiveRefinementCriteria(
    velocity_gradient_threshold = 1.0,
    pressure_gradient_threshold = 10.0,
    vorticity_threshold = 5.0,
    body_distance_threshold = 0.1,
    max_refinement_level = 3,
    min_grid_size = 0.001
)
```

## MPI Parallelization

```julia
# Enable MPI support
config = create_3d_simulation_config(..., use_mpi = true)

# Run with MPI
# mpirun -np 4 julia my_simulation.jl
```

## Examples

The `examples/` directory contains comprehensive examples:

- `simple_2d_flow.jl` - Basic 2D flow simulations
- `simple_3d_flow.jl` - Basic 3D flow simulations  
- `advanced_features.jl` - Advanced features demonstration

## Mathematical Formulation

### Governing Equations

The package solves the dimensionalized incompressible Navier-Stokes equations:

```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
∇·u = 0
```

Where:
- `u` is the velocity field
- `p` is the pressure
- `ρ` is the fluid density (constant or variable)
- `ν` is the kinematic viscosity
- `f` is the immersed boundary force

### Flexible Body Dynamics

Flexible bodies follow the equations from `flexible_bodies.pdf` (equations 2.5-2.9):

- Lagrangian formulation with tension and bending forces
- Constraint-based motion at the front end
- Free rear end dynamics
- Fluid-structure coupling via immersed boundary method

## Output Format

Results are saved in NetCDF format containing:
- Velocity fields (u, v, w)
- Pressure field (p)  
- Vorticity (2D/3D)
- Body positions and forces
- Grid coordinates and metadata

## Performance

BioFlows.jl is designed for high-performance computing:
- Efficient staggered grid finite volume discretization
- Geometric multigrid solver for optimal scaling
- MPI parallelization for distributed computing
- Adaptive mesh refinement to focus resolution where needed

## Contributing

Contributions are welcome! Please see the contributing guidelines and submit pull requests.

## License

This project is licensed under the MIT License.

## Citation

If you use BioFlows.jl in your research, please cite:

```bibtex
@software{bioflows_jl,
  title={BioFlows.jl: A Julia Package for Biological Flow Simulation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/BioFlows.jl}
}
```

## References

- Flexible body dynamics formulation: See `flexible_bodies.pdf`
- Immersed boundary method: Peskin, C.S. (2002)
- Reference implementations: WaterLily.jl, LUMA code
- Documentation: `LMUAdocs.pdf`
