# BioFlows.jl Examples

This directory contains comprehensive examples demonstrating the capabilities of BioFlows.jl for biological flow simulations.

## Overview

BioFlows.jl provides advanced computational fluid dynamics capabilities with a focus on:
- **Immersed Boundary Method (IBM)** for fluid-structure interaction
- **Flexible Body Dynamics** with PID control systems
- **Adaptive Mesh Refinement (AMR)** for efficient computation
- **MPI Parallelization** for large-scale simulations
- **Multi-physics coupling** for complex biological systems

## Examples Categories

### 1. Basic Flow Simulations

#### `flow_past_cylinder_2d_const_inlet.jl`
**Description**: 2D flow past a circular cylinder with constant inlet velocity.

**Features**:
- Rigid body immersed boundary method
- Pressure outlet boundary condition
- No-slip walls (top/bottom)
- Force coefficient calculation (Cd, Cl)
- NetCDF output for visualization

**Usage**:
```bash
julia --project examples/flow_past_cylinder_2d_const_inlet.jl
```

**Parameters**:
- Domain: 8.0 × 2.0 units
- Grid: 240 × 60 cells
- Reynolds number: ~200 (based on cylinder diameter)
- Simulation time: 10 seconds

**Expected Output**:
- `cylinder2d_const_inlet.nc`: Flow field data
- `cylinder2d_const_inlet_coeffs.nc`: Force coefficients
- Console output with final Cd/Cl values

#### `flow_past_cylinder_2d_mpi.jl`
**Description**: MPI-parallel version of cylinder flow simulation.

**Features**:
- Domain decomposition with ghost cell exchange
- Distributed pressure solving
- Synchronized I/O
- Scalable to multiple processors

**Usage**:
```bash
mpirun -np 4 julia --project examples/flow_past_cylinder_2d_mpi.jl
```

**MPI Notes**:
- Automatic domain decomposition
- Load balancing for irregular processor counts
- Collective I/O for efficient data writing

### 2. Flexible Body PID Control

#### `flexible_body_pid_control.jl`
**Description**: Two flexible flags with PID distance control.

**Features**:
- Flexible body dynamics with prescribed leading edge motion
- PID controller maintaining target inter-body distances
- Real-time distance monitoring and error tracking
- Amplitude limiting for stability

**Key Concepts**:
- **Control Variable**: Leading edge oscillation amplitude
- **Measured Variable**: Distance between trailing edges
- **Control Law**: PID feedback with gains Kp, Ki, Kd

**Usage**:
```bash
julia --project examples/flexible_body_pid_control.jl
```

**Physics Parameters**:
- Flag length: 1.5 units
- Oscillation frequency: 1.5 Hz
- Target distance: 1.5 units (20% increase from initial)
- PID gains: Kp=0.6, Ki=0.15, Kd=0.08

**Expected Behavior**:
- Initial distance: 1.0 units
- Controlled convergence to 1.5 units
- Synchronized oscillation with distance maintenance

#### `multi_flag_coordination.jl`
**Description**: Three flags forming coordinated triangular formation.

**Features**:
- Multi-body PID control system
- Sequential phase coordination (120° phase differences)
- Equilateral triangle formation control
- Performance analysis and stability metrics

**Control Strategy**:
- **Formation**: Equilateral triangle with 2.0 unit sides
- **Phase Pattern**: Sequential (0°, 120°, 240°)
- **Distance Pairs**: All three inter-flag distances controlled simultaneously

**Usage**:
```bash
julia --project examples/multi_flag_coordination.jl
```

**Advanced Features**:
- Multi-objective control (3 distance constraints)
- Phase relationship analysis
- Formation stability assessment
- Coordination quality metrics

#### `adaptive_pid_tuning.jl`
**Description**: Automatic PID gain optimization based on performance.

**Features**:
- Real-time performance monitoring
- Adaptive gain adjustment algorithms
- Convergence quality assessment
- Stability constraint enforcement

**Adaptation Logic**:
1. **High Error**: Increase Kp and Ki for faster response
2. **High Oscillation**: Increase Kd for better damping
3. **Good Performance**: Fine-tune for optimal response
4. **Stability Limits**: Enforce gain constraints

**Usage**:
```bash
julia --project examples/adaptive_pid_tuning.jl
```

**Tuning Parameters**:
- Initial gains: Kp=0.2, Ki=0.05, Kd=0.02 (conservative)
- Adaptation period: 100 time steps
- Performance metrics: Average error, stability, peak error
- Maximum iterations: 10 adaptations

### 3. Visualization and Analysis

#### `plot_vorticity_cylinder.jl`
**Description**: Visualization tool for 2D vorticity fields with cylinder overlay.

**Features**:
- NetCDF data reading and processing
- Vorticity field computation and visualization
- Cylinder geometry overlay
- Customizable time selection and parameters

**Usage**:
```bash
# Plot last time step with auto-detected cylinder
julia --project examples/plot_vorticity_cylinder.jl cylinder2d_const_inlet.nc

# Plot specific time with custom cylinder parameters
julia --project examples/plot_vorticity_cylinder.jl output.nc --time 5.0 --xc 1.2 --zc 1.0 --radius 0.1
```

**Options**:
- `--time`: Time selection (last, integer index, or float time)
- `--xc`, `--zc`: Cylinder center coordinates
- `--radius`: Cylinder radius

#### `animate_vorticity_cylinder.jl`
**Description**: Animation generation for time-series vorticity data.

**Features**:
- Multi-frame animation creation
- Temporal evolution visualization
- Customizable frame rates and quality
- GIF and MP4 export options

**Usage**:
```bash
julia --project examples/animate_vorticity_cylinder.jl cylinder2d_const_inlet.nc
```

## Running Examples

### Prerequisites
1. **Julia Environment**: Ensure BioFlows.jl is activated
   ```bash
   julia --project
   julia> ]instantiate
   ```

2. **MPI Setup** (for parallel examples):
   ```bash
   # Install MPI.jl if not already available
   julia --project -e "using Pkg; Pkg.add(\"MPI\")"
   ```

3. **Visualization Dependencies**:
   ```bash
   julia --project -e "using Pkg; Pkg.add([\"Plots\", \"PlotlyJS\"])"
   ```

### Quick Start
```bash
# Basic flow simulation
julia --project examples/flow_past_cylinder_2d_const_inlet.jl

# Flexible body control
julia --project examples/flexible_body_pid_control.jl

# Visualization
julia --project examples/plot_vorticity_cylinder.jl cylinder2d_const_inlet.nc
```

## Understanding the Output

### NetCDF Files
All simulations produce NetCDF files containing:
- **Flow fields**: u, v, w, p (velocity components and pressure)
- **Grid data**: x, y, z coordinates
- **Time series**: temporal evolution data
- **Metadata**: simulation parameters and body information

### Force Coefficients
Coefficient files (`*_coeffs.nc`) contain:
- **Cd**: Drag coefficient
- **Cl**: Lift coefficient  
- **Fx, Fy, Fz**: Force components
- **Time**: Corresponding time values

### PID Control Data
Flexible body examples output:
- Distance error evolution
- Amplitude adjustments over time
- Controller performance metrics
- Formation quality assessments

## Theory and Background

### PID Control for Flexible Bodies

The PID controller maintains target distances between flexible bodies by adjusting their leading edge oscillation amplitudes:

**Control Equation**:
```
u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de/dt
A(t+1) = A(t) + α*u(t)
```

Where:
- `e(t)`: Distance error from target
- `A(t)`: Leading edge amplitude
- `α`: Control scale factor
- `Kp, Ki, Kd`: PID gains

**Stability Considerations**:
- **Amplitude Limits**: Prevent actuator saturation
- **Gain Constraints**: Maintain closed-loop stability
- **Sampling Rate**: Match control frequency to system dynamics

### Immersed Boundary Method

BioFlows implements the direct forcing IBM with:
- **Force Spreading**: Lagrangian → Eulerian
- **Velocity Interpolation**: Eulerian → Lagrangian
- **Regularized Delta Functions**: Smooth force distribution
- **Conservative Interpolation**: Momentum preservation

## Performance Tips

### Computational Efficiency
1. **Grid Resolution**: Balance accuracy vs. computational cost
2. **Time Step Size**: Ensure CFL stability condition
3. **AMR Usage**: Enable for complex geometries
4. **MPI Parallelization**: Use for large simulations

### PID Tuning Guidelines
1. **Start Conservative**: Low gains, increase gradually
2. **Monitor Stability**: Watch for oscillations or instability
3. **Use Adaptation**: Let the system self-tune when possible
4. **Consider Physics**: Match control frequency to natural dynamics

### Troubleshooting

#### Common Issues
1. **Simulation Divergence**: Reduce time step, check CFL condition
2. **PID Instability**: Lower gains, add amplitude limits
3. **Poor Convergence**: Increase integration time, check target feasibility
4. **MPI Errors**: Verify domain decomposition and ghost cell exchange

#### Performance Optimization
1. **Memory Usage**: Monitor for large grids and long simulations
2. **I/O Frequency**: Balance data output with simulation speed
3. **Visualization**: Use reasonable frame rates for animations

## Contributing

To add new examples:
1. Follow the existing naming convention
2. Include comprehensive documentation
3. Add parameter descriptions and expected outputs
4. Test across different system configurations
5. Update this README with the new example

## References

1. **IBM Theory**: Peskin, C.S. "The immersed boundary method." Acta Numerica 11 (2002).
2. **PID Control**: Franklin, G.F. et al. "Feedback Control of Dynamic Systems." 7th Ed.
3. **Flexible Bodies**: Huang, W.-X. "An immersed boundary method for fluid–flexible structure interaction." CMAME (2009).
4. **Adaptive Control**: Åström, K.J. "Adaptive Control." 2nd Edition, Dover Publications.

---

For more detailed theory and implementation details, see `PID_Controller_Documentation.md` in the main repository directory.