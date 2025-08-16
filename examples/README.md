# BioFlow.jl Examples: 2D Flow Past Cylinder

This directory contains a comprehensive example demonstrating 2D flow past a circular cylinder using BioFlow.jl with advanced features including adaptive mesh refinement and detailed output capabilities.

## Files

### Main Simulation Scripts
- **`flow_past_cylinder_2d.jl`** - Complete cylinder flow simulation with AMR
- **`run_cylinder_example.jl`** - Simplified runner with parameter variations
- **`analyze_cylinder_results.jl`** - Post-processing and analysis tools

### Example Features

#### Physical Setup
- **Domain**: 8.0 × 4.0 m (length × height) in XZ plane
- **Cylinder**: Radius = 0.2 m, center at (2.0, 2.0)
- **Reynolds number**: 100 (based on cylinder diameter)
- **Inlet**: Constant velocity U = 1.0 m/s
- **Outlet**: Pressure flux boundary condition
- **Walls**: No-slip at top and bottom boundaries

#### Numerical Features
- **Grid**: 160 × 80 points (20 points per cylinder diameter)
- **Time integration**: 4th order Runge-Kutta
- **Adaptive mesh refinement**: Up to 3 levels near cylinder and wake
- **Immersed boundary method**: For cylinder representation
- **File output**: NetCDF format with comprehensive data

## Usage

### Quick Start

```julia
# Run basic simulation
julia run_cylinder_example.jl

# Or directly
include("flow_past_cylinder_2d.jl")
demonstrate_cylinder_flow()
```

### Custom Parameters

```julia
# Run with different Reynolds number
include("run_cylinder_example.jl")
run_cylinder_custom_re(150.0)

# High resolution simulation
run_cylinder_high_res()

# Parameter study
reynolds_study()
```

### Analysis

```julia
# Comprehensive analysis
include("analyze_cylinder_results.jl")
comprehensive_analysis("cylinder_flow_2d", Re=100.0)

# Individual analyses
t, Cd, Cl = load_force_coefficients("cylinder_flow_2d_forces.nc")
f_shed, St = analyze_shedding_frequency(t, Cl)
```

## Expected Results

### Flow Physics
At Re = 100, the flow exhibits:
- **Unsteady vortex shedding** behind the cylinder
- **Periodic lift fluctuations** due to alternating vortices
- **von Kármán vortex street** in the wake
- **Strouhal number** St ≈ 0.164 ± 0.005

### Force Coefficients
- **Mean drag coefficient**: Cd ≈ 1.33 ± 0.05
- **Lift coefficient oscillations**: Cl amplitude ≈ 0.3
- **Shedding frequency**: f ≈ 0.164 Hz

### Validation Data
The results can be compared with:
- Experimental data (Williamson, 1996)
- DNS simulations (Henderson, 1995)
- Literature correlations for circular cylinders

## Output Files

### Flow Field Data
- `cylinder_flow_2d_NNNN.nc` - Velocity, pressure, vorticity fields
- Variables: `u`, `w` (velocities), `p` (pressure), `vorticity`
- Coordinates: `x`, `z` (XZ plane), `time`

### Force Data
- `cylinder_flow_2d_forces.nc` - Time series of forces
- Variables: `drag_coefficient`, `lift_coefficient`, `drag_force`, `lift_force`

### AMR Data
- `cylinder_flow_2d_amr_NNNN.nc` - Refinement level maps
- Variables: `refinement_level`, `grid_metrics`

## Visualization

### Using Julia Plots.jl
```julia
using Plots, NCDatasets

# Load data
ds = NCDataset("cylinder_flow_2d_0001.nc")
x, z = ds["x"][:], ds["z"][:]
vorticity = ds["vorticity"][:,:,end]  # Last time step

# Create contour plot
contour(x, z, vorticity', 
        levels=20, 
        xlabel="x (m)", 
        ylabel="z (m)", 
        title="Vorticity Field")

# Add cylinder outline
θ = 0:0.1:2π
x_cyl = 2.0 .+ 0.2.*cos.(θ)
z_cyl = 2.0 .+ 0.2.*sin.(θ)
plot!(x_cyl, z_cyl, color=:black, linewidth=2)
```

### Using Python/Matplotlib
```python
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# Load data
ds = nc.Dataset('cylinder_flow_2d_0001.nc')
x, z = ds['x'][:], ds['z'][:]
vorticity = ds['vorticity'][:,:,-1]  # Last time step

# Create contour plot
plt.figure(figsize=(12, 6))
plt.contour(x, z, vorticity.T, levels=20)
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('Vorticity Field')

# Add cylinder
theta = np.linspace(0, 2*np.pi, 100)
x_cyl = 2.0 + 0.2*np.cos(theta)
z_cyl = 2.0 + 0.2*np.sin(theta)
plt.plot(x_cyl, z_cyl, 'k-', linewidth=2)
plt.axis('equal')
plt.show()
```

## Performance Notes

### Computational Requirements
- **Memory**: ~500 MB for base simulation
- **Runtime**: ~10-15 minutes on modern CPU (single core)
- **Storage**: ~100 MB output files for 20 seconds simulation

### Optimization Tips
- Use AMR to focus resolution near cylinder and wake
- Increase `max_refinement_level` for higher accuracy
- Adjust `output_interval` to balance file size vs temporal resolution
- Enable MPI for larger simulations

## Advanced Usage

### Parameter Studies
```julia
# Vary Reynolds number
Re_values = [40, 60, 80, 100, 150, 200]
for Re in Re_values
    run_cylinder_custom_re(Re)
end
```

### Custom Boundary Conditions
```julia
# Modify boundary conditions
config = create_2d_simulation_config(
    # ... other parameters ...
    wall_type = :free_slip,    # Change to free-slip walls
    outlet_type = :velocity    # Change to velocity outlet
)
```

### Different Cylinder Positions
```julia
# Move cylinder position
cylinder = Circle(0.2, [1.5, 2.0])  # Closer to inlet
```

## Troubleshooting

### Common Issues
1. **Simulation crashes**: Check CFL condition, reduce time step
2. **Poor convergence**: Increase grid resolution or AMR levels
3. **Unphysical results**: Verify boundary conditions and Reynolds number
4. **Large file sizes**: Reduce output frequency or variables saved

### Validation Checks
- Monitor drag coefficient convergence
- Check vortex shedding frequency matches literature
- Verify mass conservation in output files
- Inspect AMR refinement patterns

## References

1. Williamson, C.H.K. (1996). "Vortex dynamics in the cylinder wake." *Annual Review of Fluid Mechanics*, 28(1), 477-539.

2. Henderson, R.D. (1995). "Details of the drag curve near the onset of vortex shedding." *Physics of Fluids*, 7(9), 2102-2104.

3. Roshko, A. (1961). "Experiments on the flow past a circular cylinder at very high Reynolds number." *Journal of Fluid Mechanics*, 10(3), 345-356.

## Support

For questions or issues:
- Check the main BioFlow.jl documentation
- Review simulation parameters and boundary conditions
- Verify file paths and output directory permissions
- Compare results with provided validation data