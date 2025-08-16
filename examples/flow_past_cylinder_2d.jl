"""
2D Flow Past Cylinder Example with Adaptive Mesh Refinement

This example demonstrates:
- 2D flow simulation in XZ plane (z is vertical)
- Constant inlet velocity boundary condition
- Pressure flux outlet boundary condition
- No-slip walls at top and bottom boundaries
- Rigid cylinder body using immersed boundary method
- Adaptive mesh refinement around the cylinder
- NetCDF file output with time series data

Physical Setup:
- Domain: 16.0 × 4.0 (length × height) in XZ plane
- Cylinder: radius = 0.2, center at (2.0, 2.0)
- Reynolds number: 100 (based on cylinder diameter and inlet velocity)
- Inlet velocity: U = 1.0 m/s
- Fluid density: ρ = 1000.0 kg/m³
"""

using BioFlows
using Printf

function create_cylinder_simulation()
    """
    Create simulation configuration for flow past cylinder.
    """
    
    # Physical parameters
    U_inlet = 1.0                      # Inlet velocity [m/s]
    D_cylinder = 0.4                   # Cylinder diameter [m]
    Re = 100.0                         # Reynolds number
    ρ = 1000.0                         # Density [kg/m³]
    μ = ρ * U_inlet * D_cylinder / Re  # Dynamic viscosity [Pa·s]
    
    # Domain parameters
    Lx = 16.0               # Domain length [m]
    Lz = 4.0               # Domain height [m] (vertical in XZ plane)
    nx = 160               # Grid points in x-direction
    nz = 80                # Grid points in z-direction (vertical)
    
    # Time parameters
    dt = 0.01              # Time step [s]
    final_time = 20.0      # Final simulation time [s]
    
    # Create simulation configuration
    config = create_2d_simulation_config(
        nx = nx,
        nz = nz,
        Lx = Lx,
        Lz = Lz,
        origin = [0.0, 0.0],  # [x_min, z_min] for XZ plane
        
        # Fluid properties
        Reynolds = Re,
        density_type = :constant,
        density_value = ρ,
        viscosity = μ,
        
        # Boundary conditions
        inlet_velocity = U_inlet,     # Constant inlet velocity
        outlet_type = :pressure,      # Pressure flux at outlet
        wall_type = :no_slip,         # No-slip at top/bottom walls
        
        # Time integration
        time_scheme = :rk4,           # 4th order Runge-Kutta
        dt = dt,
        final_time = final_time,
        
        # Advanced features
        adaptive_refinement = true,   # Enable AMR
        use_mpi = false,             # Single process for this example
        
        # Output configuration
        output_interval = 0.1,        # Save every 0.1 seconds
        output_file = "cylinder_flow_2d"
    )
    
    @printf "Created simulation configuration:\n"
    @printf "  Domain: %.1f × %.1f (XZ plane)\n" Lx Lz
    @printf "  Grid: %d × %d points\n" nx nz
    @printf "  Reynolds number: %.1f\n" Re
    @printf "  Inlet velocity: %.2f m/s\n" U_inlet
    @printf "  Viscosity: %.6f Pa·s\n" μ
    
    return config
end

function create_cylinder_body()
    """
    Create rigid cylinder body for immersed boundary method.
    """
    
    # Cylinder parameters
    radius = 0.2           # Cylinder radius [m]
    center = [2.0, 2.0]    # Cylinder center [x, z] in XZ plane
    
    # Create circular rigid body
    cylinder = Circle(radius, center;
                     velocity = [0.0, 0.0],      # Stationary cylinder
                     angular_velocity = 0.0,     # No rotation
                     fixed = true,               # Fixed in place
                     mass = 1.0,                 # Mass (not used for fixed body)
                     moment_inertia = 1.0        # Moment of inertia
                     )
    
    # Create rigid body collection
    bodies = RigidBodyCollection()
    add_body!(bodies, cylinder)
    
    @printf "Created cylinder body:\n"
    @printf "  Radius: %.2f m\n" radius
    @printf "  Center: [%.1f, %.1f] (XZ plane)\n" center[1] center[2]
    @printf "  Type: Fixed circular cylinder\n"
    
    return bodies
end

function setup_adaptive_refinement()
    """
    Configure adaptive mesh refinement criteria.
    """
    
    # AMR criteria - refine near:
    # 1. High velocity gradients (boundary layers)
    # 2. High pressure gradients (stagnation/separation points)
    # 3. High vorticity (wake region)
    # 4. Near immersed bodies (cylinder surface)
    
    criteria = AdaptiveRefinementCriteria(
        velocity_gradient_threshold = 2.0,     # |∇u| threshold
        pressure_gradient_threshold = 5.0,     # |∇p| threshold  
        vorticity_threshold = 3.0,             # |ω| threshold
        body_proximity_distance = 0.5,         # Refine within 0.5m of cylinder
        max_refinement_level = 3,              # Up to 3 levels of refinement
        min_grid_size = 0.01,                  # Minimum cell size
        refinement_frequency = 10              # Check every 10 time steps
    )
    
    @printf "Configured AMR criteria:\n"
    @printf "  Velocity gradient threshold: %.1f\n" criteria.velocity_gradient_threshold
    @printf "  Pressure gradient threshold: %.1f\n" criteria.pressure_gradient_threshold
    @printf "  Vorticity threshold: %.1f\n" criteria.vorticity_threshold
    @printf "  Body proximity distance: %.2f m\n" criteria.body_proximity_distance
    @printf "  Max refinement levels: %d\n" criteria.max_refinement_level
    
    return criteria
end

function setup_output_configuration()
    """
    Configure NetCDF output settings for comprehensive data saving.
    """
    
    # Output configuration
    output_config = NetCDFConfig(
        filename = "cylinder_flow_2d",
        output_interval = 0.1,               # Save every 0.1 seconds
        max_snapshots_per_file = 50,         # 50 snapshots per file (5 seconds)
        
        # Flow field variables
        save_velocity = true,                # u, w velocity components
        save_pressure = true,                # Pressure field
        save_vorticity = true,               # Vorticity field
        
        # Additional diagnostics
        save_velocity_magnitude = true,      # |u| field
        save_pressure_coefficient = true,    # Cp field
        save_streamfunction = false,         # ψ field (expensive to compute)
        
        # Body forces and coefficients
        save_body_forces = true,             # Drag and lift forces
        save_force_coefficients = true,      # Cd and Cl coefficients
        
        # AMR information
        save_refinement_level = true,        # Refinement level field
        save_grid_metrics = true,            # Grid quality metrics
        
        # Compression and performance
        compression_level = 4,               # Moderate compression
        chunking = true,                     # Enable chunking for performance
        
        # Time series analysis
        save_time_series = true,             # Body force time series
        time_series_variables = [:drag_coefficient, :lift_coefficient, 
                                :pressure_drop, :flow_rate]
    )
    
    @printf "Configured output settings:\n"
    @printf "  Output interval: %.2f s\n" output_config.output_interval
    @printf "  Snapshots per file: %d\n" output_config.max_snapshots_per_file
    @printf "  Velocity: %s, Pressure: %s, Vorticity: %s\n" output_config.save_velocity output_config.save_pressure output_config.save_vorticity
    @printf "  Body forces: %s, Force coefficients: %s\n" output_config.save_body_forces output_config.save_force_coefficients
    @printf "  AMR data: %s\n" output_config.save_refinement_level
    
    return output_config
end

function add_bodies_and_amr!(config, bodies, criteria)
    """
    Add rigid bodies and AMR configuration to simulation.
    """
    
    # Since SimulationConfig is immutable, create new configuration
    new_config = SimulationConfig(
        config.grid_type, config.nx, config.ny, config.nz,
        config.Lx, config.Ly, config.Lz, config.origin,
        config.fluid, config.bc, config.time_scheme, config.dt, config.final_time,
        bodies, nothing, nothing,  # Add rigid bodies, no flexible bodies/controller
        config.use_mpi, true, criteria,  # Enable adaptive refinement
        config.output_config
    )
    
    @printf "Added to simulation:\n"
    @printf "  Rigid bodies: %d\n" bodies.n_bodies
    @printf "  Adaptive refinement: enabled\n"
    
    return new_config
end

function run_cylinder_simulation()
    """
    Main function to run the complete cylinder flow simulation.
    """
    
    println("="^60)
    println("2D Flow Past Cylinder Simulation")
    println("="^60)
    
    # Step 1: Create simulation configuration
    println("\n 1. Creating simulation configuration...")
    config = create_cylinder_simulation()
    
    # Step 2: Create cylinder body
    println("\n 2. Creating cylinder body...")
    cylinder_bodies = create_cylinder_body()
    
    # Step 3: Setup adaptive mesh refinement
    println("\n 3. Setting up adaptive mesh refinement...")
    amr_criteria = setup_adaptive_refinement()
    
    # Step 4: Configure output
    println("\n 4. Configuring output...")
    output_config = setup_output_configuration()
    
    # Step 5: Combine everything
    println("\n 5. Combining configuration...")
    final_config = add_bodies_and_amr!(config, cylinder_bodies, amr_criteria)
    
    # Step 6: Run simulation
    println("\n 6. Starting simulation...")
    println("   This may take several minutes...")
    
    start_time = time()
    
    try
        # Run the simulation
        run_simulation(final_config)
        
        elapsed_time = time() - start_time
        
        println("\nSimulation completed successfully!")
        @printf "Total runtime: %.2f seconds\n" elapsed_time
        
        # Print output file information
        println("\nOutput files generated:")
        println("  cylinder_flow_2d_*.nc - Flow field data")
        println("  cylinder_flow_2d_forces.nc - Body force time series")
        println("  cylinder_flow_2d_amr_*.nc - AMR refinement maps")
        
        println("\nPost-processing suggestions:")
        println("  - Load data with: ncread('cylinder_flow_2d_0001.nc', 'u')")
        println("  - Analyze drag coefficient time series")
        println("  - Visualize vorticity contours")
        println("  - Check AMR refinement patterns")
        
    catch e
        println("\nSimulation failed with error:")
        println(e)
        return false
    end
    
    return true
end

function validate_setup()
    """
    Validate the simulation setup before running.
    """
    
    println("Validating simulation setup...")
    
    # Check Reynolds number
    Re = 100.0
    if Re < 50 || Re > 200
        @warn "Reynolds number $Re may not show expected cylinder wake behavior"
    end
    
    # Check grid resolution
    D = 0.4  # Cylinder diameter
    dx = 8.0 / 160  # Grid spacing
    points_per_diameter = D / dx
    if points_per_diameter < 20
        @warn "Grid resolution may be insufficient: only $points_per_diameter points per diameter"
    else
        @printf "Grid resolution: %.1f points per diameter (good)\n" points_per_diameter
    end
    
    # Check domain size
    cylinder_center = [2.0, 2.0]
    if cylinder_center[1] < 2.0
        @warn "Cylinder too close to inlet - may affect flow development"
    end
    if (8.0 - cylinder_center[1]) < 4.0
        @warn "Cylinder too close to outlet - may affect wake development"
    end
    
    # Check time step (CFL condition)
    U = 1.0
    dt = 0.01
    dx = 8.0 / 160
    CFL = U * dt / dx
    if CFL > 0.5
        @warn "CFL number $CFL may be too high for stability"
    else
        @printf "CFL number: %.3f (stable)\n" CFL
    end
    
    println("Validation complete!")
    return true
end

# Example usage and demonstration
function demonstrate_cylinder_flow()
    """
    Demonstrate the cylinder flow example with validation.
    """
    
    println("BioFlow.jl: 2D Cylinder Flow Example")
    println("This example demonstrates advanced CFD capabilities")
    
    # Validate setup
    if !validate_setup()
        println("Setup validation failed!")
        return false
    end
    
    # Run simulation
    success = run_cylinder_simulation()
    
    if success
        println("\nExample completed successfully!")
        println("Check the output files for results.")
        
        # Provide analysis suggestions
        println("\nSuggested analysis:")
        println("1. Plot drag coefficient vs time to check for periodic vortex shedding")
        println("2. Visualize vorticity contours to see von Kármán vortex street")
        println("3. Check pressure coefficient around cylinder surface")
        println("4. Analyze AMR refinement patterns around the cylinder")
        println("5. Compare results with experimental data (Re=100)")
        
        return true
    else
        println("Example failed! Check error messages above.")
        return false
    end
end

# Run the example if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    demonstrate_cylinder_flow()
end

# Export main functions for use in other scripts
export run_cylinder_simulation, demonstrate_cylinder_flow, create_cylinder_simulation