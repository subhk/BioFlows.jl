"""
Flag Force Coefficients and NetCDF Output Demo

This example demonstrates:
1. Creating flexible flag bodies with various configurations
2. Computing drag and lift coefficients during simulation
3. Saving results to NetCDF files with flexible intervals
4. Advanced output configuration options

Run this example to see how force coefficients are calculated and saved.
"""

using BioFlow
using Printf

function main()
    println("🚩 BioFlow.jl Flag Force Coefficients Demo")
    println("=" ^ 50)
    
    # =================================================================
    # 1. SIMULATION SETUP
    # =================================================================
    
    # Grid parameters (2D XZ plane)
    nx, nz = 200, 100
    Lx, Lz = 4.0, 2.0
    
    # Create uniform grid
    grid = create_uniform_2d_grid(nx, nz, Lx, Lz)
    
    # Fluid properties 
    Reynolds = 200.0
    inlet_velocity = 1.0
    ρ = ConstantDensity(1.0)  
    μ = inlet_velocity * 1.0 / Reynolds  # Using unit length scale
    fluid = FluidProperties(μ, ρ, Reynolds)
    
    # Boundary conditions (inlet at left, outlet at right, walls at top/bottom)
    bc = BoundaryConditions2D(
        left = InletBC(:x, :left, inlet_velocity),  # Inlet velocity 
        right = PressureOutletBC(:x, :right, 0.0),  # Zero pressure outlet
        bottom = NoSlipBC(:z, :bottom),              # No-slip wall
        top = NoSlipBC(:z, :top)                     # No-slip wall
    )
    
    # =================================================================
    # 2. CREATE FLEXIBLE FLAGS WITH DIFFERENT CONFIGURATIONS
    # =================================================================
    
    println("\n🎌 Creating flexible flags...")
    
    # Flag 1: Simple horizontal flag
    flag1 = create_flag([0.5, 1.0], 0.8, 0.05; 
                       material=:flexible,
                       attachment=:fixed_leading_edge,
                       n_points=25)
    
    # Flag 2: Flag with initial angle
    flag2 = create_flag([0.5, 0.4], 0.6, 0.03;
                       initial_angle=π/6,  # 30° initial angle
                       material=:very_flexible,
                       attachment=:fixed_leading_edge,
                       n_points=20)
    
    # Flag 3: Flag with sinusoidal prescribed motion
    flag3 = create_flag([2.0, 1.5], 0.4, 0.02;
                       attachment=:pinned_leading_edge,
                       prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=2.0),
                       material=:stiff,
                       n_points=15)
    
    # Create collection
    flags = FlexibleBodyCollection()
    add_flexible_body!(flags, flag1)
    add_flexible_body!(flags, flag2) 
    add_flexible_body!(flags, flag3)
    
    println("✓ Created $(flags.n_bodies) flexible flags")
    for (i, flag) in enumerate(flags.bodies)
        println("  Flag $i: Length=$(flag.length), Points=$(flag.n_points), Material=$(flag.bending_rigidity)")
    end
    
    # =================================================================
    # 3. NETCDF OUTPUT CONFIGURATION
    # =================================================================
    
    println("\n💾 Configuring NetCDF output...")
    
    # Output configuration with flexible save options
    output_config = NetCDFConfig("flag_simulation_results";
        max_snapshots_per_file = 200,           # Max snapshots per file
        save_mode = :both,                      # Save on time AND iteration intervals
        time_interval = 0.05,                   # Save every 0.05 time units
        iteration_interval = 25,                # Save every 25 iterations
        save_flow_field = true,                 # Save u, w, p fields
        save_body_positions = true,             # Save flag positions
        save_force_coefficients = true,         # Save Cd, Cl coefficients
        reference_velocity = inlet_velocity,    # Reference for coefficients
        flow_direction = [1.0, 0.0]            # Main flow direction (x-axis)
    )
    
    # Create NetCDF writer
    writer = NetCDFWriter("output/flag_demo_results.nc", grid, output_config)
    
    println("✓ NetCDF output configured:")
    println("  • Save mode: $(output_config.save_mode)")
    println("  • Time interval: $(output_config.time_interval)")
    println("  • Iteration interval: $(output_config.iteration_interval)")
    println("  • Reference velocity: $(output_config.reference_velocity)")
    
    # =================================================================
    # 4. SIMULATION PARAMETERS
    # =================================================================
    
    # Time stepping 
    dt = 0.001
    final_time = 2.0
    max_iterations = Int(final_time / dt)
    
    # Initialize solution state
    state = SolutionState2D(nx, nz)
    
    # Set initial conditions (uniform flow)
    state.u .= inlet_velocity
    state.w .= 0.0
    state.p .= 0.0
    
    println("\n⏱️  Simulation parameters:")
    println("  • Time step: $dt")
    println("  • Final time: $final_time")
    println("  • Max iterations: $max_iterations")
    
    # =================================================================
    # 5. DEMONSTRATION OF FORCE COEFFICIENT CALCULATION
    # =================================================================
    
    println("\n🔬 Computing force coefficients (demonstration)...")
    
    # Compute coefficients for all flags (example at t=0)
    for (i, flag) in enumerate(flags.bodies)
        coeffs = compute_drag_lift_coefficients(flag, grid, state, fluid;
                                               reference_velocity=inlet_velocity,
                                               reference_length=flag.length,
                                               flow_direction=[1.0, 0.0])
        
        println("\n  Flag $i coefficients (t=0.0):")
        @printf "    Drag coefficient (Cd):     %.4f\n" coeffs.Cd
        @printf "    Lift coefficient (Cl):     %.4f\n" coeffs.Cl  
        @printf "    Pressure drag (Cd_p):      %.4f\n" coeffs.Cd_pressure
        @printf "    Viscous drag (Cd_v):       %.4f\n" coeffs.Cd_viscous
        @printf "    Total force: Fx=%.4f, Fz=%.4f\n" coeffs.Fx coeffs.Fz
        @printf "    Center of pressure: [%.3f, %.3f]\n" coeffs.center_of_pressure[1] coeffs.center_of_pressure[2]
        
        # Compute instantaneous power
        power = compute_instantaneous_power(flag, grid, state, fluid)
        @printf "    Instantaneous power:       %.6f W\n" power
    end
    
    # =================================================================
    # 6. SIMULATION LOOP WITH NETCDF OUTPUT
    # =================================================================
    
    println("\n🚀 Starting simulation with NetCDF output...")
    println("   (This is a demonstration - actual solver integration needed)")
    
    current_time = 0.0
    iteration = 0
    
    # Save initial condition
    save_complete_snapshot!(writer, state, flags, grid, fluid, current_time, iteration;
                           reference_velocity=inlet_velocity,
                           flow_direction=[1.0, 0.0])
    
    # Simulation loop (simplified demonstration)
    while current_time < final_time && iteration < max_iterations
        iteration += 1
        current_time = iteration * dt
        
        # *** HERE: Actual Navier-Stokes solve would happen ***
        # solve_step_2d!(state, grid, fluid, bc, flags, dt)
        
        # *** HERE: Update flexible body positions ***
        # for flag in flags.bodies
        #     update_flexible_body!(flag, dt)
        #     apply_boundary_conditions!(flag, current_time)
        # end
        
        # Save snapshots according to configured intervals
        if should_save(writer, current_time, iteration)
            save_complete_snapshot!(writer, state, flags, grid, fluid, 
                                  current_time, iteration;
                                  reference_velocity=inlet_velocity,
                                  flow_direction=[1.0, 0.0])
                                  
            # Print progress
            if iteration % 100 == 0
                @printf "  Time: %.3f, Iteration: %d\n" current_time iteration
            end
        end
        
        # Demonstrate coefficient calculation every 50 iterations
        if iteration % 50 == 0
            coeffs = compute_drag_lift_coefficients(flags.bodies[1], grid, state, fluid;
                                                   reference_velocity=inlet_velocity)
            @printf "  t=%.3f: Flag 1 Cd=%.4f, Cl=%.4f\n" current_time coeffs.Cd coeffs.Cl
        end
    end
    
    # Close NetCDF file
    close_netcdf!(writer)
    
    # =================================================================
    # 7. SUMMARY AND ANALYSIS TIPS
    # =================================================================
    
    println("\n📊 Simulation completed!")
    println("   NetCDF file saved: flag_demo_results.nc")
    println("\n📈 Analysis tips:")
    println("   • Load data in Python: xr.open_dataset('flag_demo_results.nc')")
    println("   • View Cd/Cl time series: data['drag_coefficient'][:, :]")
    println("   • Visualize flag motion: data['flexible_body_1_x'][:, :]")
    println("   • Plot center of pressure: data['center_of_pressure_x'][:, :]")
    
    println("\n🎯 NetCDF Variables saved:")
    println("   Flow field: u, w, p")
    println("   Flag positions: flexible_body_*_x, flexible_body_*_z") 
    println("   Force coefficients: drag_coefficient, lift_coefficient")
    println("   Detailed forces: drag_coefficient_pressure, drag_coefficient_viscous")
    println("   Other: force_x, force_z, center_of_pressure_*, instantaneous_power")
    
    return writer, flags, grid, fluid
end

# =================================================================
# UTILITY FUNCTIONS FOR ADVANCED ANALYSIS
# =================================================================

"""
    analyze_force_coefficients(netcdf_file::String)

Load and analyze force coefficient data from NetCDF file.
"""
function analyze_force_coefficients(netcdf_file::String)
    println("\n🔍 Analyzing force coefficients from: $netcdf_file")
    
    # This would use NetCDF.jl to load and analyze data
    # data = read_netcdf_data(netcdf_file)
    # 
    # # Compute statistics
    # cd_mean = mean(data["drag_coefficient"], dims=2)
    # cd_rms = sqrt(mean(data["drag_coefficient"].^2, dims=2))
    # cl_rms = sqrt(mean(data["lift_coefficient"].^2, dims=2))
    #
    # println("Flag force statistics:")
    # for i = 1:size(cd_mean, 1)
    #     @printf "  Flag %d: Cd_mean=%.4f, Cd_rms=%.4f, Cl_rms=%.4f\n" i cd_mean[i] cd_rms[i] cl_rms[i]
    # end
    
    println("   (Analysis implementation would load and process NetCDF data)")
end

"""
    create_coefficient_time_series_plot(netcdf_file::String, flag_id::Int)

Create time series plots of force coefficients (requires plotting package).
"""
function create_coefficient_time_series_plot(netcdf_file::String, flag_id::Int)
    println("   Creating time series plot for flag $flag_id")
    println("   (Would generate Cd(t) and Cl(t) plots using Plots.jl or similar)")
end

# =================================================================
# RUN THE DEMO
# =================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Create output directory
    mkpath("output")
    
    # Run the main demo
    writer, flags, grid, fluid = main()
    
    println("\n🎉 Demo completed successfully!")
    println("   Check the 'output/' directory for NetCDF results")
end