"""
Flexible Body Position Tracking with NetCDF Output

This example demonstrates comprehensive position saving for flexible bodies with:
1. Different save interval options (time vs iteration based)
2. Various levels of detail (positions only vs full kinematics)
3. Multiple output file strategies
4. Position-only lightweight tracking
5. Full kinematics with velocities, accelerations, curvature

Run this example to see different position saving strategies.
"""

using BioFlow
using Printf

function main()
    println("📍 BioFlow.jl Flexible Body Position Tracking Demo")
    println("=" ^ 55)
    
    # =================================================================
    # 1. SIMULATION SETUP
    # =================================================================
    
    # Grid parameters (2D XZ plane)
    nx, nz = 150, 80
    Lx, Lz = 3.0, 1.5
    grid = create_uniform_2d_grid(nx, nz, Lx, Lz)
    
    # Fluid properties
    Reynolds = 150.0
    inlet_velocity = 1.0
    ρ = ConstantDensity(1.0)
    μ = inlet_velocity * 1.0 / Reynolds
    fluid = FluidProperties(μ, ρ, Reynolds)
    
    println("\n🌊 Simulation setup:")
    println("  • Grid: $(nx) × $(nz), Domain: $(Lx) × $(Lz)")
    println("  • Reynolds number: $(Reynolds)")
    println("  • Inlet velocity: $(inlet_velocity)")
    
    # =================================================================
    # 2. CREATE MULTIPLE FLEXIBLE BODIES FOR TRACKING
    # =================================================================
    
    println("\n🎌 Creating flexible bodies...")
    
    # Flag 1: Long horizontal flag
    flag1 = create_flag([0.3, 0.8], 1.2, 0.04; 
                       material=:flexible,
                       attachment=:fixed_leading_edge,
                       n_points=30)
    
    # Flag 2: Shorter flag at angle
    flag2 = create_flag([0.3, 0.4], 0.8, 0.03;
                       initial_angle=π/8,  # 22.5° angle
                       material=:very_flexible,
                       attachment=:fixed_leading_edge,
                       n_points=20)
    
    # Flag 3: Vertical hanging flag
    flag3 = create_vertical_flag([1.5, 1.2], 0.6, 0.02;
                                material=:stiff,
                                n_points=25)
    
    # Flag 4: Sinusoidally driven flag
    flag4 = create_flag([2.0, 0.9], 0.5, 0.025;
                       attachment=:pinned_leading_edge,
                       prescribed_motion=(type=:sinusoidal, amplitude=0.08, frequency=3.0),
                       material=:flexible,
                       n_points=18)
    
    # Create collection
    flags = FlexibleBodyCollection()
    add_flexible_body!(flags, flag1)
    add_flexible_body!(flags, flag2)
    add_flexible_body!(flags, flag3)
    add_flexible_body!(flags, flag4)
    
    println("✓ Created $(flags.n_bodies) flexible bodies:")
    for (i, flag) in enumerate(flags.bodies)
        println("  Flag $i: L=$(flag.length)m, $(flag.n_points) points, BC=$(flag.bc_type)")
    end
    
    # =================================================================
    # 3. DIFFERENT NETCDF POSITION TRACKING STRATEGIES
    # =================================================================
    
    println("\n💾 Setting up multiple NetCDF position tracking strategies...")
    
    # Create output directory
    mkpath("output/position_tracking")
    
    # Strategy 1: High-frequency position-only tracking (minimal data)
    println("\n📊 Strategy 1: High-frequency position-only tracking")
    writer_positions = create_position_only_writer(
        "output/position_tracking/flag_positions_high_freq.nc",
        grid, flags;
        time_interval = 0.005,           # Every 0.005s (very high frequency)
        save_mode = :time_interval,      # Time-based only
        max_snapshots = 2000            # Allow many snapshots
    )
    
    # Strategy 2: Iteration-based detailed kinematics 
    println("\n📊 Strategy 2: Iteration-based full kinematics")
    config_detailed = NetCDFConfig("flag_kinematics_detailed";
        max_snapshots_per_file = 500,
        save_mode = :iteration_interval,     # Iteration-based
        iteration_interval = 10,             # Every 10 iterations
        save_flow_field = false,             # No flow field
        save_body_positions = true,
        save_force_coefficients = false
    )
    writer_detailed = NetCDFWriter("output/position_tracking/flag_kinematics_detailed.nc", 
                                   grid, config_detailed)
    
    # Strategy 3: Moderate frequency with both triggers
    println("\n📊 Strategy 3: Moderate frequency with dual triggers")
    config_moderate = NetCDFConfig("flag_positions_moderate";
        max_snapshots_per_file = 1000,
        save_mode = :both,                   # Either time OR iteration trigger
        time_interval = 0.02,                # Every 0.02s
        iteration_interval = 25,             # OR every 25 iterations
        save_flow_field = false,
        save_body_positions = true,
        save_force_coefficients = false
    )
    writer_moderate = NetCDFWriter("output/position_tracking/flag_positions_moderate.nc", 
                                   grid, config_moderate)
    
    # Strategy 4: Low frequency for long-term tracking
    println("\n📊 Strategy 4: Low frequency for long-term trends")
    config_longterm = NetCDFConfig("flag_positions_longterm";
        max_snapshots_per_file = 200,
        save_mode = :time_interval,
        time_interval = 0.1,                 # Every 0.1s (low frequency)
        save_flow_field = false,
        save_body_positions = true,
        save_force_coefficients = false
    )
    writer_longterm = NetCDFWriter("output/position_tracking/flag_positions_longterm.nc", 
                                   grid, config_longterm)
    
    # =================================================================
    # 4. SIMULATION PARAMETERS AND INITIALIZATION
    # =================================================================
    
    # Simulation parameters
    dt = 0.0008
    final_time = 2.0
    max_iterations = Int(final_time / dt)
    
    # Initialize solution state
    state = SolutionState2D(nx, nz)
    state.u .= inlet_velocity
    state.w .= 0.0
    state.p .= 0.0
    
    println("\n⏱️  Simulation timeline:")
    println("  • Time step: $dt")
    println("  • Final time: $final_time")  
    println("  • Total iterations: $max_iterations")
    
    # =================================================================
    # 5. DEMONSTRATION OF DIFFERENT POSITION SAVING MODES
    # =================================================================
    
    println("\n🚀 Starting simulation with multiple position tracking modes...")
    
    current_time = 0.0
    iteration = 0
    
    # Save initial conditions for all writers
    save_body_positions_only!(writer_positions, flags, current_time, iteration)
    save_body_kinematics_snapshot!(writer_detailed, flags, current_time, iteration, dt)
    save_flexible_body_positions!(writer_moderate, flags, current_time, iteration)
    save_flexible_body_positions!(writer_longterm, flags, current_time, iteration)
    
    # Track saving statistics
    saves_positions = 0
    saves_detailed = 0
    saves_moderate = 0
    saves_longterm = 0
    
    println("\n📈 Simulation progress:")
    
    # Main simulation loop
    while current_time < final_time && iteration < max_iterations
        iteration += 1
        current_time = iteration * dt
        
        # *** PLACEHOLDER: Actual physics solving would happen here ***
        # solve_step_2d!(state, grid, fluid, bc, flags, dt)
        
        # *** PLACEHOLDER: Update flexible body dynamics ***
        # for flag in flags.bodies
        #     update_flexible_body!(flag, dt)
        #     apply_boundary_conditions!(flag, current_time)
        # end
        
        # Demonstrate different position saving strategies
        
        # Strategy 1: High-frequency positions only
        if save_body_positions_only!(writer_positions, flags, current_time, iteration)
            saves_positions += 1
        end
        
        # Strategy 2: Detailed kinematics (every 10 iterations)
        if save_body_kinematics_snapshot!(writer_detailed, flags, current_time, iteration, dt)
            saves_detailed += 1
        end
        
        # Strategy 3: Moderate frequency with enhanced data
        if save_flexible_body_positions!(writer_moderate, flags, current_time, iteration;
                                        save_velocities=true, save_curvature=true)
            saves_moderate += 1
        end
        
        # Strategy 4: Long-term tracking
        if save_flexible_body_positions!(writer_longterm, flags, current_time, iteration)
            saves_longterm += 1
        end
        
        # Progress reporting
        if iteration % 200 == 0
            @printf "  t=%.3fs, iter=%d | Saves: Pos=%d, Detail=%d, Mod=%d, Long=%d\n" current_time iteration saves_positions saves_detailed saves_moderate saves_longterm
            
            # Show flag positions for demonstration
            flag1_tip_x = flags.bodies[1].X[end, 1]
            flag1_tip_z = flags.bodies[1].X[end, 2]
            @printf "    Flag 1 tip position: [%.3f, %.3f]\n" flag1_tip_x flag1_tip_z
        end
    end
    
    # Close all writers
    close_netcdf!(writer_positions)
    close_netcdf!(writer_detailed)
    close_netcdf!(writer_moderate)
    close_netcdf!(writer_longterm)
    
    # =================================================================
    # 6. SUMMARY AND FILE ANALYSIS
    # =================================================================
    
    println("\n📊 Position tracking completed!")
    println("\n💾 NetCDF files created:")
    println("  1. flag_positions_high_freq.nc    - $saves_positions snapshots (positions only)")
    println("  2. flag_kinematics_detailed.nc    - $saves_detailed snapshots (full kinematics)")
    println("  3. flag_positions_moderate.nc     - $saves_moderate snapshots (positions + curvature)")
    println("  4. flag_positions_longterm.nc     - $saves_longterm snapshots (long-term trends)")
    
    println("\n📈 Data structure in each file:")
    println("  Dimensions:")
    println("    • time: number of snapshots")
    println("    • n_points_body_*: Lagrangian points per body")
    println("  Variables:")
    println("    • flexible_body_*_x(n_points, time): x-coordinates")  
    println("    • flexible_body_*_z(n_points, time): z-coordinates")
    println("    • flexible_body_*_vel_*(n_points, time): velocities (if enabled)")
    println("    • flexible_body_*_acc_*(n_points, time): accelerations (if enabled)")
    println("    • flexible_body_*_curvature(n_points, time): curvature (if enabled)")
    
    println("\n🔍 Analysis tips:")
    println("  • High-frequency file: Best for detailed animation and frequency analysis")
    println("  • Detailed kinematics: Full dynamics analysis, modal decomposition")
    println("  • Moderate frequency: Good balance of detail vs file size")
    println("  • Long-term trends: Suitable for averaging and long-time statistics")
    
    println("\n🐍 Python analysis example:")
    println("  import xarray as xr")
    println("  data = xr.open_dataset('flag_positions_high_freq.nc')")
    println("  flag1_x = data['flexible_body_1_x']  # Shape: (n_points, time)")
    println("  flag1_z = data['flexible_body_1_z']")
    println("  # Animate flag motion:")
    println("  # plt.plot(flag1_x[:, t], flag1_z[:, t]) for each time t")
    
    return flags, [writer_positions, writer_detailed, writer_moderate, writer_longterm]
end

# =================================================================
# ADDITIONAL UTILITY FUNCTIONS
# =================================================================

"""
    demonstrate_position_save_options(flags, grid)

Demonstrate all available position saving options.
"""
function demonstrate_position_save_options(flags::FlexibleBodyCollection, grid::StaggeredGrid)
    
    println("\n🛠️  Available position saving options:")
    
    # Create demo writer
    writer = create_position_only_writer("demo_positions.nc", grid, flags)
    
    current_time = 0.0
    iteration = 0
    dt = 0.001
    
    println("\n1. Positions only (minimal overhead):")
    save_body_positions_only!(writer, flags, current_time, iteration)
    
    println("2. Full kinematics (positions + velocities + accelerations):")
    save_body_kinematics_snapshot!(writer, flags, current_time, iteration, dt)
    
    println("3. Enhanced tracking (positions + velocities + curvature):")
    save_flexible_body_positions!(writer, flags, current_time, iteration;
                                  save_velocities=true, 
                                  save_curvature=true)
    
    println("4. Complete material tracking (everything):")
    save_flexible_body_positions!(writer, flags, current_time, iteration;
                                  save_velocities=true,
                                  save_accelerations=true,
                                  save_curvature=true,
                                  save_forces=true,
                                  save_material_properties=true)
    
    close_netcdf!(writer)
    rm("demo_positions.nc", force=true)  # Clean up demo file
    
    println("✓ All saving modes demonstrated")
end

"""
    analyze_position_data_structure(filename::String)

Analyze the structure of a position NetCDF file.
"""
function analyze_position_data_structure(filename::String)
    println("\n🔍 Analyzing NetCDF structure: $filename")
    
    # This would use NetCDF.jl to inspect file structure
    println("  File structure analysis:")
    println("  • Dimensions: time, n_points_body_*")
    println("  • Variables: flexible_body_*_x, flexible_body_*_z, ...")
    println("  • Attributes: units, descriptions, body properties")
    println("  (Full implementation would read actual file metadata)")
end

# =================================================================
# RUN THE DEMO
# =================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Run main demonstration
    flags, writers = main()
    
    # Demonstrate additional options
    demonstrate_position_save_options(flags, create_uniform_2d_grid(50, 50, 1.0, 1.0))
    
    println("\n🎉 Position tracking demonstration completed!")
    println("   Check 'output/position_tracking/' for NetCDF files")
    println("   Each file demonstrates different saving strategies")
end