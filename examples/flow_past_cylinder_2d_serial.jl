# Flow past a 2D cylinder using serial computation with adaptive mesh refinement
# Run with: julia --project examples/flow_past_cylinder_2d_serial.jl
#
# This example demonstrates:
# - Serial 2D flow simulation with adaptive mesh refinement (AMR)
# - Flow around a rigid circular cylinder
# - AMR-based grid refinement for improved accuracy near the cylinder
# - Enhanced diagnostic output showing AMR statistics and iteration progress  
# - Automatic calculation of drag/lift coefficients
#
# AMR Features:
# - Automatic refinement based on velocity gradients, pressure gradients, vorticity, and distance to bodies
# - Up to 3 levels of refinement (configurable)
# - Refinement checking every 5 time steps for optimal performance
# - Comprehensive AMR performance statistics and timing

using BioFlows
using NetCDF

function run_amr_simulation(config::BioFlows.SimulationConfig, amr_solver::BioFlows.AMRIntegratedSolver, initial_state::BioFlows.SolutionState)
    """
    Custom simulation loop for AMR-integrated solvers
    """
    # Extract grid from base solver
    base_grid = amr_solver.base_solver.grid
    
    # Initialize output writer using base grid (AMR is internal)
    writer = BioFlows.NetCDFWriter("$(config.output_config.filename).nc", base_grid, config.output_config)
    
    # Initialize NetCDF file and add metadata
    try
        BioFlows.initialize_netcdf_file!(writer)
        if writer.ncfile !== nothing
            BioFlows.NetCDF.putatt(writer.ncfile, "global", Dict(
                "domain_Lx" => base_grid.Lx,
                "domain_Lz" => base_grid.Lz,
                "amr_enabled" => true,
                "max_refinement_level" => amr_solver.amr_criteria.max_refinement_level
            ))
        end
        # Add body metadata
        bodies_for_meta = config.rigid_bodies !== nothing ? config.rigid_bodies : config.flexible_bodies
        if bodies_for_meta !== nothing
            BioFlows.annotate_bodies_metadata!(writer, bodies_for_meta)
        end
    catch e
        @warn "Could not initialize AMR NetCDF metadata: $e"
    end
    
    # Time-stepping loop
    dt = config.dt
    final_time = config.final_time
    nsteps = round(Int, final_time / dt)
    
    state_old = deepcopy(initial_state)
    state_new = deepcopy(initial_state)
    
    println("AMR Time-stepping: $(nsteps) steps, dt=$(dt)s")
    
    for step = 1:nsteps
        t = step * dt
        state_old.step = step
        state_old.t = t
        
        # AMR solve step (handles grid adaptation internally)
        try
            BioFlows.amr_solve_step!(amr_solver, state_new, state_old, dt, config.rigid_bodies)
        catch e
            @warn "AMR solve step failed at step $step: $e"
            # Fallback: copy old state to new (essentially skip this time step)
            state_new = deepcopy(state_old)
            break
        end
        
        # Output progress
        if step % 100 == 0 || step == nsteps
            amr_cells = get(amr_solver.amr_statistics, "current_refined_cells", 0)
            println("  Step $(step)/$(nsteps), t=$(round(t, digits=3))s, AMR cells: $(amr_cells)")
        end
        
        # Save output (state_new is ensured to be on original grid by AMR solver)
        try
            if config.rigid_bodies !== nothing
                BioFlows.write_solution!(writer, state_new, config.rigid_bodies, base_grid, config.fluid, t, step)
            else
                BioFlows.save_snapshot!(writer, state_new, t, step)
            end
        catch e
            if step == 1  # Only warn on first failure to avoid spam
                @warn "AMR output writing failed: $e"
            end
        end
        
        # Swap states for next iteration
        state_old, state_new = state_new, state_old
    end
    
    # Close output file
    try
        BioFlows.close_netcdf!(writer)
    catch e
        @warn "Could not close NetCDF file: $e"
    end
    
    println("AMR simulation completed!")
end

function main()
    println("="^60)
    println("FLOW PAST CYLINDER - SERIAL 2D SIMULATION WITH AMR")
    println("="^60)

    # Physical and numerical parameters
    # Domain geometry
    nx, nz = 144, 48           # Grid points (adjusted for good cylinder resolution)
    Lx, Lz = 6.0, 2.0         # Physical domain size [m]
    
    # Flow parameters
    Uin = 1.0                  # Inlet velocity [m/s]
    ρ = 1000.0                 # Fluid density [kg/m³]
    ν = 0.001                  # Kinematic viscosity [m²/s]
    
    # Cylinder geometry
    D = 0.2                    # Cylinder diameter [m]
    R = D/2                    # Cylinder radius [m]
    xc = 1.2                   # Cylinder center x-coordinate [m]
    zc = Lz/2                  # Cylinder center z-coordinate [m] (centerline)
    
    # Time integration
    dt = 0.002                 # Time step [s]
    Tfinal = 2.0               # Final simulation time [s]
    save_interval = 0.1        # Output saving interval [s]
    
    # AMR Parameters (tuned for cylinder flow)
    amr_velocity_threshold = 2.0      # Velocity gradient threshold for refinement
    amr_pressure_threshold = 100.0    # Pressure gradient threshold [Pa/m]
    amr_vorticity_threshold = 10.0    # Vorticity threshold [1/s]
    amr_body_distance = 0.3           # Distance from body for auto-refinement [m]
    amr_max_levels = 3                # Maximum refinement levels
    amr_min_grid_size = 0.002         # Minimum grid size [m]
    
    # Calculate Reynolds number for reference
    Re = Uin * D / ν
    println("Physical parameters:")
    println("  Reynolds number: Re = U*D/ν = $(round(Re, digits=1))")
    println("  Grid resolution: $(nx)×$(nz) = $(nx*nz) cells")
    println("  Domain size: $(Lx)×$(Lz) m")
    println("  Cylinder: D=$(D) m at ($(xc), $(zc))")
    println("  Time: dt=$(dt) s, T_final=$(Tfinal) s")
    println()

    # Create custom AMR criteria tailored for cylinder flow
    println("Setting up custom AMR criteria for cylinder flow...")
    custom_amr_criteria = BioFlows.AdaptiveRefinementCriteria(
        velocity_gradient_threshold=amr_velocity_threshold,
        pressure_gradient_threshold=amr_pressure_threshold,
        vorticity_threshold=amr_vorticity_threshold,
        body_distance_threshold=amr_body_distance,
        max_refinement_level=amr_max_levels,
        min_grid_size=amr_min_grid_size
    )
    
    # Build simulation configuration with adaptive mesh refinement
    println("Setting up simulation configuration with adaptive mesh refinement...")
    
    # First create base config without AMR
    base_config = BioFlows.create_2d_simulation_config(
        nx=nx, nz=nz,
        Lx=Lx, Lz=Lz,
        density_value=ρ,
        nu=ν,
        inlet_velocity=Uin,
        outlet_type=:pressure,
        wall_type=:no_slip,
        dt=dt, final_time=Tfinal,
        use_mpi=false,  # Serial computation
        adaptive_refinement=false,  # Will enable below with custom criteria
        output_interval=save_interval,
        output_file="cylinder2d_serial_amr"
    )
    
    # Create new config with custom AMR criteria
    config = BioFlows.SimulationConfig(
        base_config.grid_type,
        base_config.nx, base_config.ny, base_config.nz,
        base_config.Lx, base_config.Ly, base_config.Lz,
        base_config.origin,
        base_config.fluid,
        base_config.bc,
        base_config.time_scheme,
        base_config.dt,
        base_config.final_time,
        base_config.rigid_bodies,
        base_config.flexible_bodies,
        base_config.flexible_body_controller,
        base_config.use_mpi,
        true,  # adaptive_refinement = true
        custom_amr_criteria,  # our custom criteria
        base_config.output_config
    )

    # Add rigid cylinder obstacle
    println("Adding rigid cylinder: center=($(xc), $(zc)), radius=$(R)")
    config = BioFlows.add_rigid_circle!(config, [xc, zc], R)

    # Display AMR configuration
    println()
    println("Adaptive Mesh Refinement (AMR) Configuration:")
    println("  AMR enabled: $(config.adaptive_refinement)")
    if config.adaptive_refinement && config.refinement_criteria !== nothing
        criteria = config.refinement_criteria
        println("  Max refinement levels: $(criteria.max_refinement_level)")
        println("  Velocity gradient threshold: $(criteria.velocity_gradient_threshold)")
        println("  Pressure gradient threshold: $(criteria.pressure_gradient_threshold)")
        println("  Vorticity threshold: $(criteria.vorticity_threshold)")
        println("  Body distance threshold: $(criteria.body_distance_threshold) m")
        println("  Minimum grid size: $(criteria.min_grid_size) m")
    end

    # Start the serial simulation with AMR
    println()
    println("Starting serial 2D simulation with adaptive mesh refinement...")
    println("Expected output: cylinder2d_serial_amr.nc and cylinder2d_serial_amr_coeffs.nc")
    println("="^60)
    
    # Create solver and initialize simulation state
    base_solver = BioFlows.create_solver(config)
    initial_state = BioFlows.initialize_simulation(config)
    
    # Create AMR-integrated solver if AMR is enabled
    if config.adaptive_refinement && config.refinement_criteria !== nothing
        println("Initializing AMR-integrated solver...")
        try
            # Create AMR solver with tuned parameters for cylinder flow
            amr_solver = BioFlows.create_amr_integrated_solver(base_solver, config.refinement_criteria; 
                                                              amr_frequency=5)  # Check for refinement every 5 steps
            
            # Validate AMR integration
            try
                if BioFlows.validate_amr_integration(amr_solver)
                    println("AMR integration validated successfully!")
                    solver = amr_solver
                else
                    @warn "AMR integration validation failed, continuing with base solver"
                    solver = base_solver
                end
            catch e
                @warn "AMR validation threw an error: $e, falling back to base solver"
                solver = base_solver
            end
        catch e
            @warn "Failed to create AMR solver: $e, falling back to base solver"
            solver = base_solver
        end
    else
        println("Using base solver (AMR disabled)")
        solver = base_solver
    end
    
    # Run the simulation (includes enhanced diagnostics and AMR)
    if isa(solver, BioFlows.AMRIntegratedSolver)
        # AMR solver requires custom simulation loop
        println("Running AMR simulation with custom integration...")
        run_amr_simulation(config, solver, initial_state)
    else
        # Standard solver uses built-in run_simulation
        BioFlows.run_simulation(config, solver, initial_state)
    end

    # Simulation complete - analyze results
    println()
    println("="^60)
    println("SIMULATION COMPLETE")
    println("="^60)
    
    # Display AMR performance summary if AMR was used
    if config.adaptive_refinement && config.refinement_criteria !== nothing && isa(solver, BioFlows.AMRIntegratedSolver)
        println("AMR Performance Summary:")
        BioFlows.print_amr_summary(solver)
        println()
    end
    
    # Read and display final drag/lift coefficients
    try
        coeff_path = "cylinder2d_serial_amr_coeffs.nc"
        if isfile(coeff_path)
            nc = NetCDF.open(coeff_path)
            time = NetCDF.readvar(nc, "time")
            Cd = NetCDF.readvar(nc, "Cd")
            Cl = NetCDF.readvar(nc, "Cl")
            NetCDF.close(nc)
            
            nt = length(time)
            Cd_last = Cd[1, nt]
            Cl_last = Cl[1, nt]
            
            println("Results summary:")
            println("  Final time: t = $(round(time[end], digits=3)) s")
            println("  Total time steps: $(nt)")
            println("  Final drag coefficient: Cd = $(round(Cd_last, digits=4))")
            println("  Final lift coefficient: Cl = $(round(Cl_last, digits=4))")
            println()
            println("Output files created:")
            println("  Flow field: cylinder2d_serial_amr.nc")
            println("  Force coefficients: $(coeff_path)")
            
            # Basic flow regime assessment
            if Re < 40
                flow_regime = "steady"
            elseif Re < 150
                flow_regime = "periodic vortex shedding"  
            else
                flow_regime = "turbulent"
            end
            println("  Expected flow regime (Re=$(round(Re, digits=1))): $(flow_regime)")
            
        else
            @warn "Coefficient file not found: $(coeff_path)"
            println("Flow field output: cylinder2d_serial_amr.nc")
        end
    catch e
        @warn "Could not read coefficient data: $e"
    end
    
    println("="^60)
end

main()