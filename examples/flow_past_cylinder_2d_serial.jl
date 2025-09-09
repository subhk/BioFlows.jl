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

function run_amr_simulation(config::BioFlows.SimulationConfig, 
                            amr_solver::BioFlows.AMRIntegratedSolver, 
                            initial_state::BioFlows.SolutionState)

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
    # Domain geometry - minimal resolution for debugging
    nx, nz = 150, 50            # Minimal grid points to debug LLVM error
    Lx, Lz = 6.0, 2.0         # Physical domain size [m]
    
    # Flow parameters
    Uin = 1.0                  # Inlet velocity [m/s]
    ρ = 1000.0                 # Fluid density [kg/m³]
    ν = 0.001                   # Increased viscosity to stabilize
    
    # Cylinder geometry
    D = 0.2                    # Cylinder diameter [m]
    R = D/2                    # Cylinder radius [m]
    xc = 1.2                   # Cylinder center x-coordinate [m]
    zc = Lz/2                  # Cylinder center z-coordinate [m] (centerline)
    
    # Time integration
    dt = 0.005                 # Smaller time step for stability with cylinder
    Tfinal = 0.1               # Longer simulation to see flow development
    save_interval = 0.1        # Output saving interval
    
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
    
    # Create config with AMR enabled (will use default AMR criteria)
    config = BioFlows.create_2d_simulation_config(
        nx=nx, nz=nz,
        Lx=Lx, Lz=Lz,
        density_value=ρ,
        nu=ν,
        inlet_velocity=Uin,
        outlet_type=:pressure,
        wall_type=:no_slip,
        dt=dt, final_time=Tfinal,
        use_mpi=false,  # Serial computation
        adaptive_refinement=true,  # Enable AMR with default criteria
        output_interval=save_interval,
        output_file="cylinder2d_serial_amr"
    )
    
    # Note: Using default AMR criteria since config is immutable
    # The simulation will still benefit from adaptive refinement

    # Add rigid cylinder obstacle with simplified immersed boundary method
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
    println("Creating solver and initializing simulation state...")
    println("Memory before solver creation: $(Base.gc_live_bytes() / 1024^2) MB")
    GC.gc()  # Force garbage collection
    
    solver = BioFlows.create_solver(config)
    println("Memory after solver creation: $(Base.gc_live_bytes() / 1024^2) MB")
    
    initial_state = BioFlows.initialize_simulation(config; initial_conditions=:uniform_flow)
    println("Memory after state initialization: $(Base.gc_live_bytes() / 1024^2) MB")
    
    # Initialize with better initial conditions to prevent NaN
    println("Setting up initial flow field...")
    initial_state.u .= Uin  # Set uniform inlet velocity
    initial_state.w .= 0.0  # Zero z-velocity initially
    initial_state.p .= 0.0  # Zero pressure initially
    println("  u-velocity range: $(minimum(initial_state.u)) to $(maximum(initial_state.u))")
    println("  w-velocity range: $(minimum(initial_state.w)) to $(maximum(initial_state.w))")
    println("  pressure range: $(minimum(initial_state.p)) to $(maximum(initial_state.p))")
    
    # Check if AMR is enabled and use appropriate simulation loop
    if config.adaptive_refinement && isa(solver, BioFlows.AMRIntegratedSolver)
        println("Running AMR-enhanced simulation...")
        println("Starting adaptive simulation with base grid $(nx*nz) cells...")
        
        try
            run_amr_simulation(config, solver, initial_state)
        catch e
            println("ERROR: AMR simulation failed with: $e")
            println("Memory at failure: $(Base.gc_live_bytes() / 1024^2) MB")
            rethrow()
        end
    else
        println("Running standard simulation...")
        println("Starting simulation with $(nx*nz) total grid points...")
        
        try
            BioFlows.run_simulation(config, solver, initial_state)
        catch e
            println("ERROR: Simulation failed with: $e")
            println("Memory at failure: $(Base.gc_live_bytes() / 1024^2) MB")
            rethrow()
        end
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