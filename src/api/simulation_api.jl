"""
High-level user API for BioFlows.jl simulations.

This module provides convenient functions for setting up and running
biological flow simulations with various configurations.
"""

# Main simulation configuration structure
struct SimulationConfig
    # Domain and discretization
    grid_type::GridType
    nx::Int
    ny::Int
    nz::Int  # Only used for 3D
    Lx::Float64
    Ly::Float64
    Lz::Float64  # Only used for 3D
    origin::Vector{Float64}
    
    # Fluid properties
    fluid::FluidProperties
    
    # Boundary conditions
    bc::BoundaryConditions
    
    # Time integration
    time_scheme::TimeSteppingScheme
    dt::Float64
    final_time::Float64
    
    # Bodies (optional)
    rigid_bodies::Union{RigidBodyCollection, Nothing}
    flexible_bodies::Union{FlexibleBodyCollection, Nothing}
    flexible_body_controller::Union{FlexibleBodyController, Nothing}
    
    # Solver options
    use_mpi::Bool
    adaptive_refinement::Bool
    refinement_criteria::Union{AdaptiveRefinementCriteria, Nothing}
    
    # Output options
    output_config::NetCDFConfig
end

"""
    create_2d_simulation_config(; kwargs...)

Create configuration for 2D biological flow simulation.

# Arguments
- `nx::Int`: Number of grid points in x-direction
- `nz::Int`: Number of grid points in z-direction (vertical for XZ plane)
- `Lx::Float64`: Domain length in x-direction
- `Lz::Float64`: Domain length in z-direction
- `grid_type::GridType = TwoDimensional`: 2D flow in XZ plane (z is vertical)
- `origin::Vector{Float64} = [0.0, 0.0]`: Domain origin coordinates
- `Reynolds::Float64 = 100.0`: Reynolds number
- `density_type::Symbol = :constant`: Density type (:constant or :variable)
- `density_value::Float64 = 1.0`: Constant density value
- `viscosity::Float64 = 1.0/Reynolds`: Dynamic viscosity
- `inlet_velocity::Float64 = 1.0`: Inlet velocity
- `outlet_type::Symbol = :pressure`: Outlet boundary condition (:pressure or :velocity)
- `wall_type::Symbol = :no_slip`: Wall boundary condition (:no_slip, :free_slip, :periodic)
- `time_scheme::Symbol = :rk2`: Time stepping scheme (:adams_bashforth, :rk2, :rk4)
- `dt::Float64 = 0.01`: Time step size
- `final_time::Float64 = 10.0`: Final simulation time
- `use_mpi::Bool = false`: Enable MPI parallelization
- `adaptive_refinement::Bool = false`: Enable adaptive mesh refinement
- `output_interval::Float64 = 0.1`: Time interval for output
- `output_file::String = "bioflow_2d"`: Output file prefix
"""
function create_2d_simulation_config(;
    nx::Int,
    nz::Int,
    Lx::Float64,
    Lz::Float64,
    grid_type::GridType = TwoDimensional,
    origin::Vector{Float64} = [0.0, 0.0],
    Reynolds::Float64 = 100.0,
    density_type::Symbol = :constant,
    density_value::Float64 = 1.0,
    nu::Union{Nothing, Float64} = nothing,       # kinematic viscosity (m^2/s)
    mu::Union{Nothing, Float64} = nothing,       # dynamic viscosity (Pa·s)
    viscosity::Float64 = density_value / Reynolds,  # legacy path (dynamic viscosity)
    inlet_velocity::Float64 = 1.0,
    outlet_type::Symbol = :pressure,
    wall_type::Symbol = :no_slip,
    time_scheme::Symbol = :rk2,
    dt::Float64 = 0.01,
    final_time::Float64 = 10.0,
    use_mpi::Bool = false,
    adaptive_refinement::Bool = false,
    output_interval::Float64 = 0.1,
    output_file::String = "bioflow_2d",
    # Output config advanced options
    output_max_snapshots::Int = 100,
    output_save_mode::Symbol = :both,
    output_iteration_interval::Int = 10,
    output_save_flow_field::Bool = true,
    output_save_body_positions::Bool = true,
    output_save_force_coefficients::Bool = true,
    output_reference_velocity::Float64 = 1.0,
    output_flow_direction::Vector{Float64} = [1.0, 0.0])
    
    # Create fluid properties
    if density_type == :constant
        ρ = ConstantDensity(density_value)
    else
        error("Variable density not yet implemented")
    end
    
    # Determine dynamic viscosity μ with clear precedence: mu > nu*density > viscosity
    μ_eff = if mu !== nothing
        mu
    elseif nu !== nothing
        density_value * nu
    else
        viscosity
    end
    
    fluid = FluidProperties(μ_eff, ρ, Reynolds)
    
    # Create boundary conditions
    inlet_bc = InletBC(inlet_velocity, 0.0)  # u_inlet, v_inlet
    
    if outlet_type == :pressure
        outlet_bc = PressureOutletBC(0.0)
    else
        outlet_bc = VelocityOutletBC(inlet_velocity, 0.0)
    end
    
    if wall_type == :no_slip
        wall_bc = NoSlipBC()
    elseif wall_type == :free_slip
        wall_bc = FreeSlipBC()
    elseif wall_type == :periodic
        wall_bc = PeriodicBC()
    else
        error("Unknown wall type: $wall_type")
    end
    
    # 2D uses XZ plane: x-horizontal, z-vertical
    bc = BoundaryConditions2D(
        left=inlet_bc,    # x-direction inlet
        right=outlet_bc,  # x-direction outlet  
        bottom=wall_bc,   # z-direction bottom wall
        top=wall_bc       # z-direction top wall
    )
    
    # Create time stepping scheme
    if time_scheme == :adams_bashforth
        time_scheme_obj = AdamsBashforth()
    elseif time_scheme == :rk2
        time_scheme_obj = RungeKutta2()
    elseif time_scheme == :rk4
        time_scheme_obj = RungeKutta4()
    else
        error("Unknown time scheme: $time_scheme")
    end
    
    # Create output configuration (align with NetCDFConfig signature)
    output_config = NetCDFConfig(output_file;
        max_snapshots_per_file=output_max_snapshots,
        save_mode=output_save_mode,
        time_interval=output_interval,
        iteration_interval=output_iteration_interval,
        save_flow_field=output_save_flow_field,
        save_body_positions=output_save_body_positions,
        save_force_coefficients=output_save_force_coefficients,
        reference_velocity=output_reference_velocity,
        flow_direction=output_flow_direction,
    )
    
    # Create refinement criteria if needed
    refinement_criteria = adaptive_refinement ? AdaptiveRefinementCriteria() : nothing
    
    return SimulationConfig(
        grid_type, nx, 0, nz, Lx, 0.0, Lz, origin,  # XZ plane: ny=0, Ly=0.0
        fluid, bc, time_scheme_obj, dt, final_time,
        nothing, nothing, nothing,  # No bodies or controller by default
        use_mpi, adaptive_refinement, refinement_criteria,
        output_config
    )
end

# Temporarily disabled FlexibleBodyController functions
# """
#     add_flexible_bodies_with_controller!(config::SimulationConfig, 
#                                         bodies::FlexibleBodyCollection,
#                                         controller::FlexibleBodyController)
# 
# Add flexible bodies and their controller to simulation configuration.
# """
# function add_flexible_bodies_with_controller!(config::SimulationConfig, 
#                                             bodies::FlexibleBodyCollection,
#                                             controller::FlexibleBodyController)
#     # Implementation temporarily disabled
#     error("FlexibleBodyController support temporarily disabled")
# end

"""
    create_coordinated_flexible_system(flag_configs::Vector, distance_matrix::Matrix{Float64}; kwargs...)

Convenience function to create coordinated flexible body system for use in simulations.

# Arguments
- `flag_configs::Vector`: Vector of flag configuration NamedTuples
- `distance_matrix::Matrix{Float64}`: Target distances between flags
- Additional control options passed to create_coordinated_flag_system

# Returns
- `FlexibleBodyCollection`: Collection of flags
- `FlexibleBodyController`: Control system for coordination

# Example
```julia
# Create simulation config
config = create_2d_simulation_config(nx=128, nz=64, Lx=4.0, Lz=2.0)

# Define coordinated flexible system
flag_configs = [
    (start_point=[1.0, 1.0], length=0.8, width=0.04, 
     prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=2.0)),
    (start_point=[2.0, 1.0], length=0.6, width=0.03,
     prescribed_motion=(type=:sinusoidal, amplitude=0.08, frequency=2.0))
]
distance_matrix = [0.0 0.8; 0.8 0.0]

# Create coordinated system
bodies, controller = create_coordinated_flexible_system(flag_configs, distance_matrix;
                                                       base_frequency=2.0,
                                                       kp=0.5, ki=0.1, kd=0.05)

# Add to simulation
config = add_flexible_bodies_with_controller!(config, bodies, controller)

# Run simulation
run_simulation(config)
```
"""
function create_coordinated_flexible_system(flag_configs::Vector, distance_matrix::Matrix{Float64}; kwargs...)
    return create_coordinated_flag_system(flag_configs, distance_matrix; kwargs...)
end

"""
    add_flexible_bodies_with_controller!(config, bodies, controller)

Attach flexible bodies and their controller to the simulation config.
"""
function add_flexible_bodies_with_controller!(config::SimulationConfig,
                                             bodies::FlexibleBodyCollection,
                                             controller::FlexibleBodyController)
    return SimulationConfig(
        config.grid_type,
        config.nx, config.ny, config.nz,
        config.Lx, config.Ly, config.Lz,
        config.origin,
        config.fluid,
        config.bc,
        config.time_scheme,
        config.dt,
        config.final_time,
        config.rigid_bodies,
        bodies,
        controller,
        config.use_mpi,
        config.adaptive_refinement,
        config.refinement_criteria,
        config.output_config,
    )
end

"""
    create_3d_simulation_config(; kwargs...)

Create configuration for 3D biological flow simulation.

# Arguments
- `nx::Int`: Number of grid points in x-direction
- `ny::Int`: Number of grid points in y-direction
- `nz::Int`: Number of grid points in z-direction
- `Lx::Float64`: Domain length in x-direction
- `Ly::Float64`: Domain length in y-direction
- `Lz::Float64`: Domain length in z-direction
- `origin::Vector{Float64} = [0.0, 0.0, 0.0]`: Domain origin coordinates
- `Reynolds::Float64 = 100.0`: Reynolds number
- `density_type::Symbol = :constant`: Density type (:constant or :variable)
- `density_value::Float64 = 1.0`: Constant density value
- `viscosity::Float64 = 1.0/Reynolds`: Dynamic viscosity
- `inlet_velocity::Float64 = 1.0`: Inlet velocity
- `outlet_type::Symbol = :pressure`: Outlet boundary condition (:pressure or :velocity)
- `wall_type::Symbol = :no_slip`: Wall boundary condition (:no_slip, :free_slip, :periodic)
- `time_scheme::Symbol = :rk2`: Time stepping scheme (:adams_bashforth, :rk2, :rk4)
- `dt::Float64 = 0.01`: Time step size
- `final_time::Float64 = 10.0`: Final simulation time
- `use_mpi::Bool = false`: Enable MPI parallelization
- `adaptive_refinement::Bool = false`: Enable adaptive mesh refinement
- `output_interval::Float64 = 0.1`: Time interval for output
- `output_file::String = "bioflow_3d"`: Output file prefix
"""
function create_3d_simulation_config(;
    nx::Int,
    ny::Int, 
    nz::Int,
    Lx::Float64,
    Ly::Float64,
    Lz::Float64,
    origin::Vector{Float64} = [0.0, 0.0, 0.0],
    Reynolds::Float64 = 100.0,
    density_type::Symbol = :constant,
    density_value::Float64 = 1.0,
    nu::Union{Nothing, Float64} = nothing,       # kinematic viscosity (m^2/s)
    mu::Union{Nothing, Float64} = nothing,       # dynamic viscosity (Pa·s)
    viscosity::Float64 = density_value / Reynolds,  # legacy path (dynamic viscosity)
    inlet_velocity::Float64 = 1.0,
    outlet_type::Symbol = :pressure,
    wall_type::Symbol = :no_slip,
    time_scheme::Symbol = :rk2,
    dt::Float64 = 0.01,
    final_time::Float64 = 10.0,
    use_mpi::Bool = false,
    adaptive_refinement::Bool = false,
    output_interval::Float64 = 0.1,
    output_file::String = "bioflow_3d",
    # Output config advanced options
    output_max_snapshots::Int = 100,
    output_save_mode::Symbol = :both,
    output_iteration_interval::Int = 10,
    output_save_flow_field::Bool = true,
    output_save_body_positions::Bool = true,
    output_save_force_coefficients::Bool = true,
    output_reference_velocity::Float64 = 1.0,
    output_flow_direction::Vector{Float64} = [1.0, 0.0])
    
    # Create fluid properties
    if density_type == :constant
        ρ = ConstantDensity(density_value)
    else
        error("Variable density not yet implemented")
    end
    
    # Determine dynamic viscosity μ with clear precedence: mu > nu*density > viscosity
    μ_eff = if mu !== nothing
        mu
    elseif nu !== nothing
        density_value * nu
    else
        viscosity
    end
    
    fluid = FluidProperties(μ_eff, ρ, Reynolds)
    
    # Create boundary conditions
    inlet_bc = InletBC(inlet_velocity, 0.0, 0.0)  # u_inlet, v_inlet, w_inlet
    
    if outlet_type == :pressure
        outlet_bc = PressureOutletBC(0.0)
    else
        outlet_bc = VelocityOutletBC(inlet_velocity, 0.0, 0.0)
    end
    
    if wall_type == :no_slip
        wall_bc = NoSlipBC()
    elseif wall_type == :free_slip
        wall_bc = FreeSlipBC()
    elseif wall_type == :periodic
        wall_bc = PeriodicBC()
    else
        error("Unknown wall type: $wall_type")
    end
    
    bc = BoundaryConditions3D(inlet_bc, outlet_bc, wall_bc, wall_bc, wall_bc, wall_bc)  # x-, x+, y-, y+, z-, z+
    
    # Create time stepping scheme
    if time_scheme == :adams_bashforth
        time_scheme_obj = AdamsBashforth()
    elseif time_scheme == :rk2
        time_scheme_obj = RungeKutta2()
    elseif time_scheme == :rk4
        time_scheme_obj = RungeKutta4()
    else
        error("Unknown time scheme: $time_scheme")
    end
    
    # Create output configuration (align with NetCDFConfig signature)
    output_config = NetCDFConfig(output_file;
        max_snapshots_per_file=output_max_snapshots,
        save_mode=output_save_mode,
        time_interval=output_interval,
        iteration_interval=output_iteration_interval,
        save_flow_field=output_save_flow_field,
        save_body_positions=output_save_body_positions,
        save_force_coefficients=output_save_force_coefficients,
        reference_velocity=output_reference_velocity,
        flow_direction=output_flow_direction,
    )
    
    # Create refinement criteria if needed
    refinement_criteria = adaptive_refinement ? AdaptiveRefinementCriteria() : nothing
    
    return SimulationConfig(
        ThreeDimensional, nx, ny, nz, Lx, Ly, Lz, origin,
        fluid, bc, time_scheme_obj, dt, final_time,
        nothing, nothing, nothing,  # No bodies or controller by default
        use_mpi, adaptive_refinement, refinement_criteria,
        output_config
    )
end

"""
    add_rigid_circle!(config, center, radius; motion_type=:stationary)

Add a rigid circular body to the simulation.

# Arguments
- `config::SimulationConfig`: Simulation configuration
- `center::Vector{Float64}`: Circle center coordinates
- `radius::Float64`: Circle radius
- `motion_type::Symbol = :stationary`: Type of motion (:stationary, :prescribed)
"""
function add_rigid_circle!(config::SimulationConfig, center::Vector{Float64}, radius::Float64; 
                          motion_type::Symbol = :stationary)
    circle = Circle(radius)
    if motion_type == :prescribed
        error(":prescribed motion for rigid bodies not yet implemented")
    end
    body = RigidBody(circle, center)
    rb = config.rigid_bodies === nothing ? RigidBodyCollection() : config.rigid_bodies
    add_body!(rb, body)
    
    return SimulationConfig(
        config.grid_type, config.nx, config.ny, config.nz,
        config.Lx, config.Ly, config.Lz, config.origin,
        config.fluid, config.bc, config.time_scheme, config.dt, config.final_time,
        rb, config.flexible_bodies, config.flexible_body_controller,
        config.use_mpi, config.adaptive_refinement, config.refinement_criteria,
        config.output_config
    )
end

"""
    add_rigid_square!(config, center, side_length; motion_type=:stationary)

Add a rigid square body to the simulation.
"""
function add_rigid_square!(config::SimulationConfig, center::Vector{Float64}, side_length::Float64;
                          motion_type::Symbol = :stationary)
    square = Square(side_length)
    if motion_type == :prescribed
        error(":prescribed motion for rigid bodies not yet implemented")
    end
    body = RigidBody(square, center)
    rb = config.rigid_bodies === nothing ? RigidBodyCollection() : config.rigid_bodies
    add_body!(rb, body)
    
    return SimulationConfig(
        config.grid_type, config.nx, config.ny, config.nz,
        config.Lx, config.Ly, config.Lz, config.origin,
        config.fluid, config.bc, config.time_scheme, config.dt, config.final_time,
        rb, config.flexible_bodies, config.flexible_body_controller,
        config.use_mpi, config.adaptive_refinement, config.refinement_criteria,
        config.output_config
    )
end

"""
    add_flexible_body!(config, front_position, length, n_points; 
                       thickness=0.01, rigidity=1.0, density=1.0,
                       front_constraint=:fixed, motion_amplitude=0.0, motion_frequency=0.0)

Add a flexible body to 2D simulation (only supported in 2D).

# Arguments
- `config::SimulationConfig`: Simulation configuration  
- `front_position::Vector{Float64}`: Position of front end
- `length::Float64`: Length of flexible body
- `n_points::Int`: Number of Lagrangian points
- `thickness::Float64 = 0.01`: Body thickness
- `rigidity::Float64 = 1.0`: Bending rigidity parameter
- `density::Float64 = 1.0`: Body density
- `front_constraint::Symbol = :fixed`: Front end constraint (:fixed, :rotation, :sinusoidal)
- `motion_amplitude::Float64 = 0.0`: Amplitude for sinusoidal motion
- `motion_frequency::Float64 = 0.0`: Frequency for sinusoidal motion
"""
function add_flexible_body!(config::SimulationConfig, front_position::Vector{Float64}, 
                           body_length::Float64, n_points::Int;
                           thickness::Float64 = 0.01,
                           initial_angle::Union{Nothing, Float64} = nothing,
                           material::Symbol = :flexible,
                           prescribed_motion::Union{Nothing, NamedTuple} = nothing)
    if config.grid_type == ThreeDimensional
        error("Flexible bodies are only supported for 2D simulations")
    end
    # Map to flexible body flag constructor (XZ plane coordinates)
    width = thickness
    flag = create_flag(front_position, body_length, width;
                       n_points=n_points,
                       initial_angle=initial_angle,
                       material=material,
                       prescribed_motion=prescribed_motion)
    if config.flexible_bodies === nothing
        config.flexible_bodies = FlexibleBodyCollection([flag])
    else
        add_flexible_body!(config.flexible_bodies, flag)
    end
    return config
end

"""
    create_solver(config::SimulationConfig)

Create appropriate solver based on configuration.
"""
function create_solver(config::SimulationConfig)
    if config.grid_type == ThreeDimensional
        return create_3d_solver(
            config.nx, config.ny, config.nz, 
            config.Lx, config.Ly, config.Lz,
            config.fluid, config.bc;
            time_scheme=config.time_scheme,
            use_mpi=config.use_mpi,
            origin_x=config.origin[1],
            origin_y=config.origin[2], 
            origin_z=config.origin[3]
        )
    else
        return create_2d_solver(
            config.nx, config.nz, config.Lx, config.Lz,
            config.fluid, config.bc;
            grid_type=config.grid_type,
            time_scheme=config.time_scheme,
            use_mpi=config.use_mpi,
            origin_x=config.origin[1],
            origin_z=config.origin[2]
        )
    end
end

"""
    initialize_simulation(config::SimulationConfig; initial_conditions=:quiescent)

Initialize simulation state.

# Arguments
- `config::SimulationConfig`: Simulation configuration
- `initial_conditions::Symbol = :quiescent`: Initial condition type (:quiescent, :uniform_flow)
"""
function initialize_simulation(config::SimulationConfig; initial_conditions::Symbol = :quiescent)
    if config.grid_type == ThreeDimensional
        state = SolutionState3D(config.nx, config.ny, config.nz)
        
        if initial_conditions == :uniform_flow
            state.u .= 1.0  # Uniform flow in x-direction
        end
        
    else
        state = SolutionState2D(config.nx, config.nz)
        
        if initial_conditions == :uniform_flow
            state.u .= 1.0  # Uniform flow in x-direction
        end
    end
    
    return state
end

"""
    run_simulation(config::SimulationConfig, solver, initial_state::SolutionState)

Run the complete simulation.

# Arguments
- `config::SimulationConfig`: Simulation configuration
- `solver`: Solver object (created with create_solver)
- `initial_state::SolutionState`: Initial solution state
"""
function run_simulation(config::SimulationConfig, solver, initial_state::SolutionState)
    println("Starting BioFlows simulation...")
    println("  Grid: $(config.nx) × $(config.grid_type == ThreeDimensional ? config.ny : config.nz)" * (config.grid_type == ThreeDimensional ? " × $(config.nz)" : ""))
    println("  Domain: $(config.Lx) × $(config.grid_type == ThreeDimensional ? config.Ly : config.Lz)" * (config.grid_type == ThreeDimensional ? " × $(config.Lz)" : ""))
    println("  Time: 0.0 → $(config.final_time), dt = $(config.dt)")
    println("  MPI: $(config.use_mpi ? "enabled" : "disabled")")
    println("  Adaptive refinement: $(config.adaptive_refinement ? "enabled" : "disabled")")
    
    if config.rigid_bodies !== nothing
        println("  Rigid bodies: $(length(config.rigid_bodies.bodies))")
    end
    if config.flexible_bodies !== nothing
        println("  Flexible bodies: $(length(config.flexible_bodies.bodies))")
    end
    
    # Initialize output
    writer = NetCDFWriter("$(config.output_config.filename).nc", solver.grid, config.output_config)
    # Attach domain metadata and body info for plotting/post-processing
    try
        # Ensure file exists and has core vars
        initialize_netcdf_file!(writer)
        NetCDF.putatt(writer.ncfile, "global", Dict(
            "domain_Lx"=>solver.grid.Lx,
            "domain_Lz"=>solver.grid.Lz,
            "inlet_velocity"=>(config.bc isa BoundaryConditions2D ? (config.bc.left isa InletBC ? (config.bc.left.u) : 0.0) : 0.0),
        ))
        # Prefer rigid bodies for geometry; flexible handled with count only
        bodies_for_meta = config.rigid_bodies !== nothing ? config.rigid_bodies : (config.flexible_bodies !== nothing ? config.flexible_bodies : nothing)
        annotate_bodies_metadata!(writer, bodies_for_meta)
    catch e
        @warn "Could not annotate NetCDF metadata: $e"
    end

    # Initialize adaptive refinement if needed
    refined_grid = nothing
    amr_solver = nothing
    if config.adaptive_refinement
        # Create AMR-integrated solver for consistent 2D AMR handling
        amr_solver = create_amr_integrated_solver(solver, config.refinement_criteria; amr_frequency=10)
        refined_grid = amr_solver.refined_grid
    end
    
    # Time stepping loop
    state_old = deepcopy(initial_state)
    state_new = deepcopy(initial_state)
    
    t = 0.0
    step = 0
    next_output_time = config.output_config.time_interval
    
    # Select bodies to save (if any)
    bodies_for_output = config.rigid_bodies !== nothing ? config.rigid_bodies : (config.flexible_bodies !== nothing ? config.flexible_bodies : nothing)
    
    if config.adaptive_refinement
        # Project AMR-enhanced data to original grid and save
        save_amr_to_netcdf!(writer, refined_grid, state_old, step, t; bodies=bodies_for_output)
    else
        write_solution!(writer, state_old, bodies_for_output, solver.grid, config.fluid, t, step)
    end
    
    # Diagnostics step frequency (env override: BIOFLOWS_DIAG_EVERY)
    diag_steps = try
        parse(Int, get(ENV, "BIOFLOWS_DIAG_EVERY", "100"))
    catch
        100
    end

    while t < config.final_time
        step += 1
        dt = min(config.dt, config.final_time - t)
        
        # Solve one time step (AMR-aware if enabled)
        if config.adaptive_refinement && amr_solver !== nothing
            amr_solve_step!(amr_solver, state_new, state_old, dt)
        else
            if config.grid_type == ThreeDimensional
                solve_step_3d!(solver, state_new, state_old, dt)
            else
                solve_step_2d!(solver, state_new, state_old, dt)
            end
        end
        
        # Apply immersed boundary forcing
        if config.rigid_bodies !== nothing
            apply_immersed_boundary_forcing!(state_new, config.rigid_bodies, solver.grid, dt)
            
            # Check for NaN/Inf after immersed boundary forcing
            if any(x -> isnan(x) || isinf(x), state_new.u) || any(x -> isnan(x) || isinf(x), state_new.w)
                @warn "NaN/Inf detected after immersed boundary forcing - cleaning"
                replace!(state_new.u, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
                replace!(state_new.w, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
            end
            
            # Only re-apply inlet boundary conditions (not all boundaries) to avoid conflicts
            if haskey(solver.bc.conditions, (:x, :left))
                condition = solver.bc.conditions[(:x, :left)]
                if condition.type == Inlet
                    # Apply inlet velocity to left boundary only
                    for j = 1:solver.grid.nz
                        if j <= size(state_new.u, 2)
                            inlet_value = isa(condition.value, Function) ? condition.value(t) : condition.value
                            state_new.u[1, j] = inlet_value
                        end
                    end
                end
            end
        end
        
        if config.flexible_bodies !== nothing
            # Apply controller-driven boundary conditions if available
            if config.flexible_body_controller !== nothing
                apply_harmonic_boundary_conditions!(config.flexible_body_controller, t)
            end
            # Update flexible body dynamics using same time scheme as fluid solver
            update_flexible_bodies!(config.flexible_bodies, state_new, solver.grid, dt, solver.time_scheme)
            if config.grid_type == ThreeDimensional
                apply_flexible_ib_forcing_3d!(state_new, config.flexible_bodies, solver.grid, dt)
            else
                apply_flexible_ib_forcing_2d!(state_new, config.flexible_bodies, solver.grid, dt)
            end
        end
        
        # Adaptive refinement handled inside AMR-integrated solver; no-op here
        
        t += dt
        state_old, state_new = state_new, state_old  # Swap states
        
        # Output
        if t >= next_output_time || (diag_steps > 0 && step % diag_steps == 0)
            if config.adaptive_refinement
                save_amr_to_netcdf!(writer, refined_grid, state_old, step, t; bodies=bodies_for_output)
            else
                write_solution!(writer, state_old, bodies_for_output, solver.grid, config.fluid, t, step)
            end
            next_output_time += config.output_config.time_interval
            
            println("Step $step: t = $(round(t, digits=4)), dt = $(round(dt, digits=6))")
            
            # Compute and display CFL number
            if config.grid_type == ThreeDimensional
                cfl = compute_cfl_3d(state_old.u, state_old.v, state_old.w, solver.grid, dt)
                max_u = maximum(abs, state_old.u)
                max_v = maximum(abs, state_old.v)
                max_w = maximum(abs, state_old.w)
                println("  max|u|=$(round(max_u, digits=4)), max|v|=$(round(max_v, digits=4)), max|w|=$(round(max_w, digits=4))")
            else
                cfl = compute_cfl_2d(state_old.u, state_old.w, solver.grid, dt)
                max_u = maximum(abs, state_old.u)
                max_w = maximum(abs, state_old.w)  # w holds vertical velocity in 2D XZ plane
                println("  max|u|=$(round(max_u, digits=4)), max|w|=$(round(max_w, digits=4))")
            end
            println("  CFL = $(round(cfl, digits=4))")
        end
    end
    
    # Final output
    if config.adaptive_refinement
        save_amr_to_netcdf!(writer, refined_grid, state_old, step, t; bodies=bodies_for_output)
    else
        write_solution!(writer, state_old, bodies_for_output, solver.grid, config.fluid, t, step)
    end
    close!(writer)
    
    println("Simulation completed!")
    println("Output saved to $(config.output_config.filename)")
    
    return state_old
end

"""
    run_bioflow_2d(; kwargs...)

Convenience function to set up and run a complete 2D simulation.

This function combines configuration creation, solver setup, initialization, and execution.
"""
function run_bioflow_2d(; kwargs...)
    config = create_2d_simulation_config(; kwargs...)
    solver = create_solver(config)
    initial_state = initialize_simulation(config)
    
    return run_simulation(config, solver, initial_state)
end

"""
    run_bioflow_3d(; kwargs...)

Convenience function to set up and run a complete 3D simulation.

This function combines configuration creation, solver setup, initialization, and execution.
"""
function run_bioflow_3d(; kwargs...)
    config = create_3d_simulation_config(; kwargs...)
    solver = create_solver(config)
    initial_state = initialize_simulation(config)
    
    return run_simulation(config, solver, initial_state)
end

"""
    run_simulation_mpi_2d(config::SimulationConfig)

Run a 2D MPI simulation end-to-end with diagnostics and global NetCDF output.
Uses the MPI-aware writer to gather and write snapshots to a single file.
"""
function run_simulation_mpi_2d(config::SimulationConfig)
    @assert config.use_mpi "run_simulation_mpi_2d requires config.use_mpi = true"
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Create solver and local states
    solver = create_solver(config)
    local_old = MPISolutionState2D(solver.decomp)
    local_new = MPISolutionState2D(solver.decomp)

    # Initialize uniform flow using inlet velocity if available
    u_init = 0.0
    try
        if hasproperty(config.bc, :left) && (config.bc.left isa InletBC)
            u_init = getfield(config.bc.left, :u)
        end
    catch
        u_init = 0.0
    end
    local_old.u .= u_init
    local_old.w .= 0.0
    local_old.p .= 0.0
    local_old.t = 0.0
    local_old.step = 0

    # Global grid + writer
    global_grid = StaggeredGrid2D(config.nx, config.nz, config.Lx, config.Lz)
    writer = NetCDFWriter("$(config.output_config.filename).nc", global_grid, config.output_config)
    if rank == 0
        initialize_netcdf_file!(writer)
        # Annotate metadata
        NetCDF.putatt(writer.ncfile, "global", Dict(
            "domain_Lx"=>config.Lx, "domain_Lz"=>config.Lz,
        ))
        bodies_for_meta = config.rigid_bodies !== nothing ? config.rigid_bodies : (config.flexible_bodies !== nothing ? config.flexible_bodies : nothing)
        annotate_bodies_metadata!(writer, bodies_for_meta)
        println("MPI run with $nprocs ranks: 2D, $(config.nx)x$(config.nz), dt=$(config.dt), T=$(config.final_time)")
    end

    # Diagnostics and output cadence
    save_interval = config.output_config.time_interval
    next_print = save_interval
    dx = config.Lx / config.nx
    dz = config.Lz / config.nz
    t = 0.0
    step = 0

    # Initial simulation info
    if rank == 0
        println("="^60)
        println("STARTING MPI 2D SIMULATION")
        println("Grid: $(config.nx) x $(config.nz) (total: $(config.nx * config.nz) cells)")
        println("Domain: $(config.Lx) x $(config.Lz)")
        println("Resolution: dx=$(round(dx, digits=6)), dz=$(round(dz, digits=6))")
        println("Time: dt=$(config.dt), T_final=$(config.final_time)")
        println("MPI ranks: $nprocs")
        println("Output interval: $save_interval (time units)")
        println("="^60)
        flush(stdout)
    end

    while t < config.final_time - 1e-12
        step += 1
        dt = min(config.dt, config.final_time - t)

        mpi_solve_step_2d!(solver, local_new, local_old, dt)

        # Swap
        local_old, local_new = local_new, local_old
        t += dt

        # Frequent diagnostics every 50 steps
        if step % 50 == 0
            # Global maxima for diagnostics
            maxu = MPI.Allreduce(maximum(abs, local_old.u), MPI.MAX, comm)
            maxw = MPI.Allreduce(maximum(abs, local_old.w), MPI.MAX, comm)
            cfl = max(maxu * dt / dx, maxw * dt / dz)
            if rank == 0
                println("DIAG: iter=$step t=$(round(t, digits=4)) dt=$(round(dt, digits=6)) CFL=$(round(cfl, digits=4)) max|u|=$(round(maxu, digits=4)) max|w|=$(round(maxw, digits=4))")
            end
        end

        # Output + detailed diagnostics
        if t + 1e-12 >= next_print
            # Global maxima for output
            maxu = MPI.Allreduce(maximum(abs, local_old.u), MPI.MAX, comm)
            maxw = MPI.Allreduce(maximum(abs, local_old.w), MPI.MAX, comm)
            cfl = max(maxu * dt / dx, maxw * dt / dz)
            if rank == 0
                println("="^60)
                println("OUTPUT: iter=$step t=$(round(t, digits=4)) dt=$(round(dt, digits=6))")
                println("        CFL=$(round(cfl, digits=4)) max|u|=$(round(maxu, digits=4)) max|w|=$(round(maxw, digits=4))")
                println("        Progress: $(round(100*t/config.final_time, digits=1))% complete")
                println("="^60)
            end
            # Global NetCDF output (auto-detects MPI state)
            write_solution!(writer, local_old, config.rigid_bodies, global_grid, solver.fluid, t, step)
            next_print += save_interval
        end
    end

    # Final simulation summary
    if rank == 0
        final_maxu = MPI.Allreduce(maximum(abs, local_old.u), MPI.MAX, comm)
        final_maxw = MPI.Allreduce(maximum(abs, local_old.w), MPI.MAX, comm)
        final_cfl = max(final_maxu * dt / dx, final_maxw * dt / dz)
        
        println("="^60)
        println("MPI 2D SIMULATION COMPLETE")
        println("Final time: $(round(t, digits=4)) / $(config.final_time)")
        println("Total iterations: $step")
        println("Final CFL: $(round(final_cfl, digits=4))")
        println("Final max|u|: $(round(final_maxu, digits=4))")
        println("Final max|w|: $(round(final_maxw, digits=4))")
        println("Output written to: $(config.output_config.filename).nc")
        println("="^60)
    else
        # Other ranks still need to participate in the MPI.Allreduce calls
        MPI.Allreduce(maximum(abs, local_old.u), MPI.MAX, comm)
        MPI.Allreduce(maximum(abs, local_old.w), MPI.MAX, comm)
    end

    return local_old
end

"""
    run_simulation_mpi_3d(config::SimulationConfig)

Run a 3D MPI simulation with diagnostics and single-file global NetCDF output.
"""
function run_simulation_mpi_3d(config::SimulationConfig)
    @assert config.use_mpi "run_simulation_mpi_3d requires config.use_mpi = true"
    @assert config.grid_type == ThreeDimensional "run_simulation_mpi_3d requires 3D config"
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Create solver and local states (use local domain sizes)
    solver = create_solver(config)
    decomp = solver.decomp
    nxl, nyl, nzl = decomp.nx_local, decomp.ny_local, decomp.nz_local
    local_old = SolutionState3D(nxl, nyl, nzl)
    local_new = SolutionState3D(nxl, nyl, nzl)

    # Initialize uniform flow using inlet U
    u_init = 0.0
    try
        if hasproperty(config.bc, :x_minus) && (config.bc.x_minus isa InletBC)
            u_init = getfield(config.bc.x_minus, :u)
        end
    catch
        u_init = 0.0
    end
    local_old.u .= u_init
    local_old.v .= 0.0
    local_old.w .= 0.0
    local_old.p .= 0.0
    local_old.t = 0.0
    local_old.step = 0

    # Global writer (root initializes)
    global_grid = StaggeredGrid3D(config.nx, config.ny, config.nz, config.Lx, config.Ly, config.Lz)
    writer = NetCDFWriter("$(config.output_config.filename).nc", global_grid, config.output_config)
    if rank == 0
        initialize_netcdf_file!(writer)
        NetCDF.putatt(writer.ncfile, "global", Dict(
            "domain_Lx"=>config.Lx, "domain_Ly"=>config.Ly, "domain_Lz"=>config.Lz,
        ))
        bodies_for_meta = config.rigid_bodies !== nothing ? config.rigid_bodies : (config.flexible_bodies !== nothing ? config.flexible_bodies : nothing)
        annotate_bodies_metadata!(writer, bodies_for_meta)
        println("MPI run with $nprocs ranks: 3D, $(config.nx)x$(config.ny)x$(config.nz), dt=$(config.dt), T=$(config.final_time)")
    end

    # Diagnostics cadence
    save_interval = config.output_config.time_interval
    next_print = save_interval
    dx = config.Lx / config.nx
    dy = config.Ly / config.ny
    dz = config.Lz / config.nz
    t = 0.0
    step = 0

    while t < config.final_time - 1e-12
        step += 1
        dt = min(config.dt, config.final_time - t)

        mpi_solve_step_3d!(solver, local_new, local_old, dt)
        local_old, local_new = local_new, local_old
        t += dt

        if t + 1e-12 >= next_print || step % 100 == 0
            # Global maxima and CFL
            maxu = MPI.Allreduce(maximum(abs, local_old.u), MPI.MAX, comm)
            maxv = MPI.Allreduce(maximum(abs, local_old.v), MPI.MAX, comm)
            maxw = MPI.Allreduce(maximum(abs, local_old.w), MPI.MAX, comm)
            cfl = maximum([maxu*dt/dx, maxv*dt/dy, maxw*dt/dz])
            if rank == 0
                @info "step=$step t=$(round(t, digits=3)) dt=$(round(dt, digits=4)) CFL=$(round(cfl, digits=3)) max|u|=$(round(maxu, digits=3)) max|v|=$(round(maxv, digits=3)) max|w|=$(round(maxw, digits=3))"
            end
            # Wrap local SolutionState3D in MPISolutionState3D for writer gather
            local_mpi = MPISolutionState3D{Float64}(local_old.u, local_old.v, local_old.w, local_old.p, local_old.t, local_old.step, decomp)
            write_solution!(writer, local_mpi, config.rigid_bodies, global_grid, solver.fluid, t, step)
            next_print += save_interval
        end
    end

    if rank == 0
        println("MPI 3D simulation complete. Global output written to $(config.output_config.filename).nc")
    end
    return local_old
end
