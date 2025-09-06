using Dates
# Optional MPI support for automatic global writes
const HAS_MPI_WRITER = let ok = false
    try
        @eval using MPI
        ok = true
    catch
        ok = false
    end
    ok
end

"""
    NetCDFConfig

Configuration for NetCDF output with flexible save intervals and options.

# Fields
- `filename::String`: Base filename for output
- `max_snapshots_per_file::Int`: Maximum snapshots per file (default: 100)
- `save_mode::Symbol`: Save trigger mode - `:time_interval`, `:iteration_interval`, or `:both` (default: `:both`)
- `time_interval::Float64`: Time interval between saves (default: 0.1)
- `iteration_interval::Int`: Iteration interval between saves (default: 10)
- `save_flow_field::Bool`: Save velocity and pressure fields (default: true)
- `save_body_positions::Bool`: Save body kinematic data (default: true)  
- `save_force_coefficients::Bool`: Save drag/lift coefficients (default: true)
- `reference_velocity::Float64`: Reference velocity for coefficients (default: 1.0)
- `flow_direction::Vector{Float64}`: Main flow direction [x, z] (default: [1.0, 0.0])
"""
struct NetCDFConfig
    filename::String
    max_snapshots_per_file::Int
    save_mode::Symbol  # :time_interval, :iteration_interval, or :both
    time_interval::Float64
    iteration_interval::Int
    save_flow_field::Bool
    save_body_positions::Bool
    save_force_coefficients::Bool
    reference_velocity::Float64
    flow_direction::Vector{Float64}
    enable_deflate::Bool
    deflate_level::Int
    shuffle_filter::Bool
    
    function NetCDFConfig(filename::String;
                         max_snapshots_per_file::Int = 100,
                         save_mode::Symbol = :both,
                         time_interval::Float64 = 0.1,
                         iteration_interval::Int = 10,
                         save_flow_field::Bool = true,
                         save_body_positions::Bool = true,
                         save_force_coefficients::Bool = true,
                         reference_velocity::Float64 = 1.0,
                         flow_direction::Vector{Float64} = [1.0, 0.0],
                         enable_deflate::Bool = true,
                         deflate_level::Int = 4,
                         shuffle_filter::Bool = true)
        
        if !(save_mode in [:time_interval, :iteration_interval, :both])
            error("save_mode must be :time_interval, :iteration_interval, or :both")
        end
        
        new(filename, max_snapshots_per_file, save_mode, time_interval, iteration_interval,
            save_flow_field, save_body_positions, save_force_coefficients,
            reference_velocity, flow_direction, enable_deflate, deflate_level, shuffle_filter)
    end
end

"""
    annotate_bodies_metadata!(writer, bodies)

Attach simple rigid/flexible body metadata as global attributes to help post-processing.
Currently records basic info for rigid bodies (type, center, size parameters).
"""
function annotate_bodies_metadata!(writer::NetCDFWriter,
                                   bodies::Union{Nothing,RigidBodyCollection,FlexibleBodyCollection})
    if writer.ncfile === nothing
        initialize_netcdf_file!(writer)
    end
    if bodies === nothing
        return
    end
    nc = writer.ncfile
    try
        if bodies isa RigidBodyCollection
            NetCDF.putatt(nc, "global", "rigid_bodies", bodies.n_bodies)
            for (i, body) in enumerate(bodies.bodies)
                NetCDF.putatt(nc, "global", "body_$(i)_angle", body.angle)
                NetCDF.putatt(nc, "global", "body_$(i)_center_x", body.center[1])
                # 2D XZ plane: use z as vertical
                NetCDF.putatt(nc, "global", "body_$(i)_center_z", length(body.center)>1 ? body.center[end] : 0.0)
                if body.shape isa Circle
                    NetCDF.putatt(nc, "global", "body_$(i)_type", "Circle")
                    NetCDF.putatt(nc, "global", "body_$(i)_radius", (body.shape::Circle).radius)
                    # For convenience, also store canonical single-cylinder keys
                    if i == 1
                        NetCDF.putatt(nc, "global", Dict(
                            "cylinder_x"=>body.center[1],
                            "cylinder_z"=>length(body.center)>1 ? body.center[end] : 0.0,
                            "cylinder_radius"=>(body.shape::Circle).radius,
                        ))
                    end
                elseif body.shape isa Square
                    NetCDF.putatt(nc, "global", "body_$(i)_type", "Square")
                    NetCDF.putatt(nc, "global", "body_$(i)_side", (body.shape::Square).side_length)
                elseif body.shape isa Rectangle
                    NetCDF.putatt(nc, "global", "body_$(i)_type", "Rectangle")
                    NetCDF.putatt(nc, "global", "body_$(i)_width", (body.shape::Rectangle).width)
                    NetCDF.putatt(nc, "global", "body_$(i)_height", (body.shape::Rectangle).height)
                else
                    NetCDF.putatt(nc, "global", "body_$(i)_type", "Unknown")
                end
            end
        elseif bodies isa FlexibleBodyCollection
            NetCDF.putatt(nc, "global", "flexible_bodies", bodies.n_bodies)
        end
    catch e
        @warn "Failed to write body metadata: $e"
    end
end

mutable struct NetCDFWriter
    filepath::String
    base_filepath::String
    grid::StaggeredGrid
    config::NetCDFConfig
    current_snapshot::Int
    last_save_time::Float64
    last_save_iteration::Int
    ncfile::Union{Nothing, NetCDF.NcFile}  # NetCDF file handle
    file_index::Int  # 0 for base file, 1 for _1, 2 for _2, ...
    
    function NetCDFWriter(filepath::String, grid::StaggeredGrid, config::NetCDFConfig)
        new(filepath, filepath, grid, config, 0, 0.0, 0, nothing, 0)
    end
    
    # Backward compatibility constructor
    function NetCDFWriter(filepath::String, grid::StaggeredGrid;
                         max_snapshots::Int=100,
                         time_interval::Float64=0.1,
                         iteration_interval::Int=10)
        config = NetCDFConfig("temp"; 
                            max_snapshots_per_file=max_snapshots,
                            time_interval=time_interval,
                            iteration_interval=iteration_interval)
        new(filepath, filepath, grid, config, 0, 0.0, 0, nothing, 0)
    end
end

function initialize_netcdf_file!(writer::NetCDFWriter)
    # Create NetCDF file with appropriate dimensions and variables
    nx = writer.grid.nx
    is_3d = writer.grid.grid_type == ThreeDimensional
    ny = is_3d ? writer.grid.ny : 0
    nz = writer.grid.nz
    
    # Remove existing file if it exists
    if isfile(writer.filepath)
        rm(writer.filepath)
    end
    
    # Minimal definition for position-only mode (no flow field)
    if !writer.config.save_flow_field
        tdim = NetCDF.NcDim("time", writer.config.max_snapshots_per_file)
        timevar = NetCDF.NcVar("time", [tdim]; t=Float64)
        ncfile = NetCDF.create(writer.filepath, timevar)
        writer.ncfile = ncfile
        return ncfile
    end

    # Define coordinate variables and dims for flow-field saving
    ncfile = NetCDF.create(writer.filepath)
    # Core dims
    NetCDF.defDim(ncfile, "nx", nx)
    if is_3d
        NetCDF.defDim(ncfile, "ny", ny)
    end
    NetCDF.defDim(ncfile, "nz", nz)
    NetCDF.defDim(ncfile, "time", writer.config.max_snapshots_per_file)
    # Staggered dims
    NetCDF.defDim(ncfile, "nx_u", nx + 1)
    if is_3d
        NetCDF.defDim(ncfile, "ny_v", ny + 1)
    end
    NetCDF.defDim(ncfile, "nz_w", nz + 1)

    # Coordinate vars
    NetCDF.defVar(ncfile, "x", Float64, ("nx",))
    if is_3d
        NetCDF.defVar(ncfile, "y", Float64, ("ny",))
    end
    NetCDF.defVar(ncfile, "z", Float64, ("nz",))
    NetCDF.defVar(ncfile, "time", Float64, ("time",))

    # Staggered coordinate vars
    NetCDF.defVar(ncfile, "xu", Float64, ("nx_u",))
    if is_3d
        NetCDF.defVar(ncfile, "yv", Float64, ("ny_v",))
    end
    NetCDF.defVar(ncfile, "zw", Float64, ("nz_w",))

    # Flow-field vars
    if writer.config.save_flow_field
        if is_3d
            NetCDF.defVar(ncfile, "u", Float64, ("nx_u", "ny", "nz", "time"))
            NetCDF.defVar(ncfile, "v", Float64, ("nx", "ny_v", "nz", "time"))
            NetCDF.defVar(ncfile, "w", Float64, ("nx", "ny", "nz_w", "time"))
            NetCDF.defVar(ncfile, "p", Float64, ("nx", "ny", "nz", "time"))
        else
            # 2D XZ plane: use z as vertical dimension
            NetCDF.defVar(ncfile, "u", Float64, ("nx_u", "nz", "time"))
            NetCDF.defVar(ncfile, "w", Float64, ("nx", "nz_w", "time"))
            NetCDF.defVar(ncfile, "p", Float64, ("nx", "nz", "time"))
        end

        # Attempt to apply compression (deflate) if available
        if writer.config.enable_deflate
            vars_to_deflate = String[]
            push!(vars_to_deflate, "u")
            if is_3d
                push!(vars_to_deflate, "v")
                push!(vars_to_deflate, "w")
            else
                push!(vars_to_deflate, "w")
            end
            push!(vars_to_deflate, "p")
            try
                for nm in vars_to_deflate
                    if haskey(ncfile.vars, nm)
                        _apply_deflate!(ncfile, nm, writer.config)
                    end
                end
            catch e
                @warn "Deflate not applied: $e"
            end
        end

        # Attributes
        NetCDF.putatt(ncfile, "u", Dict("long_name" => "x-velocity", "units" => "m/s"))
        if is_3d
            NetCDF.putatt(ncfile, "v", Dict("long_name" => "y-velocity", "units" => "m/s"))
            NetCDF.putatt(ncfile, "w", Dict("long_name" => "z-velocity", "units" => "m/s"))
        else
            NetCDF.putatt(ncfile, "w", Dict("long_name" => "vertical velocity (z)", "units" => "m/s"))
        end
        NetCDF.putatt(ncfile, "p", Dict("long_name" => "pressure", "units" => "Pa"))
    end

    # Write coordinate data
    NetCDF.putvar(ncfile, "x", writer.grid.x)
    if is_3d
        NetCDF.putvar(ncfile, "y", writer.grid.y)
    end
    NetCDF.putvar(ncfile, "z", writer.grid.z)
    NetCDF.putvar(ncfile, "xu", writer.grid.xu)
    if is_3d
        NetCDF.putvar(ncfile, "yv", writer.grid.yv)
    end
    NetCDF.putvar(ncfile, "zw", writer.grid.zw)
    
    # Add global attributes
    NetCDF.putatt(ncfile, "global", Dict(
        "title" => "BioFlows.jl simulation results",
        "institution" => "BioFlows.jl",
        "source" => "Finite volume Navier-Stokes solver with immersed boundary method",
        "grid_type" => string(writer.grid.grid_type),
        "nx" => nx,
        "ny" => is_3d ? ny : 0,
        "nz" => nz,
        "Lx" => writer.grid.Lx,
        "Ly" => writer.grid.Ly,
        "Lz" => writer.grid.Lz,
        "dx" => writer.grid.dx,
        "dy" => writer.grid.dy,
        "dz" => writer.grid.dz,
        "max_snapshots" => writer.config.max_snapshots_per_file,
        "time_interval" => writer.config.time_interval,
        "iteration_interval" => writer.config.iteration_interval
    ))
    
    # Store file handle
    writer.ncfile = ncfile
    
    return ncfile
end

# Best-effort compression application compatible with NetCDF.jl variants
function _apply_deflate!(ncfile::NetCDF.NcFile, varname::String, cfg::NetCDFConfig)
    # Try common signatures: by var object or by (file, name)
    if haskey(ncfile.vars, varname)
        var = ncfile.vars[varname]
        for f in (Symbol("defVarDeflate"), Symbol("nc_def_var_deflate"))
            if isdefined(NetCDF, f)
                try
                    # Try (var, shuffle, deflate, level)
                    getproperty(NetCDF, f)(var, cfg.shuffle_filter, true, cfg.deflate_level)
                    return
                catch
                    # Try (ncid, varid, shuffle, deflate, level)
                    try
                        getproperty(NetCDF, f)(ncfile.ncid, var.varid, cfg.shuffle_filter, true, cfg.deflate_level)
                        return
                    catch
                    end
                end
            end
        end
    end
    # If not available, silently continue
end

function should_save(writer::NetCDFWriter, current_time::Float64, current_iteration::Int)
    # Check if it's time to save based on configured save mode
    config = writer.config
    
    time_condition = (current_time - writer.last_save_time) >= config.time_interval
    iteration_condition = (current_iteration - writer.last_save_iteration) >= config.iteration_interval
    
    if config.save_mode == :time_interval
        return time_condition
    elseif config.save_mode == :iteration_interval
        return iteration_condition
    else  # :both
        return time_condition || iteration_condition
    end
end

function save_snapshot!(writer::NetCDFWriter, state::SolutionState, current_time::Float64, current_iteration::Int)
    if writer.ncfile === nothing
        initialize_netcdf_file!(writer)
    end
    
    # Check if we should save
    if !should_save(writer, current_time, current_iteration)
        return false
    end
    
    # Check if we have room for more snapshots; if not, create new file
    if writer.current_snapshot >= writer.config.max_snapshots_per_file
        create_new_file!(writer)
    end
    
    writer.current_snapshot += 1
    snapshot_idx = writer.current_snapshot
    
    try
        # Write time
        NetCDF.putvar(writer.ncfile, "time", [current_time], start=[snapshot_idx])
        
        # Only save flow field if configured to do so
        if writer.config.save_flow_field
            # Write velocity and pressure data
            is_3d = writer.grid.grid_type == ThreeDimensional
            
            if is_3d
                NetCDF.putvar(writer.ncfile, "u", state.u, start=[1, 1, 1, snapshot_idx])
                NetCDF.putvar(writer.ncfile, "v", state.v, start=[1, 1, 1, snapshot_idx])
                NetCDF.putvar(writer.ncfile, "w", state.w, start=[1, 1, 1, snapshot_idx])
                NetCDF.putvar(writer.ncfile, "p", state.p, start=[1, 1, 1, snapshot_idx])
            else
                NetCDF.putvar(writer.ncfile, "u", state.u, start=[1, 1, snapshot_idx])
                # 2D vertical velocity is stored as "w" and comes from state.v array
                NetCDF.putvar(writer.ncfile, "w", state.v, start=[1, 1, snapshot_idx])
                NetCDF.putvar(writer.ncfile, "p", state.p, start=[1, 1, snapshot_idx])
            end
        end
        
        # Update last save times
        writer.last_save_time = current_time
        writer.last_save_iteration = current_iteration
        
        println("Saved snapshot $(snapshot_idx) at time $(current_time), iteration $(current_iteration)")
        return true
        
    catch e
        @error "Failed to save snapshot: $e"
        return false
    end
end

function save_body_data!(writer::NetCDFWriter, bodies::Union{RigidBodyCollection, FlexibleBodyCollection}, 
                        current_time::Float64, current_iteration::Int)
    # Save body position and motion data
    if writer.ncfile === nothing
        initialize_netcdf_file!(writer)
    end
    
    if !should_save(writer, current_time, current_iteration)
        return false
    end
    
    snapshot_idx = writer.current_snapshot
    
    if bodies isa RigidBodyCollection
        # Save rigid body data
        n_bodies = bodies.n_bodies
        if n_bodies > 0
            # Define body variables if not already defined
            if !haskey(writer.ncfile.vars, "body_center_x")
                NetCDF.defVar(writer.ncfile, "body_center_x", Float64, ("n_bodies", "time"))
                NetCDF.defVar(writer.ncfile, "body_center_y", Float64, ("n_bodies", "time"))
                NetCDF.defVar(writer.ncfile, "body_angle", Float64, ("n_bodies", "time"))
                NetCDF.defVar(writer.ncfile, "body_velocity_x", Float64, ("n_bodies", "time"))
                NetCDF.defVar(writer.ncfile, "body_velocity_y", Float64, ("n_bodies", "time"))
                NetCDF.defVar(writer.ncfile, "body_angular_velocity", Float64, ("n_bodies", "time"))
                
                writer.ncfile.dim["n_bodies"] = n_bodies
            end
            
            # Extract body data
            centers_x = [body.center[1] for body in bodies.bodies]
            centers_y = [body.center[2] for body in bodies.bodies]
            angles = [body.angle for body in bodies.bodies]
            velocities_x = [body.velocity[1] for body in bodies.bodies]
            velocities_y = [body.velocity[2] for body in bodies.bodies]
            angular_velocities = [body.angular_velocity for body in bodies.bodies]
            
            # Save data
            NetCDF.putvar(writer.ncfile, "body_center_x", centers_x, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "body_center_y", centers_y, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "body_angle", angles, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "body_velocity_x", velocities_x, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "body_velocity_y", velocities_y, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "body_angular_velocity", angular_velocities, start=[1, snapshot_idx])
        end
        
    elseif bodies isa FlexibleBodyCollection
        # Save flexible body data
        n_bodies = bodies.n_bodies
        if n_bodies > 0
            for (body_idx, body) in enumerate(bodies.bodies)
                # For each flexible body, save all Lagrangian points
                body_var_x = "flexible_body_$(body_idx)_x"
                body_var_z = "flexible_body_$(body_idx)_z"  # Updated for XZ plane
                
                if !haskey(writer.ncfile.vars, body_var_x)
                    NetCDF.defVar(writer.ncfile, body_var_x, Float64, ("n_points_$(body_idx)", "time"))
                    NetCDF.defVar(writer.ncfile, body_var_z, Float64, ("n_points_$(body_idx)", "time"))
                    writer.ncfile.dim["n_points_$(body_idx)"] = body.n_points
                    
                    # Add attributes
                    NetCDF.putatt(writer.ncfile, body_var_x, Dict("long_name" => "Flexible body $(body_idx) x-coordinates", "units" => "m"))
                    NetCDF.putatt(writer.ncfile, body_var_z, Dict("long_name" => "Flexible body $(body_idx) z-coordinates", "units" => "m"))
                end
                
                # Save Lagrangian point positions (XZ plane)
                NetCDF.putvar(writer.ncfile, body_var_x, body.X[:, 1], start=[1, snapshot_idx])
                NetCDF.putvar(writer.ncfile, body_var_z, body.X[:, 2], start=[1, snapshot_idx])  # X[:, 2] is z-coordinate
            end
        end
    end
    
    return true
end

"""
    save_body_force_coefficients!(writer, bodies, grid, state, fluid, current_time, current_iteration; kwargs...)

Save drag and lift coefficients for all bodies to NetCDF file.
"""
function save_body_force_coefficients!(writer::NetCDFWriter, 
                                      bodies::Union{RigidBodyCollection, FlexibleBodyCollection},
                                      grid::StaggeredGrid, state::SolutionState, fluid::FluidProperties,
                                      current_time::Float64, current_iteration::Int;
                                      reference_velocity::Float64 = 1.0,
                                      reference_length::Union{Nothing, Float64} = nothing,
                                      flow_direction::Vector{Float64} = [1.0, 0.0])
    
    if writer.ncfile === nothing
        initialize_netcdf_file!(writer)
    end
    
    if !should_save(writer, current_time, current_iteration)
        return false
    end
    
    snapshot_idx = writer.current_snapshot
    
    if bodies isa FlexibleBodyCollection
        n_bodies = bodies.n_bodies
        if n_bodies > 0
            # Define force coefficient variables if not already defined
            if !haskey(writer.ncfile.vars, "drag_coefficient")
                NetCDF.defVar(writer.ncfile, "drag_coefficient", Float64, ("n_flex_bodies", "time"))
                NetCDF.defVar(writer.ncfile, "lift_coefficient", Float64, ("n_flex_bodies", "time"))
                NetCDF.defVar(writer.ncfile, "drag_coefficient_pressure", Float64, ("n_flex_bodies", "time"))
                NetCDF.defVar(writer.ncfile, "drag_coefficient_viscous", Float64, ("n_flex_bodies", "time"))
                NetCDF.defVar(writer.ncfile, "force_x", Float64, ("n_flex_bodies", "time"))
                NetCDF.defVar(writer.ncfile, "force_z", Float64, ("n_flex_bodies", "time"))
                NetCDF.defVar(writer.ncfile, "center_of_pressure_x", Float64, ("n_flex_bodies", "time"))
                NetCDF.defVar(writer.ncfile, "center_of_pressure_z", Float64, ("n_flex_bodies", "time"))
                NetCDF.defVar(writer.ncfile, "instantaneous_power", Float64, ("n_flex_bodies", "time"))
                
                writer.ncfile.dim["n_flex_bodies"] = n_bodies
                
                # Add attributes
                NetCDF.putatt(writer.ncfile, "drag_coefficient", Dict("long_name" => "Drag coefficient", "units" => "dimensionless"))
                NetCDF.putatt(writer.ncfile, "lift_coefficient", Dict("long_name" => "Lift coefficient", "units" => "dimensionless"))
                NetCDF.putatt(writer.ncfile, "drag_coefficient_pressure", Dict("long_name" => "Pressure drag coefficient", "units" => "dimensionless"))
                NetCDF.putatt(writer.ncfile, "drag_coefficient_viscous", Dict("long_name" => "Viscous drag coefficient", "units" => "dimensionless"))
                NetCDF.putatt(writer.ncfile, "force_x", Dict("long_name" => "Total x-direction force", "units" => "N"))
                NetCDF.putatt(writer.ncfile, "force_z", Dict("long_name" => "Total z-direction force", "units" => "N"))
                NetCDF.putatt(writer.ncfile, "center_of_pressure_x", Dict("long_name" => "Center of pressure x-coordinate", "units" => "m"))
                NetCDF.putatt(writer.ncfile, "center_of_pressure_z", Dict("long_name" => "Center of pressure z-coordinate", "units" => "m"))
                NetCDF.putatt(writer.ncfile, "instantaneous_power", Dict("long_name" => "Instantaneous power dissipation", "units" => "W"))
                
                # Save reference conditions as global attributes
                NetCDF.putatt(writer.ncfile, "global", Dict(
                    "reference_velocity" => reference_velocity,
                    "flow_direction_x" => flow_direction[1],
                    "flow_direction_z" => flow_direction[2]
                ))
            end
            
            # Compute coefficients for all bodies
            Cd_data = Float64[]
            Cl_data = Float64[]
            Cd_pressure_data = Float64[]
            Cd_viscous_data = Float64[]
            Fx_data = Float64[]
            Fz_data = Float64[]
            cop_x_data = Float64[]
            cop_z_data = Float64[]
            power_data = Float64[]
            
            for body in bodies.bodies
                # Use body length as reference if not specified
                ref_length = reference_length !== nothing ? reference_length : body.length
                
                # Compute force coefficients
                coeffs = compute_drag_lift_coefficients(body, grid, state, fluid;
                                                       reference_velocity=reference_velocity,
                                                       reference_length=ref_length,
                                                       flow_direction=flow_direction)
                
                # Compute instantaneous power
                power = compute_instantaneous_power(body, grid, state, fluid)
                
                # Store data
                push!(Cd_data, coeffs.Cd)
                push!(Cl_data, coeffs.Cl)
                push!(Cd_pressure_data, coeffs.Cd_pressure)
                push!(Cd_viscous_data, coeffs.Cd_viscous)
                push!(Fx_data, coeffs.Fx)
                push!(Fz_data, coeffs.Fz)
                push!(cop_x_data, coeffs.center_of_pressure[1])
                push!(cop_z_data, coeffs.center_of_pressure[2])
                push!(power_data, power)
            end
            
            # Save all coefficient data
            NetCDF.putvar(writer.ncfile, "drag_coefficient", Cd_data, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "lift_coefficient", Cl_data, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "drag_coefficient_pressure", Cd_pressure_data, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "drag_coefficient_viscous", Cd_viscous_data, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "force_x", Fx_data, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "force_z", Fz_data, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "center_of_pressure_x", cop_x_data, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "center_of_pressure_z", cop_z_data, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "instantaneous_power", power_data, start=[1, snapshot_idx])
            
            println("Saved force coefficients for $(n_bodies) flexible bodies at time $(current_time)")
        end
    end
    
    return true
end

"""
    save_complete_snapshot!(writer, state, bodies, grid, fluid, current_time, current_iteration; kwargs...)

Save a complete snapshot including flow field, body data, and force coefficients.
"""
function save_complete_snapshot!(writer::NetCDFWriter, state::SolutionState,
                                bodies::Union{Nothing, RigidBodyCollection, FlexibleBodyCollection},
                                grid::StaggeredGrid, fluid::FluidProperties,
                                current_time::Float64, current_iteration::Int; 
                                save_coefficients::Bool = true, kwargs...)
    
    # Save flow field
    saved_flow = save_snapshot!(writer, state, current_time, current_iteration)
    
    if saved_flow && bodies !== nothing
        # Save body kinematics
        save_body_data!(writer, bodies, current_time, current_iteration)
        
        # Save force coefficients if requested
        if save_coefficients
            if bodies isa FlexibleBodyCollection
                save_body_force_coefficients!(writer, bodies, grid, state, fluid, 
                                            current_time, current_iteration; kwargs...)
            elseif bodies isa RigidBodyCollection
                try
                    save_rigid_body_coefficients_series!(writer, bodies, grid, state, fluid,
                                                         current_time, current_iteration; kwargs...)
                catch e
                    @warn "Rigid-body coefficients not saved: $e"
                end
                try
                    save_rigid_body_forces_series!(writer, bodies, grid, state, fluid,
                                                   current_time, current_iteration)
                catch e
                    @warn "Rigid-body forces not saved: $e"
                end
            end
        end
    end
    
    return saved_flow
end

# ============================================================================
# Dedicated Flexible Body Position Saving Functions
# ============================================================================

"""
    save_flexible_body_positions!(writer, bodies, current_time, current_iteration; options...)

Save only flexible body positions to NetCDF with enhanced options.

# Arguments
- `writer::NetCDFWriter`: NetCDF writer instance
- `bodies::FlexibleBodyCollection`: Collection of flexible bodies
- `current_time::Float64`: Current simulation time
- `current_iteration::Int`: Current iteration number

# Keywords
- `save_velocities::Bool = false`: Also save body point velocities
- `save_accelerations::Bool = false`: Also save body point accelerations  
- `save_curvature::Bool = false`: Save local curvature at each point
- `save_forces::Bool = false`: Save forces at each Lagrangian point
- `save_material_properties::Bool = false`: Save tension, bending moments
"""
function save_flexible_body_positions!(writer::NetCDFWriter, 
                                      bodies::FlexibleBodyCollection,
                                      current_time::Float64, current_iteration::Int;
                                      save_velocities::Bool = false,
                                      save_accelerations::Bool = false,
                                      save_curvature::Bool = false,
                                      save_forces::Bool = false,
                                      save_material_properties::Bool = false)
    
    if writer.ncfile === nothing
        initialize_netcdf_file!(writer)
    end
    
    if !should_save(writer, current_time, current_iteration)
        return false
    end
    # Rotate files if needed
    if writer.current_snapshot >= writer.config.max_snapshots_per_file
        create_new_file!(writer)
    end

    # Advance snapshot index for position-only path
    writer.current_snapshot += 1
    snapshot_idx = writer.current_snapshot
    n_bodies = bodies.n_bodies
    
    if n_bodies == 0
        return false
    end
    
    # Ensure time coordinate is written
    NetCDF.putvar(writer.ncfile, "time", [current_time], start=[snapshot_idx])

    # For each flexible body, save detailed position data
    for (body_idx, body) in enumerate(bodies.bodies)
        body_id = body_idx
        
        # Basic position variables (always saved)
        pos_x_var = "flexible_body_$(body_id)_x"
        pos_z_var = "flexible_body_$(body_id)_z"
        
        # Define variables if not already defined
        if !haskey(writer.ncfile.vars, pos_x_var)
            dim_name = "n_points_body_$(body_id)"
            if !haskey(writer.ncfile.dim, dim_name)
                NetCDF.defDim(writer.ncfile, dim_name, body.n_points)
            end
            NetCDF.defVar(writer.ncfile, pos_x_var, Float64, (dim_name, "time"))
            NetCDF.defVar(writer.ncfile, pos_z_var, Float64, (dim_name, "time"))

            NetCDF.putatt(writer.ncfile, pos_x_var, Dict(
                "long_name" => "Flexible body $(body_id) x-coordinates",
                "units" => "m",
                "description" => "Lagrangian point positions in x-direction",
                "body_length" => body.length,
                "n_points" => body.n_points,
                "material_type" => "flexible"
            ))

            NetCDF.putatt(writer.ncfile, pos_z_var, Dict(
                "long_name" => "Flexible body $(body_id) z-coordinates",
                "units" => "m",
                "description" => "Lagrangian point positions in z-direction (vertical)",
                "body_length" => body.length,
                "n_points" => body.n_points
            ))

            if save_velocities
                vel_x_var = "flexible_body_$(body_id)_vel_x"
                vel_z_var = "flexible_body_$(body_id)_vel_z"
                NetCDF.defVar(writer.ncfile, vel_x_var, Float64, (dim_name, "time"))
                NetCDF.defVar(writer.ncfile, vel_z_var, Float64, (dim_name, "time"))
                NetCDF.putatt(writer.ncfile, vel_x_var, Dict("long_name" => "Body velocity x-component", "units" => "m/s"))
                NetCDF.putatt(writer.ncfile, vel_z_var, Dict("long_name" => "Body velocity z-component", "units" => "m/s"))
            end

            if save_accelerations
                acc_x_var = "flexible_body_$(body_id)_acc_x"
                acc_z_var = "flexible_body_$(body_id)_acc_z"
                NetCDF.defVar(writer.ncfile, acc_x_var, Float64, (dim_name, "time"))
                NetCDF.defVar(writer.ncfile, acc_z_var, Float64, (dim_name, "time"))
                NetCDF.putatt(writer.ncfile, acc_x_var, Dict("long_name" => "Body acceleration x-component", "units" => "m/s²"))
                NetCDF.putatt(writer.ncfile, acc_z_var, Dict("long_name" => "Body acceleration z-component", "units" => "m/s²"))
            end

            if save_curvature
                curv_var = "flexible_body_$(body_id)_curvature"
                NetCDF.defVar(writer.ncfile, curv_var, Float64, (dim_name, "time"))
                NetCDF.putatt(writer.ncfile, curv_var, Dict("long_name" => "Local curvature", "units" => "1/m"))
            end

            if save_forces
                force_x_var = "flexible_body_$(body_id)_force_x"
                force_z_var = "flexible_body_$(body_id)_force_z"
                NetCDF.defVar(writer.ncfile, force_x_var, Float64, (dim_name, "time"))
                NetCDF.defVar(writer.ncfile, force_z_var, Float64, (dim_name, "time"))
                NetCDF.putatt(writer.ncfile, force_x_var, Dict("long_name" => "Lagrangian force x-component", "units" => "N/m"))
                NetCDF.putatt(writer.ncfile, force_z_var, Dict("long_name" => "Lagrangian force z-component", "units" => "N/m"))
            end

            if save_material_properties
                tension_var = "flexible_body_$(body_id)_tension"
                NetCDF.defVar(writer.ncfile, tension_var, Float64, (dim_name, "time"))
                NetCDF.putatt(writer.ncfile, tension_var, Dict("long_name" => "Internal tension", "units" => "N"))
            end
        end
        
        # Save position data (append along time)
        NetCDF.putvar(writer.ncfile, pos_x_var, body.X[:, 1], start=[1, snapshot_idx], count=[body.n_points, 1])
        NetCDF.putvar(writer.ncfile, pos_z_var, body.X[:, 2], start=[1, snapshot_idx], count=[body.n_points, 1])
        
        # Save optional data
        if save_velocities
            vel_x_data = zeros(body.n_points)
            vel_z_data = zeros(body.n_points)
            # Compute velocities using finite differences or stored data
            if hasfield(typeof(body), :X_old) && body.X_old !== nothing
                dt = 0.001  # This should be passed as parameter in real implementation
                for i = 1:body.n_points
                    vel_x_data[i] = (body.X[i, 1] - body.X_old[i, 1]) / dt
                    vel_z_data[i] = (body.X[i, 2] - body.X_old[i, 2]) / dt
                end
            end
            NetCDF.putvar(writer.ncfile, "flexible_body_$(body_id)_vel_x", vel_x_data, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "flexible_body_$(body_id)_vel_z", vel_z_data, start=[1, snapshot_idx])
        end
        
        if save_accelerations
            # Compute accelerations using finite differences
            acc_x_data = zeros(body.n_points)
            acc_z_data = zeros(body.n_points)
            if body.X_old !== nothing && body.X_prev !== nothing
                dt = 0.001  # This should be passed as parameter in real implementation
                for i = 1:body.n_points
                    # Second-order backward difference for acceleration
                    acc_x_data[i] = (body.X[i, 1] - 2*body.X_old[i, 1] + body.X_prev[i, 1]) / dt^2
                    acc_z_data[i] = (body.X[i, 2] - 2*body.X_old[i, 2] + body.X_prev[i, 2]) / dt^2
                end
            end
            NetCDF.putvar(writer.ncfile, "flexible_body_$(body_id)_acc_x", acc_x_data, start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "flexible_body_$(body_id)_acc_z", acc_z_data, start=[1, snapshot_idx])
        end
        
        if save_curvature
            compute_curvature!(body)  # Update curvature
            NetCDF.putvar(writer.ncfile, "flexible_body_$(body_id)_curvature", body.curvature, start=[1, snapshot_idx])
        end
        
        if save_forces
            NetCDF.putvar(writer.ncfile, "flexible_body_$(body_id)_force_x", body.force[:, 1], start=[1, snapshot_idx])
            NetCDF.putvar(writer.ncfile, "flexible_body_$(body_id)_force_z", body.force[:, 2], start=[1, snapshot_idx])
        end
        
        if save_material_properties
            NetCDF.putvar(writer.ncfile, "flexible_body_$(body_id)_tension", body.tension, start=[1, snapshot_idx])
        end
    end
    
    writer.last_save_time = current_time
    writer.last_save_iteration = current_iteration
    println("Saved detailed positions for $(n_bodies) flexible bodies at time $(current_time)")
    return true
end

# ============================================================================
# Rigid-body coefficient series to a separate NetCDF file
# ============================================================================

const _RIGID_COEFF_WRITERS = Dict{String, Any}()
const _RIGID_FORCE_WRITERS = Dict{String, Any}()

mutable struct RigidCoeffWriter
    filepath::String
    base_filepath::String
    n_bodies::Int
    max_snapshots::Int
    current_snapshot::Int
    file_index::Int
    ncfile::Union{Nothing, NetCDF.NcFile}
end

function _create_rigid_coeff_writer(filepath::String, bodies::RigidBodyCollection; max_snapshots::Int=1000)
    w = RigidCoeffWriter(filepath, filepath, bodies.n_bodies, max_snapshots, 0, 0, nothing)
    _init_rigid_coeff_file!(w)
    return w
end

function _init_rigid_coeff_file!(w::RigidCoeffWriter)
    if isfile(w.filepath); rm(w.filepath); end
    n_b = w.n_bodies
    tdim = NetCDF.NcDim("time", w.max_snapshots)
    bdim = NetCDF.NcDim("n_bodies", n_b)
    vars = NetCDF.NcVar[
        NetCDF.NcVar("time", [tdim]; t=Float64),
        NetCDF.NcVar("Cd", [bdim, tdim]; t=Float64),
        NetCDF.NcVar("Cl", [bdim, tdim]; t=Float64),
        NetCDF.NcVar("Cd_pressure", [bdim, tdim]; t=Float64),
        NetCDF.NcVar("Cd_viscous", [bdim, tdim]; t=Float64),
        NetCDF.NcVar("Fx", [bdim, tdim]; t=Float64),
        NetCDF.NcVar("Fz", [bdim, tdim]; t=Float64),
    ]
    nc = NetCDF.create(w.filepath, vars)
    NetCDF.putatt(nc, "global", Dict("title"=>"Rigid body force coefficients", "n_bodies"=>n_b))
    w.ncfile = nc
    return nc
end

function _rollover!(w::RigidCoeffWriter)
    if w.current_snapshot >= w.max_snapshots
        if w.ncfile !== nothing
            try NetCDF.close(w.ncfile) catch; end
            w.ncfile = nothing
        end
        w.file_index += 1
        base, ext = splitext(w.base_filepath)
        w.filepath = string(base, "_", w.file_index, ext)
        w.current_snapshot = 0
        _init_rigid_coeff_file!(w)
        println("Rigid coefficients: created new file $(w.filepath)")
    end
end

function _get_rigid_coeff_writer(writer::NetCDFWriter, bodies::RigidBodyCollection)
    base, ext = splitext(writer.filepath)
    coeff_path = string(base, "_coeffs", ext)
    key = coeff_path
    if !haskey(_RIGID_COEFF_WRITERS, key)
        _RIGID_COEFF_WRITERS[key] = _create_rigid_coeff_writer(coeff_path, bodies; max_snapshots=writer.config.max_snapshots_per_file)
    end
    return _RIGID_COEFF_WRITERS[key]
end

function save_rigid_body_coefficients_series!(writer::NetCDFWriter,
                                             bodies::RigidBodyCollection,
                                             grid::StaggeredGrid, state::SolutionState,
                                             fluid::FluidProperties,
                                             current_time::Float64, current_iteration::Int;
                                             reference_velocity::Float64=writer.config.reference_velocity,
                                             flow_direction::Vector{Float64}=writer.config.flow_direction)
    cw = _get_rigid_coeff_writer(writer, bodies)
    coeffs = [compute_drag_lift_coefficients(body, grid, state, fluid;
                 reference_velocity=reference_velocity,
                 reference_length=body.shape isa Circle ? 2*(body.shape::Circle).radius : 1.0,
                 flow_direction=flow_direction) for body in bodies.bodies]
    _rollover!(cw)
    cw.current_snapshot += 1
    idx = cw.current_snapshot
    NetCDF.putvar(cw.ncfile, "time", [current_time], start=[idx])
    NetCDF.putvar(cw.ncfile, "Cd", [c.Cd for c in coeffs], start=[1, idx])
    NetCDF.putvar(cw.ncfile, "Cl", [c.Cl for c in coeffs], start=[1, idx])
    NetCDF.putvar(cw.ncfile, "Cd_pressure", [c.Cd_pressure for c in coeffs], start=[1, idx])
    NetCDF.putvar(cw.ncfile, "Cd_viscous", [c.Cd_viscous for c in coeffs], start=[1, idx])
    NetCDF.putvar(cw.ncfile, "Fx", [c.Fx for c in coeffs], start=[1, idx])
    NetCDF.putvar(cw.ncfile, "Fz", [c.Fz for c in coeffs], start=[1, idx])
    println("Saved rigid-body coefficients snapshot $(idx) at t=$(current_time)")
    return true
end

# ============================================================================
# Rigid-body forces and torque series (separate file)
# ============================================================================

mutable struct RigidForceWriter
    filepath::String
    base_filepath::String
    n_bodies::Int
    max_snapshots::Int
    current_snapshot::Int
    file_index::Int
    ncfile::Union{Nothing, NetCDF.NcFile}
end

function _create_rigid_force_writer(filepath::String, bodies::RigidBodyCollection; max_snapshots::Int=1000)
    w = RigidForceWriter(filepath, filepath, bodies.n_bodies, max_snapshots, 0, 0, nothing)
    _init_rigid_force_file!(w)
    return w
end

function _init_rigid_force_file!(w::RigidForceWriter)
    if isfile(w.filepath); rm(w.filepath); end
    n_b = w.n_bodies
    tdim = NetCDF.NcDim("time", w.max_snapshots)
    bdim = NetCDF.NcDim("n_bodies", n_b)
    vars = NetCDF.NcVar[
        NetCDF.NcVar("time", [tdim]; t=Float64),
        NetCDF.NcVar("Fx", [bdim, tdim]; t=Float64),
        NetCDF.NcVar("Fz", [bdim, tdim]; t=Float64),
        NetCDF.NcVar("Ty", [bdim, tdim]; t=Float64),
    ]
    nc = NetCDF.create(w.filepath, vars)
    NetCDF.putatt(nc, "global", Dict("title"=>"Rigid body forces and torque (2D XZ)", "n_bodies"=>n_b))
    w.ncfile = nc
    return nc
end

function _rollover!(w::RigidForceWriter)
    if w.current_snapshot >= w.max_snapshots
        if w.ncfile !== nothing
            try NetCDF.close(w.ncfile) catch; end
            w.ncfile = nothing
        end
        w.file_index += 1
        base, ext = splitext(w.base_filepath)
        w.filepath = string(base, "_", w.file_index, ext)
        w.current_snapshot = 0
        _init_rigid_force_file!(w)
        println("Rigid forces: created new file $(w.filepath)")
    end
end

function _get_rigid_force_writer(writer::NetCDFWriter, bodies::RigidBodyCollection)
    base, ext = splitext(writer.filepath)
    fpath = string(base, "_forces", ext)
    key = fpath
    if !haskey(_RIGID_FORCE_WRITERS, key)
        _RIGID_FORCE_WRITERS[key] = _create_rigid_force_writer(fpath, bodies; max_snapshots=writer.config.max_snapshots_per_file)
    end
    return _RIGID_FORCE_WRITERS[key]
end

function save_rigid_body_forces_series!(writer::NetCDFWriter,
                                       bodies::RigidBodyCollection,
                                       grid::StaggeredGrid, state::SolutionState,
                                       fluid::FluidProperties,
                                       current_time::Float64, current_iteration::Int)
    # Only 2D XZ currently; extend to 3D as needed
    if grid.grid_type != TwoDimensional
        @warn "Rigid-body force writer currently supports 2D XZ only"
        return false
    end
    fw = _get_rigid_force_writer(writer, bodies)
    _rollover!(fw)
    fw.current_snapshot += 1
    idx = fw.current_snapshot
    # Compute forces for each body
    Fx = Float64[]; Fz = Float64[]; Ty = Float64[]
    for body in bodies.bodies
        F, torque = compute_body_forces_2d(body, grid, state, fluid)
        push!(Fx, F[1]); push!(Fz, F[2]); push!(Ty, torque)
    end
    NetCDF.putvar(fw.ncfile, "time", [current_time], start=[idx])
    NetCDF.putvar(fw.ncfile, "Fx", Fx, start=[1, idx])
    NetCDF.putvar(fw.ncfile, "Fz", Fz, start=[1, idx])
    NetCDF.putvar(fw.ncfile, "Ty", Ty, start=[1, idx])
    println("Saved rigid-body forces snapshot $(idx) at t=$(current_time)")
    return true
end

"""
    create_position_only_writer(filepath, grid, bodies; kwargs...)

Create a NetCDF writer specifically for body position tracking with minimal overhead.

# Arguments
- `filepath::String`: Output NetCDF file path
- `grid::StaggeredGrid`: Computational grid (for metadata)
- `bodies::FlexibleBodyCollection`: Bodies to track

# Keywords
- `time_interval::Float64 = 0.01`: Time interval between saves
- `iteration_interval::Int = 1`: Iteration interval between saves  
- `save_mode::Symbol = :time_interval`: Save trigger mode
- `max_snapshots::Int = 1000`: Maximum snapshots per file
- `detailed_tracking::Bool = false`: Save velocities, curvature, etc.
"""
function create_position_only_writer(filepath::String, grid::StaggeredGrid, 
                                   bodies::FlexibleBodyCollection;
                                   time_interval::Float64 = 0.01,
                                   iteration_interval::Int = 1,
                                   save_mode::Symbol = :time_interval,
                                   max_snapshots::Int = 1000,
                                   detailed_tracking::Bool = false)
    
    # Create lightweight config for position-only tracking
    config = NetCDFConfig(basename(filepath);
        max_snapshots_per_file = max_snapshots,
        save_mode = save_mode,
        time_interval = time_interval,
        iteration_interval = iteration_interval,
        save_flow_field = false,              # Don't save u, w, p
        save_body_positions = true,           # Save positions
        save_force_coefficients = false      # Don't save coefficients
    )
    
    writer = NetCDFWriter(filepath, grid, config)
    initialize_position_only_file!(writer, bodies)
    
    println("Created position-only NetCDF writer:")
    println("  • File: $filepath")
    println("  • Save mode: $save_mode")
    println("  • Time interval: $time_interval")
    println("  • Iteration interval: $iteration_interval")
    println("  • Max snapshots: $max_snapshots")
    println("  • Bodies to track: $(bodies.n_bodies)")
    
    return writer
end

"""
    save_body_kinematics_snapshot!(writer, bodies, current_time, current_iteration, dt)

Save comprehensive kinematics (position, velocity, acceleration) for all bodies.
"""
function save_body_kinematics_snapshot!(writer::NetCDFWriter,
                                       bodies::FlexibleBodyCollection,
                                       current_time::Float64, 
                                       current_iteration::Int,
                                       dt::Float64)
    
    return save_flexible_body_positions!(writer, bodies, current_time, current_iteration;
                                        save_velocities = true,
                                        save_accelerations = true,
                                        save_curvature = true)
end

"""
    save_body_positions_only!(writer, bodies, current_time, current_iteration)

Save only positions (minimal data) for lightweight tracking.
"""
function save_body_positions_only!(writer::NetCDFWriter,
                                  bodies::FlexibleBodyCollection,
                                  current_time::Float64,
                                  current_iteration::Int)
    
    return save_flexible_body_positions!(writer, bodies, current_time, current_iteration;
                                        save_velocities = false,
                                        save_accelerations = false,
                                        save_curvature = false,
                                        save_forces = false,
                                        save_material_properties = false)
end

function initialize_position_only_file!(writer::NetCDFWriter, bodies::FlexibleBodyCollection)
    # Remove existing file if it exists
    if isfile(writer.filepath)
        rm(writer.filepath)
    end
    # Define time dim and per-body dims and variables
    time_dim = NetCDF.NcDim("time", writer.config.max_snapshots_per_file)
    vars = NetCDF.NcVar[ NetCDF.NcVar("time", [time_dim]; t=Float64) ]
    for (i, body) in enumerate(bodies.bodies)
        dim_name = "n_points_body_$(i)"
        points_dim = NetCDF.NcDim(dim_name, body.n_points)
        push!(vars, NetCDF.NcVar("flexible_body_$(i)_x", [points_dim, time_dim]; t=Float64))
        push!(vars, NetCDF.NcVar("flexible_body_$(i)_z", [points_dim, time_dim]; t=Float64))
    end
    nc = NetCDF.create(writer.filepath, vars)
    writer.ncfile = nc
    return nc
end

function close_netcdf!(writer::NetCDFWriter)
    if writer.ncfile !== nothing
        try
            NetCDF.close(writer.ncfile)
            writer.ncfile = nothing
            println("Closed NetCDF file: $(writer.filepath)")
        catch e
            @error "Failed to close NetCDF file: $e"
        end
    end
end

# Alias for compatibility with exports
function close!(writer::NetCDFWriter)
    close_netcdf!(writer)
end

function create_new_file!(writer::NetCDFWriter)
    # Close current file and create a new one with numeric suffix
    close_netcdf!(writer)
    # Determine next file index
    writer.file_index += 1
    base_path, ext = splitext(writer.base_filepath)
    # First rollover goes to _1, then _2, ...
    writer.filepath = "$(base_path)_$(writer.file_index)$(ext)"
    
    # Reset snapshot counter
    writer.current_snapshot = 0
    
    # Initialize new file
    initialize_netcdf_file!(writer)
    
    println("Created new NetCDF file: $(writer.filepath)")
end

# Convenience function for setting up output
function setup_netcdf_output(output_dir::String, simulation_name::String, grid::StaggeredGrid;
                            max_snapshots::Int=100,
                            time_interval::Float64=0.1,
                            iteration_interval::Int=10)
    
    # Create output directory if it doesn't exist
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Create output filename with timestamp
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "$(simulation_name)_$(timestamp).nc"
    filepath = joinpath(output_dir, filename)
    
    writer = NetCDFWriter(filepath, grid;
                         max_snapshots=max_snapshots,
                         time_interval=time_interval,
                         iteration_interval=iteration_interval)
    
    return writer
end

# Function to read NetCDF data back (for post-processing)
function read_netcdf_data(filepath::String)
    ncfile = NetCDF.open(filepath)
    
    # Read grid information
    x = NetCDF.readvar(ncfile, "x")
    y = haskey(ncfile.vars, "y") ? NetCDF.readvar(ncfile, "y") : nothing
    z = haskey(ncfile.vars, "z") ? NetCDF.readvar(ncfile, "z") : nothing
    time_data = NetCDF.readvar(ncfile, "time")
    
    # Read solution variables
    u_data = haskey(ncfile.vars, "u") ? NetCDF.readvar(ncfile, "u") : nothing
    v_data = haskey(ncfile.vars, "v") ? NetCDF.readvar(ncfile, "v") : nothing
    p_data = haskey(ncfile.vars, "p") ? NetCDF.readvar(ncfile, "p") : nothing
    # 2D stores vertical velocity as "w"
    w_data = haskey(ncfile.vars, "w") ? NetCDF.readvar(ncfile, "w") : nothing
    
    NetCDF.close(ncfile)
    
    return Dict(
        "x" => x,
        "y" => y,
        "z" => z,
        "time" => time_data,
        "u" => u_data,
        "v" => v_data,
        "w" => w_data,
        "p" => p_data
    )
end

# Data validation functions
function validate_state_data(state::SolutionState, grid::StaggeredGrid)
    """Validate that solution state has correct dimensions and no NaN/Inf values."""
    
    # Check dimensions by grid type
    if grid.grid_type == TwoDimensional
        if size(state.u, 1) != grid.nx + 1 || size(state.u, 2) != grid.nz
            @error "u-velocity dimensions mismatch: expected $(grid.nx+1)×$(grid.nz), got $(size(state.u))"
            return false
        end
        if size(state.v, 1) != grid.nx || size(state.v, 2) != grid.nz + 1
            @error "v-velocity dimensions mismatch (2D): expected $(grid.nx)×$(grid.nz+1), got $(size(state.v))"
            return false
        end
        if size(state.p, 1) != grid.nx || size(state.p, 2) != grid.nz
            @error "Pressure dimensions mismatch: expected $(grid.nx)×$(grid.nz), got $(size(state.p))"
            return false
        end
    else
        if size(state.u, 1) != grid.nx + 1 || size(state.u, 2) != grid.ny || size(state.u, 3) != grid.nz
            @error "u-velocity dimensions mismatch (3D)"
            return false
        end
        if size(state.v, 1) != grid.nx || size(state.v, 2) != grid.ny + 1 || size(state.v, 3) != grid.nz
            @error "v-velocity dimensions mismatch (3D)"
            return false
        end
        if size(state.p, 1) != grid.nx || size(state.p, 2) != grid.ny || size(state.p, 3) != grid.nz
            @error "Pressure dimensions mismatch (3D)"
            return false
        end
    end
    
    # Check for NaN/Inf values
    if any(isnan, state.u) || any(isinf, state.u)
        @error "u-velocity contains NaN or Inf values"
        return false
    end
    
    if any(isnan, state.v) || any(isinf, state.v)
        @error "v-velocity contains NaN or Inf values"
        return false
    end
    
    if any(isnan, state.p) || any(isinf, state.p)
        @error "Pressure contains NaN or Inf values"
        return false
    end
    
    # Check 3D if applicable
    if grid.grid_type == ThreeDimensional
        if size(state.w, 1) != grid.nx || size(state.w, 2) != grid.ny || size(state.w, 3) != grid.nz + 1
            @error "w-velocity dimensions mismatch for 3D"
            return false
        end
        
        if any(isnan, state.w) || any(isinf, state.w)
            @error "w-velocity contains NaN or Inf values"
            return false
        end
    end

    return true
end

function validate_body_data(bodies::Union{RigidBodyCollection, FlexibleBodyCollection})
    """Validate body data before saving."""
    
    if bodies isa FlexibleBodyCollection
        for (i, body) in enumerate(bodies.bodies)
            if any(isnan, body.X) || any(isinf, body.X)
                @error "Flexible body $(i) positions contain NaN or Inf values"
                return false
            end
            
            if any(isnan, body.force) || any(isinf, body.force)
                @error "Flexible body $(i) forces contain NaN or Inf values"
                return false
            end
        end
    elseif bodies isa RigidBodyCollection
        for (i, body) in enumerate(bodies.bodies)
            if any(isnan, body.center) || any(isinf, body.center)
                @error "Rigid body $(i) center contains NaN or Inf values"
                return false
            end
            
            if any(isnan, body.velocity) || any(isinf, body.velocity)
                @error "Rigid body $(i) velocity contains NaN or Inf values"
                return false
            end
        end
    end
    
    return true
end

function write_solution!(writer::NetCDFWriter, 
                        state::SolutionState, 
                        bodies::Union{Nothing, RigidBodyCollection, FlexibleBodyCollection},
                        grid::StaggeredGrid, 
                        fluid::FluidProperties,
                        current_time::Float64, 
                        current_iteration::Int; 
                        validate_data::Bool = true, kwargs...)
    """
    Main function to write complete solution data to NetCDF.
    Validates data and handles all output types.
    """
    
    # Data validation
    if validate_data
        if !validate_state_data(state, grid)
            @error "State data validation failed - not saving"
            return false
        end
        
        if bodies !== nothing && !validate_body_data(bodies)
            @error "Body data validation failed - not saving"
            return false
        end
    end
    
    # Save complete snapshot
    return save_complete_snapshot!(writer, state, bodies, grid, fluid, 
                                  current_time, current_iteration; kwargs...)
end

# =============================================================================
# MPI-aware write (2D XZ plane): gather local fields to root and write globally
# =============================================================================
function write_solution!(writer::NetCDFWriter,
                        state::MPISolutionState2D,
                        bodies::Union{Nothing, RigidBodyCollection, FlexibleBodyCollection},
                        grid::StaggeredGrid,
                        fluid::FluidProperties,
                        current_time::Float64,
                        current_iteration::Int; kwargs...)
    HAS_MPI_WRITER || return false
    decomp = state.decomp
    comm = decomp.comm
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # Local interior indices
    ils, ile = decomp.i_local_start, decomp.i_local_end
    jls, jle = decomp.j_local_start, decomp.j_local_end
    is, ie = decomp.i_start, decomp.i_end
    js, je = decomp.j_start, decomp.j_end
    nxg, nzg = decomp.nx_global, decomp.nz_global

    # Include domain boundary faces (staggered)
    u_i_hi = ile + (ie == nxg ? 1 : 0)
    v_j_hi = jle + (je == nzg ? 1 : 0)

    u_blk = @view state.u[ils:u_i_hi, jls:jle]
    v_blk = @view state.v[ils:ile, jls:v_j_hi]
    p_blk = @view state.p[ils:ile, jls:jle]

    # Root assembles global arrays and writes
    if rank == 0
        # Ensure writer file is ready
        initialize_netcdf_file!(writer)
        u_glob = zeros(Float64, writer.grid.nx + 1, writer.grid.nz)
        v_glob = zeros(Float64, writer.grid.nx, writer.grid.nz + 1)
        p_glob = zeros(Float64, writer.grid.nx, writer.grid.nz)

        # Place rank 0
        u_glob[is:ie + (ie == nxg ? 1 : 0), js:je] .= u_blk
        v_glob[is:ie, js:je + (je == nzg ? 1 : 0)] .= v_blk
        p_glob[is:ie, js:je] .= p_blk

        # Receive others
        for src in 1:size-1
            hdr = Array{Int}(undef, 4)
            MPI.Recv!(hdr, src, 9100, comm)
            isrc, iesrc, jsrc, jesrc = hdr...
            u_count_i = iesrc - isrc + 1 + (iesrc == nxg ? 1 : 0)
            v_count_j = jesrc - jsrc + 1 + (jesrc == nzg ? 1 : 0)
            u_recv = Array{Float64}(undef, u_count_i, jesrc - jsrc + 1)
            v_recv = Array{Float64}(undef, iesrc - isrc + 1, v_count_j)
            p_recv = Array{Float64}(undef, iesrc - isrc + 1, jesrc - jsrc + 1)
            MPI.Recv!(u_recv, src, 9101, comm)
            MPI.Recv!(v_recv, src, 9102, comm)
            MPI.Recv!(p_recv, src, 9103, comm)
            u_glob[isrc:iesrc + (iesrc == nxg ? 1 : 0), jsrc:jesrc] .= u_recv
            v_glob[isrc:iesrc, jsrc:jesrc + (jesrc == nzg ? 1 : 0)] .= v_recv
            p_glob[isrc:iesrc, jsrc:jesrc] .= p_recv
        end

        # Build global state and write
        gstate = SolutionState2D(writer.grid.nx, writer.grid.nz)
        gstate.u .= u_glob
        gstate.v .= v_glob
        gstate.p .= p_glob
        gstate.t = current_time
        gstate.step = current_iteration
        return save_complete_snapshot!(writer, gstate, bodies, writer.grid, fluid, current_time, current_iteration; kwargs...)

    else
        # Send local header and data
        hdr = Int[is, ie, js, je]
        MPI.Send(hdr, 0, 9100, comm)
        MPI.Send(Array(u_blk), 0, 9101, comm)
        MPI.Send(Array(v_blk), 0, 9102, comm)
        MPI.Send(Array(p_blk), 0, 9103, comm)
        return true
    end
end

function write_solution!(writer::NetCDFWriter,
                        state::MPISolutionState3D,
                        bodies::Union{Nothing, RigidBodyCollection, FlexibleBodyCollection},
                        grid::StaggeredGrid,
                        fluid::FluidProperties,
                        current_time::Float64,
                        current_iteration::Int; kwargs...)
    HAS_MPI_WRITER || return false
    decomp = state.decomp
    comm = decomp.comm
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # Global ranges
    is, ie = decomp.i_start, decomp.i_end
    js, je = decomp.j_start, decomp.j_end
    ks, ke = decomp.k_start, decomp.k_end
    nxg, nyg, nzg = decomp.nx_global, decomp.ny_global, decomp.nz_global

    # Local sizes (assume no ghosts for 3D local arrays)
    nx_loc, ny_loc, nz_loc = size(state.p)
    # Staggered face inclusion on domain boundaries only
    u_i_hi_loc = nx_loc + (ie == nxg ? 1 : 0)
    v_j_hi_loc = ny_loc + (je == nyg ? 1 : 0)
    w_k_hi_loc = nz_loc + (ke == nzg ? 1 : 0)

    u_blk = @view state.u[1:u_i_hi_loc, 1:ny_loc, 1:nz_loc]
    v_blk = @view state.v[1:nx_loc, 1:v_j_hi_loc, 1:nz_loc]
    w_blk = @view state.w[1:nx_loc, 1:ny_loc, 1:w_k_hi_loc]
    p_blk = @view state.p[1:nx_loc, 1:ny_loc, 1:nz_loc]

    if rank == 0
        initialize_netcdf_file!(writer)
        u_glob = zeros(Float64, writer.grid.nx + 1, writer.grid.ny, writer.grid.nz)
        v_glob = zeros(Float64, writer.grid.nx, writer.grid.ny + 1, writer.grid.nz)
        w_glob = zeros(Float64, writer.grid.nx, writer.grid.ny, writer.grid.nz + 1)
        p_glob = zeros(Float64, writer.grid.nx, writer.grid.ny, writer.grid.nz)

        u_glob[is:ie + (ie == nxg ? 1 : 0), js:je, ks:ke] .= u_blk
        v_glob[is:ie, js:je + (je == nyg ? 1 : 0), ks:ke] .= v_blk
        w_glob[is:ie, js:je, ks:ke + (ke == nzg ? 1 : 0)] .= w_blk
        p_glob[is:ie, js:je, ks:ke] .= p_blk

        # Receive from others
        for src in 1:size-1
            hdr = Array{Int}(undef, 6)
            MPI.Recv!(hdr, src, 9200, comm)
            isrc, iesrc, jsrc, jesrc, ksrc, kesrc = hdr...
            u_count_i = iesrc - isrc + 1 + (iesrc == nxg ? 1 : 0)
            v_count_j = jesrc - jsrc + 1 + (jesrc == nyg ? 1 : 0)
            w_count_k = kesrc - ksrc + 1 + (kesrc == nzg ? 1 : 0)
            u_recv = Array{Float64}(undef, u_count_i, jesrc - jsrc + 1, kesrc - ksrc + 1)
            v_recv = Array{Float64}(undef, iesrc - isrc + 1, v_count_j, kesrc - ksrc + 1)
            w_recv = Array{Float64}(undef, iesrc - isrc + 1, jesrc - jsrc + 1, w_count_k)
            p_recv = Array{Float64}(undef, iesrc - isrc + 1, jesrc - jsrc + 1, kesrc - ksrc + 1)
            MPI.Recv!(u_recv, src, 9201, comm)
            MPI.Recv!(v_recv, src, 9202, comm)
            MPI.Recv!(w_recv, src, 9203, comm)
            MPI.Recv!(p_recv, src, 9204, comm)
            u_glob[isrc:iesrc + (iesrc == nxg ? 1 : 0), jsrc:jesrc, ksrc:kesrc] .= u_recv
            v_glob[isrc:iesrc, jsrc:jesrc + (jesrc == nyg ? 1 : 0), ksrc:kesrc] .= v_recv
            w_glob[isrc:iesrc, jsrc:jesrc, ksrc:kesrc + (kesrc == nzg ? 1 : 0)] .= w_recv
            p_glob[isrc:iesrc, jsrc:jesrc, ksrc:kesrc] .= p_recv
        end

        gstate = SolutionState3D(writer.grid.nx, writer.grid.ny, writer.grid.nz)
        gstate.u .= u_glob
        gstate.v .= v_glob
        gstate.w .= w_glob
        gstate.p .= p_glob
        gstate.t = current_time
        gstate.step = current_iteration
        return save_complete_snapshot!(writer, gstate, bodies, writer.grid, fluid, current_time, current_iteration; kwargs...)
    else
        hdr = Int[is, ie, js, je, ks, ke]
        MPI.Send(hdr, 0, 9200, comm)
        MPI.Send(Array(u_blk), 0, 9201, comm)
        MPI.Send(Array(v_blk), 0, 9202, comm)
        MPI.Send(Array(w_blk), 0, 9203, comm)
        MPI.Send(Array(p_blk), 0, 9204, comm)
        return true
    end
end
