struct NetCDFWriter
    filepath::String
    grid::StaggeredGrid
    max_snapshots::Int
    current_snapshot::Int
    time_interval::Float64
    iteration_interval::Int
    last_save_time::Float64
    last_save_iteration::Int
    ncfile::Any  # NetCDF file handle
    
    function NetCDFWriter(filepath::String, grid::StaggeredGrid;
                         max_snapshots::Int=100,
                         time_interval::Float64=0.1,
                         iteration_interval::Int=10)
        new(filepath, grid, max_snapshots, 0, time_interval, iteration_interval, 
            0.0, 0, nothing)
    end
end

function initialize_netcdf_file!(writer::NetCDFWriter)
    # Create NetCDF file with appropriate dimensions and variables
    nx, ny = writer.grid.nx, writer.grid.ny
    is_3d = writer.grid.grid_type == ThreeDimensional
    nz = is_3d ? writer.grid.nz : 1
    
    # Remove existing file if it exists
    if isfile(writer.filepath)
        rm(writer.filepath)
    end
    
    # Create file using NetCDF.jl
    ncfile = NetCDF.create(writer.filepath, Dict(
        "nx" => nx,
        "ny" => ny,
        "nz" => nz,
        "time" => writer.max_snapshots
    ))
    
    # Define coordinate variables
    NetCDF.defVar(ncfile, "x", Float64, ("nx",))
    NetCDF.defVar(ncfile, "y", Float64, ("ny",))
    if is_3d
        NetCDF.defVar(ncfile, "z", Float64, ("nz",))
    end
    NetCDF.defVar(ncfile, "time", Float64, ("time",))
    
    # Define staggered grid coordinates
    NetCDF.defVar(ncfile, "xu", Float64, ("nx_u",)) where {
        ncfile.dim["nx_u"] = nx + 1
    }
    NetCDF.defVar(ncfile, "yv", Float64, ("ny_v",)) where {
        ncfile.dim["ny_v"] = ny + 1
    }
    if is_3d
        NetCDF.defVar(ncfile, "zw", Float64, ("nz_w",)) where {
            ncfile.dim["nz_w"] = nz + 1
        }
    end
    
    # Define solution variables
    if is_3d
        NetCDF.defVar(ncfile, "u", Float64, ("nx_u", "ny", "nz", "time"))
        NetCDF.defVar(ncfile, "v", Float64, ("nx", "ny_v", "nz", "time"))
        NetCDF.defVar(ncfile, "w", Float64, ("nx", "ny", "nz_w", "time"))
        NetCDF.defVar(ncfile, "p", Float64, ("nx", "ny", "nz", "time"))
    else
        NetCDF.defVar(ncfile, "u", Float64, ("nx_u", "ny", "time"))
        NetCDF.defVar(ncfile, "v", Float64, ("nx", "ny_v", "time"))
        NetCDF.defVar(ncfile, "p", Float64, ("nx", "ny", "time"))
    end
    
    # Add variable attributes
    NetCDF.putatt(ncfile, "u", Dict("long_name" => "x-velocity", "units" => "m/s"))
    NetCDF.putatt(ncfile, "v", Dict("long_name" => "y-velocity", "units" => "m/s"))
    NetCDF.putatt(ncfile, "p", Dict("long_name" => "pressure", "units" => "Pa"))
    if is_3d
        NetCDF.putatt(ncfile, "w", Dict("long_name" => "z-velocity", "units" => "m/s"))
    end
    
    # Write coordinate data
    NetCDF.putvar(ncfile, "x", writer.grid.x)
    NetCDF.putvar(ncfile, "y", writer.grid.y)
    NetCDF.putvar(ncfile, "xu", writer.grid.xu)
    NetCDF.putvar(ncfile, "yv", writer.grid.yv)
    if is_3d
        NetCDF.putvar(ncfile, "z", writer.grid.z)
        NetCDF.putvar(ncfile, "zw", writer.grid.zw)
    end
    
    # Add global attributes
    NetCDF.putatt(ncfile, "global", Dict(
        "title" => "BioFlow.jl simulation results",
        "institution" => "BioFlow.jl",
        "source" => "Finite volume Navier-Stokes solver with immersed boundary method",
        "grid_type" => string(writer.grid.grid_type),
        "nx" => nx,
        "ny" => ny,
        "nz" => nz,
        "Lx" => writer.grid.Lx,
        "Ly" => writer.grid.Ly,
        "Lz" => writer.grid.Lz,
        "dx" => writer.grid.dx,
        "dy" => writer.grid.dy,
        "dz" => writer.grid.dz,
        "max_snapshots" => writer.max_snapshots,
        "time_interval" => writer.time_interval,
        "iteration_interval" => writer.iteration_interval
    ))
    
    # Store file handle
    writer.ncfile = ncfile
    
    return ncfile
end

function should_save(writer::NetCDFWriter, current_time::Float64, current_iteration::Int)
    # Check if it's time to save based on time interval or iteration interval
    time_condition = (current_time - writer.last_save_time) >= writer.time_interval
    iteration_condition = (current_iteration - writer.last_save_iteration) >= writer.iteration_interval
    
    return time_condition || iteration_condition
end

function save_snapshot!(writer::NetCDFWriter, state::SolutionState, current_time::Float64, current_iteration::Int)
    if writer.ncfile === nothing
        initialize_netcdf_file!(writer)
    end
    
    # Check if we should save
    if !should_save(writer, current_time, current_iteration)
        return false
    end
    
    # Check if we have room for more snapshots
    if writer.current_snapshot >= writer.max_snapshots
        @warn "Maximum number of snapshots ($(writer.max_snapshots)) reached. Not saving."
        return false
    end
    
    writer.current_snapshot += 1
    snapshot_idx = writer.current_snapshot
    
    # Write time
    NetCDF.putvar(writer.ncfile, "time", [current_time], start=[snapshot_idx])
    
    # Write velocity and pressure data
    is_3d = writer.grid.grid_type == ThreeDimensional
    
    if is_3d
        NetCDF.putvar(writer.ncfile, "u", state.u, start=[1, 1, 1, snapshot_idx])
        NetCDF.putvar(writer.ncfile, "v", state.v, start=[1, 1, 1, snapshot_idx])
        NetCDF.putvar(writer.ncfile, "w", state.w, start=[1, 1, 1, snapshot_idx])
        NetCDF.putvar(writer.ncfile, "p", state.p, start=[1, 1, 1, snapshot_idx])
    else
        NetCDF.putvar(writer.ncfile, "u", state.u, start=[1, 1, snapshot_idx])
        NetCDF.putvar(writer.ncfile, "v", state.v, start=[1, 1, snapshot_idx])
        NetCDF.putvar(writer.ncfile, "p", state.p, start=[1, 1, snapshot_idx])
    end
    
    # Update last save times
    writer.last_save_time = current_time
    writer.last_save_iteration = current_iteration
    
    println("Saved snapshot $(snapshot_idx) at time $(current_time), iteration $(current_iteration)")
    return true
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
                body_var_y = "flexible_body_$(body_idx)_y"
                
                if !haskey(writer.ncfile.vars, body_var_x)
                    NetCDF.defVar(writer.ncfile, body_var_x, Float64, ("n_points_$(body_idx)", "time"))
                    NetCDF.defVar(writer.ncfile, body_var_y, Float64, ("n_points_$(body_idx)", "time"))
                    writer.ncfile.dim["n_points_$(body_idx)"] = body.n_points
                end
                
                # Save Lagrangian point positions
                NetCDF.putvar(writer.ncfile, body_var_x, body.X[:, 1], start=[1, snapshot_idx])
                NetCDF.putvar(writer.ncfile, body_var_y, body.X[:, 2], start=[1, snapshot_idx])
            end
        end
    end
    
    return true
end

function close_netcdf!(writer::NetCDFWriter)
    if writer.ncfile !== nothing
        NetCDF.close(writer.ncfile)
        writer.ncfile = nothing
        println("Closed NetCDF file: $(writer.filepath)")
    end
end

function create_new_file!(writer::NetCDFWriter, file_suffix::String)
    # Close current file and create a new one
    close_netcdf!(writer)
    
    # Update filepath with suffix
    base_path, ext = splitext(writer.filepath)
    writer.filepath = "$(base_path)_$(file_suffix)$(ext)"
    
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
    y = NetCDF.readvar(ncfile, "y")
    time_data = NetCDF.readvar(ncfile, "time")
    
    # Read solution variables
    u_data = NetCDF.readvar(ncfile, "u")
    v_data = NetCDF.readvar(ncfile, "v")
    p_data = NetCDF.readvar(ncfile, "p")
    
    # Check if 3D
    has_w = haskey(ncfile.vars, "w")
    w_data = has_w ? NetCDF.readvar(ncfile, "w") : nothing
    
    NetCDF.close(ncfile)
    
    return Dict(
        "x" => x,
        "y" => y,
        "time" => time_data,
        "u" => u_data,
        "v" => v_data,
        "w" => w_data,
        "p" => p_data
    )
end