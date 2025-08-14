function create_3d_solver(nx::Int, ny::Int, nz::Int, Lx::Float64, Ly::Float64, Lz::Float64,
                         fluid::FluidProperties, bc::BoundaryConditions;
                         time_scheme::TimeSteppingScheme=RungeKutta3(),
                         use_mpi::Bool=false,
                         origin_x::Float64=0.0, origin_y::Float64=0.0, origin_z::Float64=0.0)
    
    grid = StaggeredGrid3D(nx, ny, nz, Lx, Ly, Lz; 
                          origin_x=origin_x, origin_y=origin_y, origin_z=origin_z)
    
    if use_mpi
        return MPINavierStokesSolver3D(nx, ny, nz, Lx, Ly, Lz, fluid, bc, time_scheme)
    else
        return NavierStokesSolver3D(grid, fluid, bc, time_scheme)
    end
end

function create_uniform_3d_grid(nx::Int, ny::Int, nz::Int, Lx::Float64, Ly::Float64, Lz::Float64;
                               origin_x::Float64=0.0, origin_y::Float64=0.0, origin_z::Float64=0.0)
    return StaggeredGrid3D(nx, ny, nz, Lx, Ly, Lz; 
                          origin_x=origin_x, origin_y=origin_y, origin_z=origin_z)
end

function create_stretched_3d_grid(x_points::Vector{Float64}, y_points::Vector{Float64}, z_points::Vector{Float64})
    nx = length(x_points) - 1
    ny = length(y_points) - 1
    nz = length(z_points) - 1
    
    Lx = x_points[end] - x_points[1]
    Ly = y_points[end] - y_points[1]
    Lz = z_points[end] - z_points[1]
    
    grid = StaggeredGrid3D(nx, ny, nz, Lx, Ly, Lz; 
                          origin_x=x_points[1], origin_y=y_points[1], origin_z=z_points[1])
    
    # Override with custom spacing
    grid.x .= 0.5 .* (x_points[1:end-1] .+ x_points[2:end])  # Cell centers
    grid.y .= 0.5 .* (y_points[1:end-1] .+ y_points[2:end])
    grid.z .= 0.5 .* (z_points[1:end-1] .+ z_points[2:end])
    grid.xu .= x_points  # Face centers for u-velocity
    grid.yv .= y_points  # Face centers for v-velocity
    grid.zw .= z_points  # Face centers for w-velocity
    
    return grid
end

function create_cylindrical_3d_grid(nr::Int, ntheta::Int, nz::Int, 
                                   r_max::Float64, theta_max::Float64, Lz::Float64;
                                   r_min::Float64=0.0, theta_min::Float64=0.0, origin_z::Float64=0.0)
    # Create cylindrical coordinate grid (converted to Cartesian)
    r_points = LinRange(r_min, r_max, nr + 1)
    theta_points = LinRange(theta_min, theta_max, ntheta + 1)
    z_points = LinRange(origin_z, origin_z + Lz, nz + 1)
    
    # Convert to Cartesian grid points
    # This is a simplified approach - a full implementation would handle the coordinate transformation properly
    x_extent = 2 * r_max
    y_extent = 2 * r_max
    
    # Create equivalent Cartesian grid
    nx = 2 * nr  # Approximate mapping
    ny = 2 * nr
    
    return create_uniform_3d_grid(nx, ny, nz, x_extent, y_extent, Lz; 
                                 origin_x=-r_max, origin_y=-r_max, origin_z=origin_z)
end

function refine_3d_grid_near_bodies(base_grid::StaggeredGrid, bodies::Union{RigidBodyCollection, FlexibleBodyCollection},
                                   refinement_factor::Int=2, refinement_radius::Float64=0.5)
    refined_grid = RefinedGrid(base_grid)
    
    # Mark cells for refinement based on distance to bodies
    for k = 1:base_grid.nz, j = 1:base_grid.ny, i = 1:base_grid.nx
        x = base_grid.x[i]
        y = base_grid.y[j]
        z = base_grid.z[k]
        
        min_distance = Inf
        
        if bodies isa RigidBodyCollection
            for body in bodies.bodies
                # For 3D, need to extend distance calculation
                dist = sqrt((x - body.center[1])^2 + (y - body.center[2])^2 + 
                           (length(body.center) > 2 ? (z - body.center[3])^2 : z^2))
                
                if body.shape isa Circle  # Treat as sphere
                    dist = dist - body.shape.radius
                end
                
                min_distance = min(min_distance, abs(dist))
            end
        elseif bodies isa FlexibleBodyCollection
            for body in bodies.bodies
                for l = 1:body.n_points
                    # Assume flexible body has 2D points, extend to 3D by ignoring z-component
                    dist = sqrt((x - body.X[l, 1])^2 + (y - body.X[l, 2])^2 + z^2)
                    min_distance = min(min_distance, dist)
                end
            end
        end
        
        if min_distance < refinement_radius
            level = min(3, Int(ceil(log2(refinement_radius / min_distance))))
            refined_grid.refined_cells[(i, j, k)] = level  # Correct 3D indexing
        end
    end
    
    return refined_grid
end

function compute_grid_metrics_3d(grid::StaggeredGrid)
    metrics = Dict{String, Any}()
    metrics["grid_type"] = string(grid.grid_type)
    metrics["nx"] = grid.nx
    metrics["ny"] = grid.ny
    metrics["nz"] = grid.nz
    metrics["Lx"] = grid.Lx
    metrics["Ly"] = grid.Ly
    metrics["Lz"] = grid.Lz
    metrics["dx"] = grid.dx
    metrics["dy"] = grid.dy
    metrics["dz"] = grid.dz
    metrics["aspect_ratio_xy"] = grid.Lx / grid.Ly
    metrics["aspect_ratio_xz"] = grid.Lx / grid.Lz
    metrics["aspect_ratio_yz"] = grid.Ly / grid.Lz
    metrics["total_cells"] = grid.nx * grid.ny * grid.nz
    metrics["volume"] = grid.Lx * grid.Ly * grid.Lz
    
    return metrics
end

function print_grid_info_3d(grid::StaggeredGrid)
    println("3D Grid Information:")
    println("  Type: $(grid.grid_type)")
    println("  Dimensions: $(grid.nx) × $(grid.ny) × $(grid.nz)")
    println("  Domain size: $(grid.Lx) × $(grid.Ly) × $(grid.Lz)")
    println("  Grid spacing: Δx = $(grid.dx), Δy = $(grid.dy), Δz = $(grid.dz)")
    println("  Total cells: $(grid.nx * grid.ny * grid.nz)")
    println("  Aspect ratios: xy = $(grid.Lx / grid.Ly), xz = $(grid.Lx / grid.Lz), yz = $(grid.Ly / grid.Lz)")
    println("  Domain volume: $(grid.Lx * grid.Ly * grid.Lz)")
end

function validate_3d_grid(grid::StaggeredGrid)
    checks_passed = true
    
    if grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0
        println("ERROR: Grid dimensions must be positive")
        checks_passed = false
    end
    
    if grid.Lx <= 0 || grid.Ly <= 0 || grid.Lz <= 0
        println("ERROR: Domain size must be positive")
        checks_passed = false
    end
    
    if grid.dx <= 0 || grid.dy <= 0 || grid.dz <= 0
        println("ERROR: Grid spacing must be positive")
        checks_passed = false
    end
    
    # Check array dimensions
    if length(grid.x) != grid.nx || length(grid.y) != grid.ny || length(grid.z) != grid.nz
        println("ERROR: Coordinate array dimensions don't match grid size")
        checks_passed = false
    end
    
    if length(grid.xu) != grid.nx + 1 || length(grid.yv) != grid.ny + 1 || length(grid.zw) != grid.nz + 1
        println("ERROR: Staggered grid array dimensions incorrect")
        checks_passed = false
    end
    
    if checks_passed
        println("Grid validation passed ✓")
    else
        error("Grid validation failed")
    end
    
    return checks_passed
end

function create_channel_3d_grid(nx::Int, ny::Int, nz::Int, 
                               channel_length::Float64, channel_width::Float64, channel_height::Float64;
                               inlet_length::Float64=0.2*channel_length,
                               outlet_length::Float64=0.3*channel_length)
    # Create a channel grid with inlet and outlet regions
    
    Lx = inlet_length + channel_length + outlet_length
    Ly = channel_width
    Lz = channel_height
    
    # Create non-uniform grid with refined regions at inlet and outlet
    x_points = Vector{Float64}()
    
    # Inlet region - uniform
    inlet_nx = Int(round(nx * inlet_length / Lx))
    append!(x_points, LinRange(0.0, inlet_length, inlet_nx + 1))
    
    # Main channel - uniform
    main_nx = Int(round(nx * channel_length / Lx))
    append!(x_points, LinRange(inlet_length, inlet_length + channel_length, main_nx + 1)[2:end])
    
    # Outlet region - uniform
    outlet_nx = nx - inlet_nx - main_nx
    append!(x_points, LinRange(inlet_length + channel_length, Lx, outlet_nx + 1)[2:end])
    
    y_points = LinRange(0.0, Ly, ny + 1)
    z_points = LinRange(0.0, Lz, nz + 1)
    
    return create_stretched_3d_grid(x_points, collect(y_points), collect(z_points))
end

function create_wake_refined_3d_grid(nx::Int, ny::Int, nz::Int, 
                                    Lx::Float64, Ly::Float64, Lz::Float64,
                                    body_center::Vector{Float64},
                                    wake_length::Float64, wake_width::Float64)
    # Create grid with refinement in wake region behind a body
    
    x_body = body_center[1]
    y_body = length(body_center) > 1 ? body_center[2] : Ly/2
    z_body = length(body_center) > 2 ? body_center[3] : Lz/2
    
    # Create base uniform grid
    grid = create_uniform_3d_grid(nx, ny, nz, Lx, Ly, Lz)
    
    # Mark wake region for refinement
    refined_grid = RefinedGrid(grid)
    
    for k = 1:nz, j = 1:ny, i = 1:nx
        x = grid.x[i]
        y = grid.y[j]
        z = grid.z[k]
        
        # Check if point is in wake region
        in_wake_x = x > x_body && x < x_body + wake_length
        in_wake_y = abs(y - y_body) < wake_width/2
        in_wake_z = abs(z - z_body) < wake_width/2
        
        if in_wake_x && in_wake_y && in_wake_z
            # Distance-based refinement level
            distance_from_body = sqrt((x - x_body)^2 + (y - y_body)^2 + (z - z_body)^2)
            if distance_from_body < wake_width/4
                refined_grid.refined_cells[(i, j, k)] = 2  # High refinement near body
            elseif distance_from_body < wake_width/2
                refined_grid.refined_cells[(i, j, k)] = 1  # Moderate refinement in wake
            end
        end
    end
    
    return refined_grid
end

function export_3d_grid_vtk(grid::StaggeredGrid, filename::String)
    # Export 3D grid structure to VTK format
    
    open(filename, "w") do file
        println(file, "# vtk DataFile Version 3.0")
        println(file, "3D Staggered Grid")
        println(file, "ASCII")
        println(file, "DATASET STRUCTURED_GRID")
        println(file, "DIMENSIONS $(grid.nx+1) $(grid.ny+1) $(grid.nz+1)")
        println(file, "POINTS $((grid.nx+1)*(grid.ny+1)*(grid.nz+1)) float")
        
        # Write grid points
        for k = 0:grid.nz
            for j = 0:grid.ny
                for i = 0:grid.nx
                    x = i * grid.dx + (grid.x[1] - grid.dx/2)
                    y = j * grid.dy + (grid.y[1] - grid.dy/2)
                    z = k * grid.dz + (grid.z[1] - grid.dz/2)
                    println(file, "$x $y $z")
                end
            end
        end
    end
    
    println("3D Grid exported to $filename")
end