function create_2d_solver(nx::Int, nz::Int, Lx::Float64, Lz::Float64,
                         fluid::FluidProperties, bc::BoundaryConditions;
                         grid_type::GridType=TwoDimensional,
                         time_scheme::TimeSteppingScheme=RungeKutta3(),
                         use_mpi::Bool=false,
                         origin_x::Float64=0.0, origin_z::Float64=0.0)
    
    if grid_type == TwoDimensional
        grid = StaggeredGrid2D(nx, nz, Lx, Lz; origin_x=origin_x, origin_z=origin_z)
    else
        error("Invalid grid type for 2D solver: $grid_type")
    end
    
    if use_mpi
        return MPINavierStokesSolver2D(nx, nz, Lx, Lz, fluid, bc, time_scheme)
    else
        return NavierStokesSolver2D(grid, fluid, bc, time_scheme)
    end
end

function create_uniform_2d_grid(nx::Int, nz::Int, Lx::Float64, Lz::Float64;
                               grid_type::GridType=TwoDimensional,
                               origin_x::Float64=0.0, origin_z::Float64=0.0)
    if grid_type == TwoDimensional
        return StaggeredGrid2D(nx, nz, Lx, Lz; origin_x=origin_x, origin_z=origin_z)
    else
        error("Invalid grid type for 2D: $grid_type")
    end
end

function create_stretched_2d_grid(x_points::Vector{Float64}, z_points::Vector{Float64};
                                 grid_type::GridType=TwoDimensional)
    # Create non-uniform grid with specified grid points (XZ plane)
    nx = length(x_points) - 1
    nz = length(z_points) - 1
    
    Lx = x_points[end] - x_points[1]
    Lz = z_points[end] - z_points[1]
    
    if grid_type == TwoDimensional
        grid = StaggeredGrid2D(nx, nz, Lx, Lz; origin_x=x_points[1], origin_z=z_points[1])
        
        # Override with custom spacing
        grid.x .= 0.5 .* (x_points[1:end-1] .+ x_points[2:end])  # Cell centers
        grid.z .= 0.5 .* (z_points[1:end-1] .+ z_points[2:end])
        grid.xu .= x_points  # Face centers for u-velocity
        grid.zw .= z_points  # Face centers for w-velocity
        
        return grid
    else
        error("Invalid grid type for 2D: $grid_type")
    end
end

function refine_2d_grid_near_bodies(base_grid::StaggeredGrid, 
                                bodies::Union{RigidBodyCollection, FlexibleBodyCollection},
                                refinement_factor::Int=2, 
                                refinement_radius::Float64=0.5)

    # Create locally refined grid near immersed bodies
    refined_grid = RefinedGrid(base_grid)
    
    # Mark cells for refinement based on distance to bodies
    for j = 1:base_grid.nz, i = 1:base_grid.nx
        x = base_grid.x[i]
        z = base_grid.z[j]
        
        min_distance = Inf
        
        if bodies isa RigidBodyCollection
            for body in bodies.bodies
                dist = abs(distance_to_surface_xz(body, x, z))
                min_distance = min(min_distance, dist)
            end
        elseif bodies isa FlexibleBodyCollection
            for body in bodies.bodies
                for k = 1:body.n_points
                    dist = sqrt((x - body.X[k, 1])^2 + (z - body.X[k, 2])^2)  # X[k,2] is z-coordinate
                    min_distance = min(min_distance, dist)
                end
            end
        end
        
        if min_distance < refinement_radius
            level = min(3, Int(ceil(log2(refinement_radius / min_distance))))
            refined_grid.refined_cells[(i, j)] = level
        end
    end
    
    return refined_grid
end

# Utility functions for 2D grids
function compute_grid_metrics_2d(grid::StaggeredGrid)
    metrics = Dict{String, Any}()
    metrics["grid_type"] = string(grid.grid_type)
    metrics["nx"] = grid.nx
    metrics["nz"] = grid.nz
    metrics["Lx"] = grid.Lx
    metrics["Lz"] = grid.Lz
    metrics["dx_min"] = minimum([grid.dx])
    metrics["dz_min"] = minimum([grid.dz])
    metrics["dx_max"] = maximum([grid.dx])
    metrics["dz_max"] = maximum([grid.dz])
    metrics["aspect_ratio"] = grid.Lx / grid.Lz
    metrics["total_cells"] = grid.nx * grid.nz
    
    return metrics
end

function print_grid_info_2d(grid::StaggeredGrid)
    println("2D Grid Information:")
    println("  Type: $(grid.grid_type)")
    println("  Dimensions: $(grid.nx) × $(grid.nz) (XZ plane)")
    println("  Domain size: $(grid.Lx) × $(grid.Lz)")
    println("  Grid spacing: Δx = $(grid.dx), Δz = $(grid.dz)")
    println("  Total cells: $(grid.nx * grid.nz)")
    println("  Aspect ratio: $(grid.Lx / grid.Lz)")
end

function validate_2d_grid(grid::StaggeredGrid)
    # Perform basic validation checks
    checks_passed = true
    
    if grid.nx <= 0 || grid.nz <= 0
        println("ERROR: Grid dimensions must be positive")
        checks_passed = false
    end
    
    if grid.Lx <= 0 || grid.Lz <= 0
        println("ERROR: Domain size must be positive")
        checks_passed = false
    end
    
    if grid.dx <= 0 || grid.dz <= 0
        println("ERROR: Grid spacing must be positive")
        checks_passed = false
    end
    
    # Check array dimensions
    if length(grid.x) != grid.nx || length(grid.z) != grid.nz
        println("ERROR: Coordinate array dimensions don't match grid size")
        checks_passed = false
    end
    
    if length(grid.xu) != grid.nx + 1 || length(grid.zw) != grid.nz + 1
        println("ERROR: Staggered grid array dimensions incorrect")
        checks_passed = false
    end
    
    if checks_passed
        println("Grid validation passed")
    else
        error("Grid validation failed")
    end
    
    return checks_passed
end

function export_2d_grid_vtk(grid::StaggeredGrid, filename::String)
    # Export grid structure to VTK format for visualization
    # This is a simplified implementation
    
    open(filename, "w") do file
        println(file, "# vtk DataFile Version 3.0")
        println(file, "2D Staggered Grid")
        println(file, "ASCII")
        println(file, "DATASET STRUCTURED_GRID")
        println(file, "DIMENSIONS $(grid.nx+1) $(grid.nz+1) 1")
        println(file, "POINTS $((grid.nx+1)*(grid.nz+1)) float")
        
        # Write grid points (XZ plane)
        for k = 0:grid.nz
            for i = 0:grid.nx
                x = i * grid.dx + (grid.x[1] - grid.dx/2)
                z = k * grid.dz + (grid.z[1] - grid.dz/2)
                println(file, "$x 0.0 $z")  # y=0 for XZ plane
            end
        end
    end
    
    println("Grid exported to $filename")
end
