struct ImmersedBoundaryData
    body_mask::Array{Bool}      # Mask for solid cells
    distance_function::Array{Float64}  # Signed distance to nearest surface
    normal_vectors::Array{Vector{Float64}}  # Surface normal vectors
    boundary_points::Vector{Vector{Float64}}  # Lagrangian boundary points
    forcing_points::Vector{Tuple{Vector{Int}, Float64}}  # Eulerian forcing points and weights
end

function ImmersedBoundaryData2D(bodies::RigidBodyCollection, grid::StaggeredGrid)
    nx, ny = grid.nx, grid.ny
    
    body_mask = bodies_mask_2d(bodies, grid)
    distance_function = compute_distance_function_2d(bodies, grid)
    normal_vectors = compute_normal_vectors_2d(bodies, grid, distance_function)
    boundary_points = generate_boundary_points_2d(bodies, grid)
    forcing_points = compute_forcing_points_2d(boundary_points, grid)
    
    ImmersedBoundaryData(body_mask, distance_function, normal_vectors, 
                        boundary_points, forcing_points)
end

function ImmersedBoundaryData3D(bodies::RigidBodyCollection, grid::StaggeredGrid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    body_mask = bodies_mask_3d(bodies, grid)
    distance_function = compute_distance_function_3d(bodies, grid)
    normal_vectors = compute_normal_vectors_3d(bodies, grid, distance_function)
    boundary_points = generate_boundary_points_3d(bodies, grid)
    forcing_points = compute_forcing_points_3d(boundary_points, grid)
    
    ImmersedBoundaryData(body_mask, distance_function, normal_vectors, 
                        boundary_points, forcing_points)
end

function compute_distance_function_2d(bodies::RigidBodyCollection, grid::StaggeredGrid)
    nx, ny = grid.nx, grid.ny
    distance_function = fill(Inf, nx, ny)
    
    for j = 1:ny, i = 1:nx
        x = grid.x[i]
        y = grid.y[j]
        
        min_distance = Inf
        for body in bodies.bodies
            dist = distance_to_surface(body, x, y)
            if abs(dist) < abs(min_distance)
                min_distance = dist
            end
        end
        distance_function[i, j] = min_distance
    end
    
    return distance_function
end

function compute_distance_function_3d(bodies::RigidBodyCollection, grid::StaggeredGrid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    distance_function = fill(Inf, nx, ny, nz)
    
    for k = 1:nz, j = 1:ny, i = 1:nx
        x = grid.x[i]
        y = grid.y[j]
        z = grid.z[k]
        
        min_distance = Inf
        for body in bodies.bodies
            # 3D distance computation would be implemented here
            dist = distance_to_surface(body.shape, body.center, body.angle, x, y)  # Placeholder
            if abs(dist) < abs(min_distance)
                min_distance = dist
            end
        end
        distance_function[i, j, k] = min_distance
    end
    
    return distance_function
end

function compute_normal_vectors_2d(bodies::RigidBodyCollection, grid::StaggeredGrid, 
                                  distance_function::Matrix{Float64})
    nx, ny = grid.nx, grid.ny
    normal_vectors = Array{Vector{Float64}}(undef, nx, ny)
    
    for j = 1:ny, i = 1:nx
        x = grid.x[i]
        y = grid.y[j]
        
        # Find closest body and compute normal
        closest_body = nothing
        min_distance = Inf
        
        for body in bodies.bodies
            dist = abs(distance_to_surface(body, x, y))
            if dist < min_distance
                min_distance = dist
                closest_body = body
            end
        end
        
        if closest_body !== nothing
            normal_vectors[i, j] = surface_normal(closest_body, x, y)
        else
            normal_vectors[i, j] = [0.0, 0.0]
        end
    end
    
    return normal_vectors
end

function compute_normal_vectors_3d(bodies::RigidBodyCollection, grid::StaggeredGrid,
                                  distance_function::Array{Float64,3})
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    normal_vectors = Array{Vector{Float64}}(undef, nx, ny, nz)
    
    for k = 1:nz, j = 1:ny, i = 1:nx
        # Similar to 2D but for 3D
        normal_vectors[i, j, k] = [0.0, 0.0, 0.0]  # Placeholder
    end
    
    return normal_vectors
end

function generate_boundary_points_2d(bodies::RigidBodyCollection, grid::StaggeredGrid)
    boundary_points = Vector{Vector{Float64}}()
    
    for body in bodies.bodies
        if body.shape isa Circle
            # Generate points around circle circumference
            n_points = max(16, Int(round(2π * body.shape.radius / min(grid.dx, grid.dy))))
            for i = 1:n_points
                θ = 2π * (i-1) / n_points
                x = body.center[1] + body.shape.radius * cos(θ + body.angle)
                y = body.center[2] + body.shape.radius * sin(θ + body.angle)
                push!(boundary_points, [x, y])
            end
        elseif body.shape isa Square
            # Generate points around square perimeter
            side = body.shape.side_length
            n_per_side = max(4, Int(round(side / min(grid.dx, grid.dy))))
            
            # Four sides of the square
            for side_idx = 1:4
                for i = 1:n_per_side
                    s = (i-1) / n_per_side
                    local_x, local_y = 0.0, 0.0
                    
                    if side_idx == 1      # Bottom
                        local_x = -side/2 + s*side
                        local_y = -side/2
                    elseif side_idx == 2  # Right
                        local_x = side/2
                        local_y = -side/2 + s*side
                    elseif side_idx == 3  # Top
                        local_x = side/2 - s*side
                        local_y = side/2
                    else                  # Left
                        local_x = -side/2
                        local_y = side/2 - s*side
                    end
                    
                    # Rotate and translate
                    cos_θ = cos(body.angle)
                    sin_θ = sin(body.angle)
                    x = body.center[1] + cos_θ * local_x - sin_θ * local_y
                    y = body.center[2] + sin_θ * local_x + cos_θ * local_y
                    push!(boundary_points, [x, y])
                end
            end
        end
    end
    
    return boundary_points
end

function generate_boundary_points_3d(bodies::RigidBodyCollection, grid::StaggeredGrid)
    # Similar to 2D but generate surface points for 3D bodies
    boundary_points = Vector{Vector{Float64}}()
    
    for body in bodies.bodies
        if body.shape isa Circle  # Sphere in 3D
            # Generate points on sphere surface using spherical coordinates
            n_theta = max(8, Int(round(2π * body.shape.radius / min(grid.dx, grid.dy))))
            n_phi = max(4, Int(round(π * body.shape.radius / grid.dz)))
            
            for i = 1:n_theta, j = 1:n_phi
                θ = 2π * (i-1) / n_theta
                φ = π * (j-1) / (n_phi-1)
                
                x = body.center[1] + body.shape.radius * sin(φ) * cos(θ)
                y = body.center[2] + body.shape.radius * sin(φ) * sin(θ)
                z = body.center[3] + body.shape.radius * cos(φ)
                push!(boundary_points, [x, y, z])
            end
        end
    end
    
    return boundary_points
end

function compute_forcing_points_2d(boundary_points::Vector{Vector{Float64}}, 
                                  grid::StaggeredGrid)
    forcing_points = Vector{Tuple{Vector{Int}, Float64}}()
    
    for bp in boundary_points
        x, y = bp[1], bp[2]
        
        # Find surrounding grid cells and compute interpolation weights
        i = searchsortedfirst(grid.x, x)
        j = searchsortedfirst(grid.y, y)
        
        if i > 1 && i <= length(grid.x) && j > 1 && j <= length(grid.y)
            # Bilinear interpolation weights
            dx = (x - grid.x[i-1]) / grid.dx
            dy = (y - grid.y[j-1]) / grid.dy
            
            # Four surrounding points with weights
            push!(forcing_points, ([i-1, j-1], (1-dx)*(1-dy)))
            push!(forcing_points, ([i, j-1], dx*(1-dy)))
            push!(forcing_points, ([i-1, j], (1-dx)*dy))
            push!(forcing_points, ([i, j], dx*dy))
        end
    end
    
    return forcing_points
end

function compute_forcing_points_3d(boundary_points::Vector{Vector{Float64}}, 
                                  grid::StaggeredGrid)
    forcing_points = Vector{Tuple{Vector{Int}, Float64}}()
    
    for bp in boundary_points
        x, y, z = bp[1], bp[2], bp[3]
        
        # Find surrounding grid cells for trilinear interpolation
        i = searchsortedfirst(grid.x, x)
        j = searchsortedfirst(grid.y, y)
        k = searchsortedfirst(grid.z, z)
        
        if i > 1 && i <= length(grid.x) && j > 1 && j <= length(grid.y) && 
           k > 1 && k <= length(grid.z)
            
            dx = (x - grid.x[i-1]) / grid.dx
            dy = (y - grid.y[j-1]) / grid.dy
            dz = (z - grid.z[k-1]) / grid.dz
            
            # Eight surrounding points with trilinear weights
            weights = [
                (1-dx)*(1-dy)*(1-dz), dx*(1-dy)*(1-dz),
                (1-dx)*dy*(1-dz), dx*dy*(1-dz),
                (1-dx)*(1-dy)*dz, dx*(1-dy)*dz,
                (1-dx)*dy*dz, dx*dy*dz
            ]
            
            indices = [
                [i-1,j-1,k-1], [i,j-1,k-1], [i-1,j,k-1], [i,j,k-1],
                [i-1,j-1,k], [i,j-1,k], [i-1,j,k], [i,j,k]
            ]
            
            for (idx, w) in zip(indices, weights)
                push!(forcing_points, (idx, w))
            end
        end
    end
    
    return forcing_points
end

function apply_immersed_boundary_forcing!(state::SolutionState, ib_data::ImmersedBoundaryData,
                                        bodies::RigidBodyCollection, grid::StaggeredGrid, dt::Float64)
    if grid.grid_type == TwoDimensional
        apply_ib_forcing_2d!(state, ib_data, bodies, grid, dt)
    elseif grid.grid_type == ThreeDimensional
        apply_ib_forcing_3d!(state, ib_data, bodies, grid, dt)
    end
end

function apply_ib_forcing_2d!(state::SolutionState, ib_data::ImmersedBoundaryData,
                             bodies::RigidBodyCollection, grid::StaggeredGrid, dt::Float64)
    # Direct forcing method: set velocity inside bodies to body velocity
    for j = 1:grid.ny, i = 1:grid.nx+1
        if i <= grid.nx && ib_data.body_mask[i, j]
            # Find which body this point belongs to
            x = grid.xu[i]  # u-velocity location
            y = grid.y[j]   # cell center y
            
            for body in bodies.bodies
                if is_inside(body, x, y)
                    body_vel = get_body_velocity_at_point(body, x, y)
                    state.u[i, j] = body_vel[1]
                    break
                end
            end
        end
    end
    
    for j = 1:grid.ny+1, i = 1:grid.nx
        if j <= grid.ny && ib_data.body_mask[i, j]
            x = grid.x[i]    # cell center x
            y = grid.yv[j]   # v-velocity location
            
            for body in bodies.bodies
                if is_inside(body, x, y)
                    body_vel = get_body_velocity_at_point(body, x, y)
                    state.v[i, j] = body_vel[2]
                    break
                end
            end
        end
    end
end

function apply_ib_forcing_3d!(state::SolutionState, ib_data::ImmersedBoundaryData,
                             bodies::RigidBodyCollection, grid::StaggeredGrid, dt::Float64)
    # Similar to 2D but for 3D velocity components
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # u-velocity forcing
    for k = 1:nz, j = 1:ny, i = 1:nx+1
        if i <= nx && ib_data.body_mask[i, j, k]
            x = grid.xu[i]
            y = grid.y[j]
            z = grid.z[k]
            
            for body in bodies.bodies
                if is_inside(body, x, y, z)
                    body_vel = get_body_velocity_at_point(body, x, y)  # Extended to 3D
                    state.u[i, j, k] = body_vel[1]
                    break
                end
            end
        end
    end
    
    # v-velocity and w-velocity forcing similar...
end