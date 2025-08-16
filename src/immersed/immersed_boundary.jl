struct ImmersedBoundaryData
    body_mask::Array{Bool}      # Mask for solid cells
    distance_function::Array{Float64}  # Signed distance to nearest surface
    normal_vectors::Array{Vector{Float64}}  # Surface normal vectors
    boundary_points::Vector{Vector{Float64}}  # Lagrangian boundary points
    forcing_points::Vector{Tuple{Vector{Int}, Float64}}  # Eulerian forcing points and weights
end

function ImmersedBoundaryData2D(bodies::RigidBodyCollection, grid::StaggeredGrid)
    nx, nz = grid.nx, grid.nz  # Use XZ plane for 2D
    
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

function bodies_mask_2d(bodies::RigidBodyCollection, grid::StaggeredGrid)
    nx, nz = grid.nx, grid.nz  # Use XZ plane for 2D
    body_mask = falses(nx, nz)
    
    for j = 1:nz, i = 1:nx
        x = grid.x[i]
        z = grid.z[j]  # Use z coordinate for XZ plane
        
        for body in bodies.bodies
            if is_inside_xz(body, x, z)  # Use XZ plane for 2D
                body_mask[i, j] = true
                break
            end
        end
    end
    
    return body_mask
end

function bodies_mask_3d(bodies::RigidBodyCollection, grid::StaggeredGrid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    body_mask = falses(nx, ny, nz)
    
    for k = 1:nz, j = 1:ny, i = 1:nx
        x = grid.x[i]
        y = grid.y[j]
        z = grid.z[k]
        
        for body in bodies.bodies
            if is_inside(body, x, y, z)
                body_mask[i, j, k] = true
                break
            end
        end
    end
    
    return body_mask
end

function compute_distance_function_2d(bodies::RigidBodyCollection, grid::StaggeredGrid)
    nx, nz = grid.nx, grid.nz  # Use XZ plane for 2D
    distance_function = fill(Inf, nx, nz)
    
    for j = 1:nz, i = 1:nx
        x = grid.x[i]
        z = grid.z[j]  # Use z coordinate for XZ plane
        
        min_distance = Inf
        for body in bodies.bodies
            dist = distance_to_surface_xz(body, x, z)  # Use XZ plane version
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
            # Use 3D distance computation
            dist = distance_to_surface_3d(body, x, y, z)
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
    nx, nz = grid.nx, grid.nz  # Use XZ plane for 2D
    normal_vectors = Array{Vector{Float64}}(undef, nx, nz)
    
    for j = 1:nz, i = 1:nx
        x = grid.x[i]
        z = grid.z[j]  # Use z coordinate for XZ plane
        
        # Find closest body and compute normal
        closest_body = nothing
        min_distance = Inf
        
        for body in bodies.bodies
            dist = abs(distance_to_surface_xz(body, x, z))  # Use XZ plane version
            if dist < min_distance
                min_distance = dist
                closest_body = body
            end
        end
        
        if closest_body !== nothing
            normal_vectors[i, j] = surface_normal_xz(closest_body, x, z)  # Use XZ plane version
        else
            normal_vectors[i, j] = [0.0, 0.0]  # Normal vector in XZ plane
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
            # Generate points around circle circumference in XZ plane
            n_points = max(16, Int(round(2π * body.shape.radius / min(grid.dx, grid.dz))))
            for i = 1:n_points
                θ = 2π * (i-1) / n_points
                x = body.center[1] + body.shape.radius * cos(θ + body.angle)
                z = (length(body.center) > 2 ? body.center[3] : body.center[2]) + body.shape.radius * sin(θ + body.angle)
                push!(boundary_points, [x, z])  # XZ plane coordinates
            end
        elseif body.shape isa Square
            # Generate points around square perimeter in XZ plane
            side = body.shape.side_length
            n_per_side = max(4, Int(round(side / min(grid.dx, grid.dz))))
            
            # Four sides of the square in XZ plane
            for side_idx = 1:4
                for i = 1:n_per_side
                    s = (i-1) / n_per_side
                    local_x, local_z = 0.0, 0.0
                    
                    if side_idx == 1      # Bottom (in XZ plane)
                        local_x = -side/2 + s*side
                        local_z = -side/2
                    elseif side_idx == 2  # Right
                        local_x = side/2
                        local_z = -side/2 + s*side
                    elseif side_idx == 3  # Top
                        local_x = side/2 - s*side
                        local_z = side/2
                    else                  # Left
                        local_x = -side/2
                        local_z = side/2 - s*side
                    end
                    
                    # Rotate and translate in XZ plane
                    cos_θ = cos(body.angle)
                    sin_θ = sin(body.angle)
                    x = body.center[1] + cos_θ * local_x - sin_θ * local_z
                    z = (length(body.center) > 2 ? body.center[3] : body.center[2]) + sin_θ * local_x + cos_θ * local_z
                    push!(boundary_points, [x, z])  # XZ plane coordinates
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
        x, z = bp[1], bp[2]  # XZ plane coordinates
        
        # Find surrounding grid cells and compute interpolation weights
        i = searchsortedfirst(grid.x, x)
        j = searchsortedfirst(grid.z, z)  # Use z coordinate for XZ plane
        
        if i > 1 && i <= length(grid.x) && j > 1 && j <= length(grid.z)
            # Bilinear interpolation weights in XZ plane
            dx = (x - grid.x[i-1]) / grid.dx
            dz = (z - grid.z[j-1]) / grid.dz  # Use dz for XZ plane
            
            # Four surrounding points with weights in XZ plane
            push!(forcing_points, ([i-1, j-1], (1-dx)*(1-dz)))
            push!(forcing_points, ([i, j-1], dx*(1-dz)))
            push!(forcing_points, ([i-1, j], (1-dx)*dz))
            push!(forcing_points, ([i, j], dx*dz))
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
                                        bodies::Union{RigidBodyCollection, FlexibleBodyCollection}, 
                                        grid::StaggeredGrid, dt::Float64)
    if grid.grid_type == TwoDimensional
        apply_ib_forcing_2d!(state, ib_data, bodies, grid, dt)
    elseif grid.grid_type == ThreeDimensional
        apply_ib_forcing_3d!(state, ib_data, bodies, grid, dt)
    end
end

function apply_immersed_boundary_forcing!(state::SolutionState, 
                                        flexible_bodies::FlexibleBodyCollection, 
                                        grid::StaggeredGrid, dt::Float64; δh::Float64=2.0*max(grid.dx, grid.dy))
    """
    Apply immersed boundary forcing for flexible bodies using force spreading.
    """
    if grid.grid_type == TwoDimensional
        apply_flexible_ib_forcing_2d!(state, flexible_bodies, grid, dt, δh)
    elseif grid.grid_type == ThreeDimensional
        apply_flexible_ib_forcing_3d!(state, flexible_bodies, grid, dt, δh)
    end
end

function apply_flexible_ib_forcing_2d!(state::SolutionState, bodies::FlexibleBodyCollection, 
                                     grid::StaggeredGrid, dt::Float64, δh::Float64)
    """
    Apply IBM forcing for flexible bodies in 2D using force spreading method.
    For 2D XZ plane: u is x-direction velocity, w is z-direction velocity.
    """
    nx, nz = grid.nx, grid.nz  # Use XZ plane for 2D
    
    # Create force field arrays for XZ plane
    force_field = Array{Vector{Float64},2}(undef, nx, nz)
    for j = 1:nz, i = 1:nx
        force_field[i, j] = [0.0, 0.0]  # [fx, fz] forces in XZ plane
    end
    
    # Collect Lagrangian positions and forces from all flexible bodies
    lagrangian_positions = Vector{Vector{Float64}}()
    lagrangian_forces = Vector{Vector{Float64}}()
    
    for body in bodies.bodies
        for i = 1:body.n_points
            # For XZ plane: X[i,1] is x-position, X[i,2] is z-position
            push!(lagrangian_positions, [body.X[i, 1], body.X[i, 2]])
            push!(lagrangian_forces, [body.force[i, 1], body.force[i, 2]])
        end
    end
    
    # Spread forces to Eulerian grid
    apply_force_spreading_2d!(force_field, lagrangian_forces, lagrangian_positions, grid, δh)
    
    # Add forces to momentum equations as source terms
    # For staggered grid, interpolate forces to appropriate locations
    
    # u-velocity forcing (x-direction)
    for j = 1:nz, i = 1:nx+1
        if i <= nx
            # u-velocity forcing at (xu[i], z[j])
            x_u = grid.xu[i]
            z_u = grid.z[j]
            
            # Interpolate force from cell centers to u-location
            if i == 1
                force_u = force_field[i, j][1]
            else
                force_u = 0.5 * (force_field[i-1, j][1] + force_field[i, j][1])
            end
            
            state.u[i, j] += force_u * dt
        end
    end
    
    # w-velocity forcing (z-direction) - note: state.v represents w-velocity in XZ plane
    for j = 1:nz+1, i = 1:nx
        if j <= nz
            # w-velocity forcing at (x[i], zw[j])
            x_w = grid.x[i]
            z_w = grid.zw[j]  # Use zw for w-velocity locations
            
            # Interpolate force from cell centers to w-location
            if j == 1
                force_w = force_field[i, j][2]
            else
                force_w = 0.5 * (force_field[i, j-1][2] + force_field[i, j][2])
            end
            
            # Apply to w-velocity (represented as state.v in XZ plane)
            state.v[i, j] += force_w * dt
        end
    end
end

function apply_flexible_ib_forcing_3d!(state::SolutionState, bodies::FlexibleBodyCollection, 
                                     grid::StaggeredGrid, dt::Float64, δh::Float64)
    """
    Apply IBM forcing for flexible bodies in 3D using force spreading method.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # Create 3D force field arrays
    force_field = Array{Vector{Float64},3}(undef, nx, ny, nz)
    for k = 1:nz, j = 1:ny, i = 1:nx
        force_field[i, j, k] = [0.0, 0.0, 0.0]
    end
    
    # Collect Lagrangian positions and forces from all flexible bodies
    lagrangian_positions = Vector{Vector{Float64}}()
    lagrangian_forces = Vector{Vector{Float64}}()
    
    for body in bodies.bodies
        for i = 1:body.n_points
            # For 2D flexible bodies in 3D domain, extend to include z-component
            push!(lagrangian_positions, [body.X[i, 1], body.X[i, 2], 0.0])
            push!(lagrangian_forces, [body.force[i, 1], body.force[i, 2], 0.0])
        end
    end
    
    # Spread forces to Eulerian grid
    apply_force_spreading_3d!(force_field, lagrangian_forces, lagrangian_positions, grid, δh)
    
    # Add forces to momentum equations (similar to 2D but with w-component)
    # Implementation would follow similar pattern as 2D case
end

function apply_ib_forcing_2d!(state::SolutionState, ib_data::ImmersedBoundaryData,
                             bodies::Union{RigidBodyCollection, FlexibleBodyCollection}, grid::StaggeredGrid, dt::Float64)
    # Direct forcing method: set velocity inside bodies to body velocity
    # For 2D XZ plane flows
    
    # u-velocity forcing (x-direction)
    for j = 1:grid.nz, i = 1:grid.nx+1
        if i <= grid.nx && ib_data.body_mask[i, j]
            # Find which body this point belongs to
            x = grid.xu[i]  # u-velocity location
            z = grid.z[j]   # cell center z for XZ plane
            
            for body in bodies.bodies
                if is_inside_xz(body, x, z)  # Use XZ plane version
                    body_vel = get_body_velocity_at_point_xz(body, x, z)
                    state.u[i, j] = body_vel[1]  # u-velocity (x-direction)
                    break
                end
            end
        end
    end
    
    # w-velocity forcing (z-direction) - represented as state.v in XZ plane
    for j = 1:grid.nz+1, i = 1:grid.nx
        if j <= grid.nz && ib_data.body_mask[i, j]
            x = grid.x[i]    # cell center x
            z = grid.zw[j]   # w-velocity location in z-direction
            
            for body in bodies.bodies
                if is_inside_xz(body, x, z)  # Use XZ plane version
                    body_vel = get_body_velocity_at_point_xz(body, x, z)
                    state.v[i, j] = body_vel[2]  # w-velocity (z-direction)
                    break
                end
            end
        end
    end
end

function apply_ib_forcing_3d!(state::SolutionState, ib_data::ImmersedBoundaryData,
                             bodies::Union{RigidBodyCollection, FlexibleBodyCollection}, grid::StaggeredGrid, dt::Float64)
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
                    body_vel = get_body_velocity_at_point(body, x, y, z)
                    state.u[i, j, k] = body_vel[1]
                    break
                end
            end
        end
    end
    
    # v-velocity forcing
    for k = 1:nz, j = 1:ny+1, i = 1:nx
        if j <= ny && ib_data.body_mask[i, j, k]
            x = grid.x[i]
            y = grid.yv[j]
            z = grid.z[k]
            
            for body in bodies.bodies
                if is_inside(body, x, y, z)
                    body_vel = get_body_velocity_at_point(body, x, y, z)
                    state.v[i, j, k] = body_vel[2]
                    break
                end
            end
        end
    end
    
    # w-velocity forcing
    for k = 1:nz+1, j = 1:ny, i = 1:nx
        if k <= nz && ib_data.body_mask[i, j, k]
            x = grid.x[i]
            y = grid.y[j]
            z = grid.zw[k]
            
            for body in bodies.bodies
                if is_inside(body, x, y, z)
                    body_vel = get_body_velocity_at_point(body, x, y, z)
                    state.w[i, j, k] = body_vel[3]
                    break
                end
            end
        end
    end
end

function get_body_velocity_at_point(body::RigidBody, x::Float64, y::Float64)
    # For rigid body motion: V = V_center + ω × r
    dx = x - body.center[1]
    dy = y - body.center[2]
    
    # Translational velocity + rotational velocity
    u_body = body.velocity[1] - body.angular_velocity * dy
    v_body = body.velocity[2] + body.angular_velocity * dx
    
    return [u_body, v_body]
end

function get_body_velocity_at_point(body::RigidBody, x::Float64, y::Float64, z::Float64)
    # For 3D rigid body motion
    dx = x - body.center[1]
    dy = y - body.center[2]
    dz = length(body.center) > 2 ? z - body.center[3] : z
    
    # For simplicity, assume rotation only about z-axis
    u_body = body.velocity[1] - body.angular_velocity * dy
    v_body = body.velocity[2] + body.angular_velocity * dx
    w_body = length(body.velocity) > 2 ? body.velocity[3] : 0.0
    
    return [u_body, v_body, w_body]
end

# XZ plane version for 2D flows
function get_body_velocity_at_point_xz(body::RigidBody, x::Float64, z::Float64)
    # For rigid body motion in XZ plane: V = V_center + ω × r
    dx = x - body.center[1]
    dz = z - (length(body.center) > 2 ? body.center[3] : body.center[2])  # Use z coordinate
    
    # Translational velocity + rotational velocity (rotation about y-axis)
    u_body = body.velocity[1] + body.angular_velocity * dz  # u + ω × dz
    w_body = (length(body.velocity) > 2 ? body.velocity[3] : body.velocity[2]) - body.angular_velocity * dx  # w - ω × dx
    
    return [u_body, w_body]  # [u, w] velocities in XZ plane
end

function apply_force_spreading_2d!(force_field::Array{Vector{Float64},2}, 
                                  lagrangian_forces::Vector{Vector{Float64}},
                                  lagrangian_positions::Vector{Vector{Float64}},
                                  grid::StaggeredGrid, δh::Float64)
    """
    Spread Lagrangian forces to Eulerian grid using regularized delta function.
    For 2D XZ plane flows.
    """
    nx, nz = size(force_field)  # Use XZ plane dimensions
    
    # Clear force field
    for j = 1:nz, i = 1:nx
        force_field[i, j] = [0.0, 0.0]  # [fx, fz] for XZ plane
    end
    
    # Spread forces from Lagrangian points to Eulerian grid
    for (pos, force) in zip(lagrangian_positions, lagrangian_forces)
        x_lag, z_lag = pos[1], pos[2]  # XZ plane coordinates
        fx, fz = force[1], force[2]    # XZ plane forces
        
        # Find influence region in XZ plane
        i_min = max(1, Int(floor((x_lag - 2*δh - grid.x[1]) / grid.dx)) + 1)
        i_max = min(nx, Int(ceil((x_lag + 2*δh - grid.x[1]) / grid.dx)) + 1)
        j_min = max(1, Int(floor((z_lag - 2*δh - grid.z[1]) / grid.dz)) + 1)
        j_max = min(nz, Int(ceil((z_lag + 2*δh - grid.z[1]) / grid.dz)) + 1)
        
        for j = j_min:j_max, i = i_min:i_max
            x_grid = grid.x[i]
            z_grid = grid.z[j]  # Use z coordinates for XZ plane
            
            # Compute delta function value for XZ plane
            δ_val = regularized_delta_2d(x_lag - x_grid, z_lag - z_grid, δh, grid.dx, grid.dz)
            
            # Spread force in XZ plane
            force_field[i, j][1] += fx * δ_val * grid.dx * grid.dz
            force_field[i, j][2] += fz * δ_val * grid.dx * grid.dz
        end
    end
end

function apply_force_spreading_3d!(force_field::Array{Vector{Float64},3}, 
                                  lagrangian_forces::Vector{Vector{Float64}},
                                  lagrangian_positions::Vector{Vector{Float64}},
                                  grid::StaggeredGrid, δh::Float64)
    """
    Spread Lagrangian forces to 3D Eulerian grid using regularized delta function.
    """
    nx, ny, nz = size(force_field)
    
    # Clear force field
    for k = 1:nz, j = 1:ny, i = 1:nx
        force_field[i, j, k] = [0.0, 0.0, 0.0]
    end
    
    # Spread forces from Lagrangian points to Eulerian grid
    for (pos, force) in zip(lagrangian_positions, lagrangian_forces)
        x_lag, y_lag, z_lag = pos[1], pos[2], pos[3]
        fx, fy, fz = force[1], force[2], force[3]
        
        # Find influence region
        i_min = max(1, Int(floor((x_lag - 2*δh - grid.x[1]) / grid.dx)) + 1)
        i_max = min(nx, Int(ceil((x_lag + 2*δh - grid.x[1]) / grid.dx)) + 1)
        j_min = max(1, Int(floor((y_lag - 2*δh - grid.y[1]) / grid.dy)) + 1)
        j_max = min(ny, Int(ceil((y_lag + 2*δh - grid.y[1]) / grid.dy)) + 1)
        k_min = max(1, Int(floor((z_lag - 2*δh - grid.z[1]) / grid.dz)) + 1)
        k_max = min(nz, Int(ceil((z_lag + 2*δh - grid.z[1]) / grid.dz)) + 1)
        
        for k = k_min:k_max, j = j_min:j_max, i = i_min:i_max
            x_grid = grid.x[i]
            y_grid = grid.y[j]
            z_grid = grid.z[k]
            
            # Compute 3D delta function value
            δ_val = regularized_delta_3d(x_lag - x_grid, y_lag - y_grid, z_lag - z_grid, 
                                       δh, grid.dx, grid.dy, grid.dz)
            
            # Spread force
            vol = grid.dx * grid.dy * grid.dz
            force_field[i, j, k][1] += fx * δ_val * vol
            force_field[i, j, k][2] += fy * δ_val * vol
            force_field[i, j, k][3] += fz * δ_val * vol
        end
    end
end

function regularized_delta_2d(dx::Float64, dz::Float64, δh::Float64, 
                             grid_dx::Float64, grid_dz::Float64)
    """
    2D regularized delta function (Peskin's 4-point function) for XZ plane.
    """
    
    # Normalize distances by delta width
    r_x = abs(dx) / δh
    r_z = abs(dz) / δh
    
    # 4-point regularized delta function (1D)
    function δ_1d(r::Float64)
        if r <= 1.0
            return 0.125 * (3 - 2*r + sqrt(1 + 4*r - 4*r^2))
        elseif r <= 2.0
            return 0.125 * (5 - 2*r - sqrt(-7 + 12*r - 4*r^2))
        else
            return 0.0
        end
    end
    
    # 2D delta function is product of 1D functions
    δ_val = δ_1d(r_x) * δ_1d(r_z) / (δh^2)
    
    return δ_val
end

function regularized_delta_3d(dx::Float64, dy::Float64, dz::Float64, δh::Float64, 
                             grid_dx::Float64, grid_dy::Float64, grid_dz::Float64)
    """
    3D regularized delta function (Peskin's 4-point function).
    """
    
    # Normalize distances by delta width
    r_x = abs(dx) / δh
    r_y = abs(dy) / δh
    r_z = abs(dz) / δh
    
    # 4-point regularized delta function (1D)
    function δ_1d(r::Float64)
        if r <= 1.0
            return 0.125 * (3 - 2*r + sqrt(1 + 4*r - 4*r^2))
        elseif r <= 2.0
            return 0.125 * (5 - 2*r - sqrt(-7 + 12*r - 4*r^2))
        else
            return 0.0
        end
    end
    
    # 3D delta function is product of 1D functions
    δ_val = δ_1d(r_x) * δ_1d(r_y) * δ_1d(r_z) / (δh^3)
    
    return δ_val
end