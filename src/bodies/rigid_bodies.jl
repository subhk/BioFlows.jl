using ParametricBodies

abstract type RigidBodyShape end

struct Circle <: RigidBodyShape
    radius::Float64
end

struct Square <: RigidBodyShape
    side_length::Float64
end

struct Rectangle <: RigidBodyShape
    width::Float64
    height::Float64
end

struct RigidBody <: AbstractBody
    shape::RigidBodyShape
    center::Vector{Float64}
    velocity::Vector{Float64}
    angular_velocity::Float64
    angle::Float64
    mass::Float64
    moment_inertia::Float64
    fixed::Bool
    id::Int
end

function RigidBody(shape::RigidBodyShape, center::Vector{Float64}; 
                  velocity=zeros(length(center)), angular_velocity=0.0, angle=0.0,
                  mass=1.0, moment_inertia=1.0, fixed=false, id=1)
    RigidBody(shape, center, velocity, angular_velocity, angle, mass, moment_inertia, fixed, id)
end

function Circle(radius::Float64, center::Vector{Float64}; kwargs...)
    RigidBody(Circle(radius), center; kwargs...)
end

function Square(side_length::Float64, center::Vector{Float64}; kwargs...)
    RigidBody(Square(side_length), center; kwargs...)
end

function Rectangle(width::Float64, height::Float64, center::Vector{Float64}; kwargs...)
    RigidBody(Rectangle(width, height), center; kwargs...)
end

mutable struct RigidBodyCollection
    bodies::Vector{RigidBody}
    n_bodies::Int
end

function RigidBodyCollection()
    RigidBodyCollection(RigidBody[], 0)
end

function add_body!(collection::RigidBodyCollection, body::RigidBody)
    push!(collection.bodies, body)
    collection.n_bodies += 1
    body.id = collection.n_bodies
end

function is_inside(body::RigidBody, x::Float64, y::Float64)
    return is_inside(body.shape, body.center, body.angle, x, y)
end

function is_inside(body::RigidBody, x::Float64, y::Float64, z::Float64)
    return is_inside(body.shape, body.center, body.angle, x, y, z)
end

# XZ plane version for 2D flows
function is_inside_xz(body::RigidBody, x::Float64, z::Float64)
    return is_inside_xz(body.shape, body.center, body.angle, x, z)
end

function is_inside(shape::Circle, center::Vector{Float64}, angle::Float64, x::Float64, y::Float64)
    dx = x - center[1]
    dy = y - center[2]
    return dx^2 + dy^2 <= shape.radius^2
end

function is_inside(shape::Square, center::Vector{Float64}, angle::Float64, x::Float64, y::Float64)
    # Transform to body-centered coordinates with rotation
    cos_θ = cos(angle)
    sin_θ = sin(angle)
    
    dx = x - center[1]
    dy = y - center[2]
    
    # Rotate coordinates
    dx_rot = cos_θ * dx + sin_θ * dy
    dy_rot = -sin_θ * dx + cos_θ * dy
    
    half_side = shape.side_length / 2
    return abs(dx_rot) <= half_side && abs(dy_rot) <= half_side
end

function is_inside(shape::Rectangle, center::Vector{Float64}, angle::Float64, x::Float64, y::Float64)
    cos_θ = cos(angle)
    sin_θ = sin(angle)
    
    dx = x - center[1]
    dy = y - center[2]
    
    dx_rot = cos_θ * dx + sin_θ * dy
    dy_rot = -sin_θ * dx + cos_θ * dy
    
    return abs(dx_rot) <= shape.width/2 && abs(dy_rot) <= shape.height/2
end

# XZ plane versions for 2D flows
function is_inside_xz(shape::Circle, center::Vector{Float64}, angle::Float64, x::Float64, z::Float64)
    dx = x - center[1]
    dz = z - (length(center) > 2 ? center[3] : center[2])  # Use z coordinate from center[3] or center[2]
    return dx^2 + dz^2 <= shape.radius^2
end

function is_inside_xz(shape::Square, center::Vector{Float64}, angle::Float64, x::Float64, z::Float64)
    # Transform to body-centered coordinates with rotation (in XZ plane)
    cos_θ = cos(angle)
    sin_θ = sin(angle)
    
    dx = x - center[1]
    dz = z - (length(center) > 2 ? center[3] : center[2])  # Use z coordinate
    
    # Rotate coordinates in XZ plane
    dx_rot = cos_θ * dx + sin_θ * dz
    dz_rot = -sin_θ * dx + cos_θ * dz
    
    half_side = shape.side_length / 2
    return abs(dx_rot) <= half_side && abs(dz_rot) <= half_side
end

function is_inside_xz(shape::Rectangle, center::Vector{Float64}, angle::Float64, x::Float64, z::Float64)
    cos_θ = cos(angle)
    sin_θ = sin(angle)
    
    dx = x - center[1]
    dz = z - (length(center) > 2 ? center[3] : center[2])  # Use z coordinate
    
    dx_rot = cos_θ * dx + sin_θ * dz
    dz_rot = -sin_θ * dx + cos_θ * dz
    
    return abs(dx_rot) <= shape.width/2 && abs(dz_rot) <= shape.height/2
end

function is_inside(shape::Circle, center::Vector{Float64}, angle::Float64, 
                  x::Float64, y::Float64, z::Float64)
    dx = x - center[1]
    dy = y - center[2]
    dz = length(center) > 2 ? z - center[3] : z
    return dx^2 + dy^2 + dz^2 <= shape.radius^2
end

function distance_to_surface(body::RigidBody, x::Float64, y::Float64)
    return distance_to_surface(body.shape, body.center, body.angle, x, y)
end

function distance_to_surface_3d(body::RigidBody, x::Float64, y::Float64, z::Float64)
    return distance_to_surface_3d(body.shape, body.center, body.angle, x, y, z)
end

# XZ plane version for 2D flows
function distance_to_surface_xz(body::RigidBody, x::Float64, z::Float64)
    return distance_to_surface_xz(body.shape, body.center, body.angle, x, z)
end

function distance_to_surface(shape::Circle, center::Vector{Float64}, angle::Float64, 
                           x::Float64, y::Float64)
    dx = x - center[1]
    dy = y - center[2]
    distance_to_center = sqrt(dx^2 + dy^2)
    return distance_to_center - shape.radius
end

function distance_to_surface(shape::Square, center::Vector{Float64}, angle::Float64,
                           x::Float64, y::Float64)
    cos_θ = cos(angle)
    sin_θ = sin(angle)
    
    dx = x - center[1]
    dy = y - center[2]
    
    dx_rot = cos_θ * dx + sin_θ * dy
    dy_rot = -sin_θ * dx + cos_θ * dy
    
    half_side = shape.side_length / 2
    
    # Distance to closest edge
    dist_x = abs(dx_rot) - half_side
    dist_y = abs(dy_rot) - half_side
    
    if dist_x <= 0 && dist_y <= 0
        return max(dist_x, dist_y)  # Inside
    elseif dist_x > 0 && dist_y <= 0
        return dist_x
    elseif dist_x <= 0 && dist_y > 0
        return dist_y
    else
        return sqrt(dist_x^2 + dist_y^2)  # Corner distance
    end
end

# XZ plane distance functions for 2D flows
function distance_to_surface_xz(shape::Circle, center::Vector{Float64}, angle::Float64, 
                               x::Float64, z::Float64)
    dx = x - center[1]
    dz = z - (length(center) > 2 ? center[3] : center[2])  # Use z coordinate
    distance_to_center = sqrt(dx^2 + dz^2)
    return distance_to_center - shape.radius
end

function distance_to_surface_xz(shape::Square, center::Vector{Float64}, angle::Float64,
                               x::Float64, z::Float64)
    cos_θ = cos(angle)
    sin_θ = sin(angle)
    
    dx = x - center[1]
    dz = z - (length(center) > 2 ? center[3] : center[2])  # Use z coordinate
    
    dx_rot = cos_θ * dx + sin_θ * dz
    dz_rot = -sin_θ * dx + cos_θ * dz
    
    half_side = shape.side_length / 2
    
    # Distance to closest edge
    dist_x = abs(dx_rot) - half_side
    dist_z = abs(dz_rot) - half_side
    
    if dist_x <= 0 && dist_z <= 0
        return max(dist_x, dist_z)  # Inside
    elseif dist_x > 0 && dist_z <= 0
        return dist_x
    elseif dist_x <= 0 && dist_z > 0
        return dist_z
    else
        return sqrt(dist_x^2 + dist_z^2)  # Corner distance
    end
end

function distance_to_surface_xz(shape::Rectangle, center::Vector{Float64}, angle::Float64,
                               x::Float64, z::Float64)
    cos_θ = cos(angle)
    sin_θ = sin(angle)
    
    dx = x - center[1]
    dz = z - (length(center) > 2 ? center[3] : center[2])  # Use z coordinate
    
    dx_rot = cos_θ * dx + sin_θ * dz
    dz_rot = -sin_θ * dx + cos_θ * dz
    
    # Distance to closest edge
    dist_x = abs(dx_rot) - shape.width/2
    dist_z = abs(dz_rot) - shape.height/2
    
    if dist_x <= 0 && dist_z <= 0
        return max(dist_x, dist_z)  # Inside
    elseif dist_x > 0 && dist_z <= 0
        return dist_x
    elseif dist_x <= 0 && dist_z > 0
        return dist_z
    else
        return sqrt(dist_x^2 + dist_z^2)  # Corner distance
    end
end

# 3D distance functions
function distance_to_surface_3d(shape::Circle, center::Vector{Float64}, angle::Float64, 
                               x::Float64, y::Float64, z::Float64)
    # For sphere in 3D
    dx = x - center[1]
    dy = y - center[2]
    dz = length(center) > 2 ? z - center[3] : z
    distance_to_center = sqrt(dx^2 + dy^2 + dz^2)
    return distance_to_center - shape.radius
end

function distance_to_surface_3d(shape::Square, center::Vector{Float64}, angle::Float64,
                               x::Float64, y::Float64, z::Float64)
    # For cube in 3D (simplified - no rotation about x,y axes)
    dx = abs(x - center[1])
    dy = abs(y - center[2])
    dz = length(center) > 2 ? abs(z - center[3]) : abs(z)
    
    half_side = shape.side_length / 2
    
    # Distance to cube surface
    dist_x = dx - half_side
    dist_y = dy - half_side
    dist_z = dz - half_side
    
    if dist_x <= 0 && dist_y <= 0 && dist_z <= 0
        return max(dist_x, dist_y, dist_z)  # Inside
    else
        # Outside - compute distance to nearest surface/edge/corner
        dist_x = max(0, dist_x)
        dist_y = max(0, dist_y)
        dist_z = max(0, dist_z)
        return sqrt(dist_x^2 + dist_y^2 + dist_z^2)
    end
end

function distance_to_surface_3d(shape::Rectangle, center::Vector{Float64}, angle::Float64,
                               x::Float64, y::Float64, z::Float64)
    # For rectangular box in 3D (simplified - no rotation)
    dx = abs(x - center[1])
    dy = abs(y - center[2])
    dz = length(center) > 2 ? abs(z - center[3]) : abs(z)
    
    # Distance to rectangular box surface
    dist_x = dx - shape.width/2
    dist_y = dy - shape.height/2
    dist_z = dz - shape.width/2  # Assume depth = width for simplicity
    
    if dist_x <= 0 && dist_y <= 0 && dist_z <= 0
        return max(dist_x, dist_y, dist_z)  # Inside
    else
        # Outside - compute distance to nearest surface/edge/corner
        dist_x = max(0, dist_x)
        dist_y = max(0, dist_y)
        dist_z = max(0, dist_z)
        return sqrt(dist_x^2 + dist_y^2 + dist_z^2)
    end
end

function surface_normal(body::RigidBody, x::Float64, y::Float64)
    return surface_normal(body.shape, body.center, body.angle, x, y)
end

# XZ plane version for 2D flows
function surface_normal_xz(body::RigidBody, x::Float64, z::Float64)
    return surface_normal_xz(body.shape, body.center, body.angle, x, z)
end

function surface_normal(shape::Circle, center::Vector{Float64}, angle::Float64,
                       x::Float64, y::Float64)
    dx = x - center[1]
    dy = y - center[2]
    distance = sqrt(dx^2 + dy^2)
    
    if distance < 1e-12
        return [1.0, 0.0]  # Arbitrary direction for center point
    end
    
    return [dx/distance, dy/distance]
end

function surface_normal(shape::Square, center::Vector{Float64}, angle::Float64,
                       x::Float64, y::Float64)
    cos_θ = cos(angle)
    sin_θ = sin(angle)
    
    dx = x - center[1]
    dy = y - center[2]
    
    dx_rot = cos_θ * dx + sin_θ * dy
    dy_rot = -sin_θ * dx + cos_θ * dy
    
    half_side = shape.side_length / 2
    
    # Determine which face is closest
    if abs(dx_rot) > abs(dy_rot)
        # Closest to left/right face
        nx_local = sign(dx_rot)
        ny_local = 0.0
    else
        # Closest to top/bottom face
        nx_local = 0.0
        ny_local = sign(dy_rot)
    end
    
    # Rotate normal back to global coordinates
    nx = cos_θ * nx_local - sin_θ * ny_local
    ny = sin_θ * nx_local + cos_θ * ny_local
    
    return [nx, ny]
end

# XZ plane surface normal functions for 2D flows
function surface_normal_xz(shape::Circle, center::Vector{Float64}, angle::Float64,
                          x::Float64, z::Float64)
    dx = x - center[1]
    dz = z - (length(center) > 2 ? center[3] : center[2])  # Use z coordinate
    distance = sqrt(dx^2 + dz^2)
    
    if distance < 1e-12
        return [1.0, 0.0]  # Arbitrary direction for center point
    end
    
    return [dx/distance, dz/distance]
end

function surface_normal_xz(shape::Square, center::Vector{Float64}, angle::Float64,
                          x::Float64, z::Float64)
    cos_θ = cos(angle)
    sin_θ = sin(angle)
    
    dx = x - center[1]
    dz = z - (length(center) > 2 ? center[3] : center[2])  # Use z coordinate
    
    dx_rot = cos_θ * dx + sin_θ * dz
    dz_rot = -sin_θ * dx + cos_θ * dz
    
    half_side = shape.side_length / 2
    
    # Determine which face is closest
    if abs(dx_rot) - half_side >= abs(dz_rot) - half_side
        # Closer to vertical faces
        nx_local = sign(dx_rot)
        nz_local = 0.0
    else
        # Closer to horizontal faces
        nx_local = 0.0
        nz_local = sign(dz_rot)
    end
    
    # Rotate normal back to global coordinates
    nx = cos_θ * nx_local - sin_θ * nz_local
    nz = sin_θ * nx_local + cos_θ * nz_local
    
    return [nx, nz]
end

function surface_normal_xz(shape::Rectangle, center::Vector{Float64}, angle::Float64,
                          x::Float64, z::Float64)
    cos_θ = cos(angle)
    sin_θ = sin(angle)
    
    dx = x - center[1]
    dz = z - (length(center) > 2 ? center[3] : center[2])  # Use z coordinate
    
    dx_rot = cos_θ * dx + sin_θ * dz
    dz_rot = -sin_θ * dx + cos_θ * dz
    
    # Determine which face is closest
    dist_x = abs(dx_rot) - shape.width/2
    dist_z = abs(dz_rot) - shape.height/2
    
    if dist_x >= dist_z
        # Closer to vertical faces
        nx_local = sign(dx_rot)
        nz_local = 0.0
    else
        # Closer to horizontal faces
        nx_local = 0.0
        nz_local = sign(dz_rot)
    end
    
    # Rotate normal back to global coordinates
    nx = cos_θ * nx_local - sin_θ * nz_local
    nz = sin_θ * nx_local + cos_θ * nz_local
    
    return [nx, nz]
end

function update_body_motion!(body::RigidBody, dt::Float64, forces::Vector{Float64}, torque::Float64)
    if body.fixed
        return
    end
    
    # Update linear motion
    acceleration = forces / body.mass
    body.velocity .+= acceleration .* dt
    body.center .+= body.velocity .* dt
    
    # Update angular motion
    angular_acceleration = torque / body.moment_inertia
    body.angular_velocity += angular_acceleration * dt
    body.angle += body.angular_velocity * dt
    
    # Keep angle in [0, 2π]
    body.angle = mod(body.angle, 2π)
end

function compute_body_forces_2d(body::RigidBody, grid::StaggeredGrid, state::SolutionState,
                               fluid::FluidProperties)
    # This function computes forces and torque on the body due to fluid pressure and viscosity
    # Implementation would involve surface integration over the body boundary
    
    force_x = 0.0
    force_y = 0.0
    torque = 0.0
    
    # Surface force integration (simplified approach)
    # In practice, this would use the immersed boundary method results
    
    return [force_x, force_y], torque
end

function get_body_velocity_at_point(body::RigidBody, x::Float64, y::Float64)
    # Velocity of body surface at point (x,y) due to translation + rotation
    dx = x - body.center[1]
    dy = y - body.center[2]
    
    # Translational velocity + rotational velocity
    u_body = body.velocity[1] - body.angular_velocity * dy
    v_body = body.velocity[2] + body.angular_velocity * dx
    
    return [u_body, v_body]
end

# XZ plane version for 2D flows
function get_body_velocity_at_point_xz(body::RigidBody, x::Float64, z::Float64)
    # Velocity of body surface at point (x,z) due to translation + rotation in XZ plane
    dx = x - body.center[1]
    dz = z - (length(body.center) > 2 ? body.center[3] : body.center[2])  # Use z coordinate
    
    # Translational velocity + rotational velocity (rotation about y-axis)
    u_body = body.velocity[1] + body.angular_velocity * dz  # u + ω × dz
    w_body = (length(body.velocity) > 2 ? body.velocity[3] : body.velocity[2]) - body.angular_velocity * dx  # w - ω × dx
    
    return [u_body, w_body]  # [u, w] velocities in XZ plane
end

function bodies_mask_2d(bodies::RigidBodyCollection, grid::StaggeredGrid)
    nx, nz = grid.nx, grid.nz  # Use XZ plane for 2D
    mask = falses(nx, nz)
    
    for j = 1:nz, i = 1:nx
        x = grid.x[i]
        z = grid.z[j]  # Use z coordinate for XZ plane
        
        for body in bodies.bodies
            if is_inside_xz(body, x, z)  # Use XZ plane version
                mask[i, j] = true
                break
            end
        end
    end
    
    return mask
end

function bodies_mask_3d(bodies::RigidBodyCollection, grid::StaggeredGrid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    mask = falses(nx, ny, nz)
    
    for k = 1:nz, j = 1:ny, i = 1:nx
        x = grid.x[i]
        y = grid.y[j]
        z = grid.z[k]
        
        for body in bodies.bodies
            if is_inside(body, x, y, z)
                mask[i, j, k] = true
                break
            end
        end
    end
    
    return mask
end