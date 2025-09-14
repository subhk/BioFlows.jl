# Note: ParametricBodies dependency removed; keep pure Julia rigid body types here

abstract type RigidBodyShape end

struct Circle <: RigidBodyShape
    radius::Float64
end

# Primary rigid body type must be defined before methods use it
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

# ============================================================
# Drag/Lift Coefficients for Rigid Bodies (2D XZ plane)
# ============================================================

"""
    compute_drag_lift_coefficients(body::RigidBody, grid, state, fluid; reference_velocity=1.0, reference_length=1.0, flow_direction=[1.0,0.0])

Compute approximate drag and lift coefficients for a rigid body in 2D XZ plane by integrating pressure and viscous stresses over a discretized boundary.
"""
function compute_drag_lift_coefficients(body::RigidBody, grid::StaggeredGrid,
                                       state::SolutionState, fluid::FluidProperties;
                                       reference_velocity::Float64 = 1.0,
                                       reference_length::Float64 = (body.shape isa Circle ? 2*(body.shape::Circle).radius : 1.0),
                                       flow_direction::Vector{Float64} = [1.0, 0.0])
    @assert grid.grid_type == TwoDimensional "Rigid-body coefficients: 2D XZ implementation"

    # Fluid properties
    ρ = fluid.ρ isa ConstantDensity ? fluid.ρ.ρ : error("Variable density not supported")
    μ = fluid.μ

    # Cell-centered velocity fields and their gradients
    u_cc = interpolate_u_to_cell_center(state.u, grid)
    w_cc = interpolate_w_to_cell_center(state.w, grid)  # w is z-velocity in XZ
    dudx = ddx(u_cc, grid); dudz = ddz(u_cc, grid)
    dwdx = ddx(w_cc, grid); dwdz = ddz(w_cc, grid)

    # Boundary discretization
    points = Vector{Vector{Float64}}()
    if body.shape isa Circle
        r = (body.shape::Circle).radius
        npts = max(32, Int(round(2π * r / min(grid.dx, grid.dz))))
        for k = 1:npts
            θ = 2π * (k-1) / npts
            x = body.center[1] + r * cos(θ + body.angle)
            z = (length(body.center) > 2 ? body.center[3] : body.center[2]) + r * sin(θ + body.angle)
            push!(points, [x, z])
        end
    elseif body.shape isa Square
        side = (body.shape::Square).side_length
        n_per = max(8, Int(round(side / min(grid.dx, grid.dz))))
        # Four edges
        for s in LinRange(-0.5*side, 0.5*side, n_per)
            # bottom/top in local
            for (lx,lz) in ((s,-0.5*side), (s,0.5*side))
                c = cos(body.angle); sθ = sin(body.angle)
                x = body.center[1] + c*lx - sθ*lz
                z = (length(body.center) > 2 ? body.center[3] : body.center[2]) + sθ*lx + c*lz
                push!(points, [x,z])
            end
        end
        for s in LinRange(-0.5*side, 0.5*side, n_per)
            # left/right
            for (lx,lz) in ((-0.5*side,s), (0.5*side,s))
                c = cos(body.angle); sθ = sin(body.angle)
                x = body.center[1] + c*lx - sθ*lz
                z = (length(body.center) > 2 ? body.center[3] : body.center[2]) + sθ*lx + c*lz
                push!(points, [x,z])
            end
        end
    else
        # Fallback: bounding-box sampling (coarse)
        push!(points, [body.center[1], (length(body.center)>2 ? body.center[3] : body.center[2])])
    end

    # Force accumulation
    Fp = zeros(2)  # pressure
    Fv = zeros(2)  # viscous

    # Flow and lift directions
    flow_dir = flow_direction / norm(flow_direction)
    lift_dir = [-flow_dir[2], flow_dir[1]]

    # Helper to clamp indices
    clampi(i, lo, hi) = max(lo, min(hi, i))

    # Integrate along boundary (trapezoidal weights)
    np = length(points)
    for k = 1:np
        xk, zk = points[k]; knext = k == np ? 1 : k+1
        ds = hypot(points[knext][1]-xk, points[knext][2]-zk)
        n = surface_normal_xz(body, xk, zk)
        # Tangent direction
        t̂ = [ -n[2], n[1] ]

        # Nearest cell indices
        ic = clampi(Int(round((xk - (grid.x[1])) / grid.dx)) + 1, 1, grid.nx)
        jc = clampi(Int(round((zk - (grid.z[1])) / grid.dz)) + 1, 1, grid.nz)

        # Local pressure and gradients (simple nearest-cell sample)
        p_loc = state.p[ic, jc]
        # Rate-of-strain tensor E ≈ [[dudx, 0.5(dudz+dwdx)],[0.5(dudz+dwdx), dwdz]] at cell center
        exx = dudx[ic, jc]
        ezz = dwdz[ic, jc]
        exz = 0.5 * (dudz[ic, jc] + dwdx[ic, jc])
        # Traction vector τ·n = -p n + 2μ E·n
        En = [exx*n[1] + exz*n[2], exz*n[1] + ezz*n[2]]
        traction = -p_loc .* n .+ 2μ .* En

        # Integrate
        Fp .+= (-p_loc .* n) .* ds
        Fv .+= (2μ .* En) .* ds
    end

    F = Fp + Fv
    # Project to drag and lift
    D = dot(F, flow_dir)
    L = dot(F, lift_dir)
    Dp = dot(Fp, flow_dir)
    Dv = dot(Fv, flow_dir)

    qref = 0.5 * ρ * reference_velocity^2
    Cd = D / (qref * reference_length)
    Cl = L / (qref * reference_length)
    Cd_p = Dp / (qref * reference_length)
    Cd_v = Dv / (qref * reference_length)

    return (Cd=Cd, Cl=Cl, Fx=F[1], Fz=F[2], Cd_pressure=Cd_p, Cd_viscous=Cd_v,
            reference_velocity=reference_velocity, reference_length=reference_length,
            dynamic_pressure=qref)
end

struct Square <: RigidBodyShape
    side_length::Float64
end

struct Rectangle <: RigidBodyShape
    width::Float64
    height::Float64
end

# RigidBody type defined earlier in file

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
    # Avoid mutating immutable RigidBody by creating a copy with updated id
    new_id = collection.n_bodies + 1
    new_body = RigidBody(body.shape, copy(body.center);
                         velocity=copy(body.velocity),
                         angular_velocity=body.angular_velocity,
                         angle=body.angle,
                         mass=body.mass,
                         moment_inertia=body.moment_inertia,
                         fixed=body.fixed,
                         id=new_id)
    push!(collection.bodies, new_body)
    collection.n_bodies = new_id
    return new_body
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
    """
    Compute integrated hydrodynamic force and torque on a rigid body in 2D XZ plane
    using a simple surface traction integration based on cell-centered fields.

    Returns ([Fx, Fz], Ty) where Ty is out-of-plane torque about the body center.
    """
    @assert grid.grid_type == TwoDimensional "compute_body_forces_2d uses XZ-plane convention"

    # Fluid properties and cell-centered gradients
    ρ = fluid.ρ isa ConstantDensity ? fluid.ρ.ρ : error("Variable density not supported")
    μ = fluid.μ
    u_cc = interpolate_u_to_cell_center(state.u, grid)
    w_cc = interpolate_v_to_cell_center(state.w, grid)  # w is z-velocity in 2D XZ
    dudx = ddx(u_cc, grid); dudz = ddz(u_cc, grid)
    dwdx = ddx(w_cc, grid); dwdz = ddz(w_cc, grid)

    # Discretize body boundary
    boundary_pts = Vector{Vector{Float64}}()
    if body.shape isa Circle
        r = (body.shape::Circle).radius
        npts = max(32, Int(round(2π * r / min(grid.dx, grid.dz))))
        for k = 1:npts
            θ = 2π * (k-1) / npts
            x = body.center[1] + r * cos(θ + body.angle)
            z = (length(body.center) > 2 ? body.center[3] : body.center[2]) + r * sin(θ + body.angle)
            push!(boundary_pts, [x, z])
        end
    elseif body.shape isa Square
        side = (body.shape::Square).side_length
        n_per = max(8, Int(round(side / min(grid.dx, grid.dz))))
        for s in LinRange(-0.5*side, 0.5*side, n_per)
            for (lx,lz) in ((s,-0.5*side), (s,0.5*side), (-0.5*side,s), (0.5*side,s))
                c = cos(body.angle); sθ = sin(body.angle)
                x = body.center[1] + c*lx - sθ*lz
                z = (length(body.center) > 2 ? body.center[3] : body.center[2]) + sθ*lx + c*lz
                push!(boundary_pts, [x,z])
            end
        end
    else
        # Fallback to a small ring around center
        push!(boundary_pts, [body.center[1], (length(body.center)>2 ? body.center[3] : body.center[2])])
    end

    # Accumulate force and torque
    Fx = 0.0; Fz = 0.0; Ty = 0.0
    np = length(boundary_pts)
    clampi(i, lo, hi) = max(lo, min(hi, i))
    x0 = body.center[1]
    z0 = (length(body.center) > 2 ? body.center[3] : body.center[2])

    for k = 1:np
        xk, zk = boundary_pts[k]; knext = k == np ? 1 : k+1
        ds = hypot(boundary_pts[knext][1]-xk, boundary_pts[knext][2]-zk)
        n = surface_normal_xz(body, xk, zk)
        # Nearest cell
        ic = clampi(Int(round((xk - grid.x[1]) / grid.dx)) + 1, 1, grid.nx)
        jc = clampi(Int(round((zk - grid.z[1]) / grid.dz)) + 1, 1, grid.nz)
        p_loc = state.p[ic, jc]
        exx = dudx[ic, jc]
        ezz = dwdz[ic, jc]
        exz = 0.5 * (dudz[ic, jc] + dwdx[ic, jc])
        En = [exx*n[1] + exz*n[2], exz*n[1] + ezz*n[2]]
        # Traction = -p n + 2μ E·n
        tx = -p_loc * n[1] + 2μ * En[1]
        tz = -p_loc * n[2] + 2μ * En[2]
        Fx += tx * ds
        Fz += tz * ds
        # Torque about (x0,z0) around y-axis: τ_y = r_x f_z - r_z f_x
        rx = xk - x0; rz = zk - z0
        Ty += rx * tz - rz * tx
    end

    return [Fx, Fz], Ty
end

"""
    compute_body_forces_3d(body, grid, state, fluid)

Compute integrated hydrodynamic force (Fx,Fy,Fz) and torque (Tx,Ty,Tz) on a rigid body in 3D
using a simple surface traction integration based on cell-centered fields.
"""
function compute_body_forces_3d(body::RigidBody, grid::StaggeredGrid, state::SolutionState,
                               fluid::FluidProperties)
    @assert grid.grid_type == ThreeDimensional "compute_body_forces_3d requires 3D grid"

    ρ = fluid.ρ isa ConstantDensity ? fluid.ρ.ρ : error("Variable density not supported")
    μ = fluid.μ

    # Cell-centered velocities and gradients
    u_cc, v_cc, w_cc = interpolate_to_cell_center_3d(state.u, state.v, state.w, grid)
    dudx = ddx(u_cc, grid); dudy = ddy(u_cc, grid); dudz = ddz(u_cc, grid)
    dvdx = ddx(v_cc, grid); dvdy = ddy(v_cc, grid); dvdz = ddz(v_cc, grid)
    dwdx = ddx(w_cc, grid); dwdy = ddy(w_cc, grid); dwdz = ddz(w_cc, grid)

    # Discretize surface: implement for spheres (Circle shape interpreted as sphere)
    Fx = 0.0; Fy = 0.0; Fz = 0.0
    Tx = 0.0; Ty = 0.0; Tz = 0.0
    xc = body.center[1]
    yc = body.center[2]
    zc = length(body.center) > 2 ? body.center[3] : 0.0

    if body.shape isa Circle
        r = (body.shape::Circle).radius
        nθ = max(24, Int(round(2π * r / min(grid.dx, grid.dy))))
        nφ = max(12, Int(round(π * r / grid.dz)))
        for iφ = 1:nφ-1  # exclude poles for now
            φ = π * iφ / nφ
            sinφ = sin(φ); cosφ = cos(φ)
            Δφ = π / nφ
            for iθ = 1:nθ
                θ = 2π * (iθ-1) / nθ
                Δθ = 2π / nθ
                cθ = cos(θ); sθ = sin(θ)
                # Surface point
                x = xc + r * sinφ * cθ
                y = yc + r * sinφ * sθ
                z = zc + r * cosφ
                # Outward normal on sphere
                n = [sinφ * cθ, sinφ * sθ, cosφ]
                # Area element on sphere
                dA = r^2 * sinφ * Δθ * Δφ

                # Nearest cell indices (clamped)
                ic = clamp(Int(round((x - grid.x[1]) / grid.dx)) + 1, 1, grid.nx)
                jc = clamp(Int(round((y - grid.y[1]) / grid.dy)) + 1, 1, grid.ny)
                kc = clamp(Int(round((z - grid.z[1]) / grid.dz)) + 1, 1, grid.nz)

                p_loc = state.p[ic, jc, kc]
                # Symmetric rate-of-strain tensor times n: (E·n)
                Ex = dudx[ic,jc,kc]*n[1] + 0.5*(dudy[ic,jc,kc]+dvdx[ic,jc,kc])*n[2] + 0.5*(dudz[ic,jc,kc]+dwdx[ic,jc,kc])*n[3]
                Ey = 0.5*(dvdx[ic,jc,kc]+dudy[ic,jc,kc])*n[1] + dvdy[ic,jc,kc]*n[2] + 0.5*(dvdz[ic,jc,kc]+dwdy[ic,jc,kc])*n[3]
                Ez = 0.5*(dwdx[ic,jc,kc]+dudz[ic,jc,kc])*n[1] + 0.5*(dwdy[ic,jc,kc]+dvdz[ic,jc,kc])*n[2] + dwdz[ic,jc,kc]*n[3]
                # Traction vector
                tx = -p_loc*n[1] + 2μ*Ex
                ty = -p_loc*n[2] + 2μ*Ey
                tz = -p_loc*n[3] + 2μ*Ez

                Fx += tx * dA
                Fy += ty * dA
                Fz += tz * dA
                rx = x - xc; ry = y - yc; rz = z - zc
                # Torque r × f
                Tx += ry*tz - rz*ty
                Ty += rz*tx - rx*tz
                Tz += rx*ty - ry*tx
            end
        end
    else
        @warn "compute_body_forces_3d: shape $(typeof(body.shape)) not yet supported; returning zeros"
    end

    return [Fx, Fy, Fz], [Tx, Ty, Tz]
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

# bodies_mask_* implementations live in immersed/immersed_boundary.jl to avoid duplication
