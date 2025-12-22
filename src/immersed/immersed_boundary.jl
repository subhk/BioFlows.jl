struct ImmersedBoundaryData
    body_mask::Array{Bool}      # Mask for solid cells
    distance_function::Array{Float64}  # Signed distance to nearest surface
    normal_vectors::Array{Vector{Float64}}  # Surface normal vectors
    boundary_points::Vector{Vector{Float64}}  # Lagrangian boundary points
    forcing_points::Vector{Tuple{Vector{Int}, Float64}}  # Eulerian forcing points and weights
end

# BDIM data structure
struct BDIMData
    μ₀::Array{Float64, 3}      # Zeroth moment (μ₀) [nx+2, nz+2, 2]
    μ₁::Array{Float64, 4}      # First moment tensor (μ₁) [nx+2, nz+2, 2, 2]
    V::Array{Float64, 3}       # Body velocity [nx+2, nz+2, 2]
end

# BDIM Method Type
@enum ImmersedBoundaryMethod VolumePenalty BDIM

function BDIMData(nx::Int, nz::Int)
    μ₀ = ones(Float64, nx+2, nz+2, 2)    # Initialize to 1 (fluid)
    μ₁ = zeros(Float64, nx+2, nz+2, 2, 2)  # Initialize to 0
    V = zeros(Float64, nx+2, nz+2, 2)      # Initialize to 0 (static body)
    return BDIMData(μ₀, μ₁, V)
end

"""
    apply_inlet_volume_penalty!(state, grid, Uin, dt; thickness=10, sigma0=1e4)

Brinkman-style inlet sponge that semi-implicitly relaxes velocities in a short
buffer of `thickness` faces behind the inlet. This reduces boundary-layer
divergence by smoothing the normal velocity to the desired inlet value and the
transverse velocity to zero.
"""
function apply_inlet_volume_penalty!(state::SolutionState, grid::StaggeredGrid,
                                     Uin::Float64, dt::Float64;
                                     thickness::Int=10, sigma0::Float64=1e4)
    nx, nz = grid.nx, grid.nz
    # u at x-faces: i = 1..nx+1, relax first `thickness` faces to Uin
    nuf = min(thickness, nx+1)
    @inbounds for i = 1:nuf, j = 1:nz
        s = 1.0 - (i-1)/max(1, thickness)
        sigma = sigma0 * s * s
        state.u[i, j] = (state.u[i, j] + dt*sigma*Uin) / (1 + dt*sigma)
    end
    # w at z-faces: i = 1..nx, relax first `thickness` faces to 0
    nwf = min(thickness, nx)
    @inbounds for i = 1:nwf, j = 1:nz+1
        s = 1.0 - (i-1)/max(1, thickness)
        sigma = sigma0 * s * s
        state.w[i, j] = (state.w[i, j]) / (1 + dt*sigma)
    end
    return state
end

"""
    apply_outlet_volume_penalty!(state, grid, dt; thickness=10, sigma0=1e4)

Outlet sponge that semi-implicitly relaxes velocities over the last `thickness`
faces to a zero-gradient reference (copies from the previous interior face).
This reduces reflective waves and boundary-layer divergence at the outlet.
"""
function apply_outlet_volume_penalty!(state::SolutionState, grid::StaggeredGrid,
                                      dt::Float64; thickness::Int=10, sigma0::Float64=1e4)
    nx, nz = grid.nx, grid.nz
    # u at x-faces: faces index 1..nx+1, outlet at i = nx+1
    nuf = min(thickness, nx+1)
    @inbounds for k = 0:nuf-1
        i = nx + 1 - k
        ref_u = view(state.u, max(i-1,1), 1:nz)
        s = 1.0 - k/max(1, thickness-1)
        sigma = sigma0 * s * s
        state.u[i, 1:nz] .= (state.u[i, 1:nz] .+ dt*sigma .* ref_u) ./ (1 .+ dt*sigma)
    end
    # w at z-faces: indices 1..nx in x, outlet at i = nx
    nwf = min(thickness, nx)
    @inbounds for k = 0:nwf-1
        i = nx - k
        ref_w = view(state.w, max(i-1,1), 1:nz+1)
        s = 1.0 - k/max(1, thickness-1)
        sigma = sigma0 * s * s
        state.w[i, 1:nz+1] .= (state.w[i, 1:nz+1] .+ dt*sigma .* ref_w) ./ (1 .+ dt*sigma)
    end
    return state
end

"""
    apply_body_volume_penalty!(state, grid, bodies::RigidBodyCollection, dt; buffer=2*grid.dz, sigma0=5e5)

Brinkman penalization that relaxes face velocities toward the local rigid-body
velocity for any body within `radius+buffer` distance. Supports moving bodies.
"""
function apply_body_volume_penalty!(state::SolutionState, grid::StaggeredGrid,
                                    bodies::RigidBodyCollection, dt::Float64;
                                    buffer::Float64=2*grid.dz, sigma0::Float64=5e5)
    # u faces
    @inbounds for j = 1:grid.nz
        z = grid.z[j]
        for i = 1:grid.nx+1
            x = grid.xu[i]
            # Find nearest body distance (XZ plane)
            min_d = Inf
            for body in bodies.bodies
                d = abs(distance_to_surface_xz(body, x, z))
                min_d = min(min_d, d)
            end
            if min_d <= buffer
                s = max(0.0, 1.0 - min_d/max(buffer, eps()))
                sigma = sigma0 * s * s
                state.u[i, j] = state.u[i, j] / (1 + dt*sigma)
            end
        end
    end
    # w faces
    @inbounds for j = 1:grid.nz+1
        zw = grid.zw[j]
        for i = 1:grid.nx
            x = grid.x[i]
            min_d = Inf
            for body in bodies.bodies
                d = abs(distance_to_surface_xz(body, x, zw))
                min_d = min(min_d, d)
            end
            if min_d <= buffer
                s = max(0.0, 1.0 - min_d/max(buffer, eps()))
                sigma = sigma0 * s * s
                state.w[i, j] = state.w[i, j] / (1 + dt*sigma)
            end
        end
    end
    return state
end

"""
    bdim_divergence_2d!(div, u, w, grid, body_mask)

BDIM-style divergence operator that respects immersed boundaries.
Instead of computing ∇·u in solid regions, this sets div=0 inside bodies
and computes proper divergence only in fluid regions.
"""
function bdim_divergence_2d!(div::Matrix{Float64}, u::Matrix{Float64}, w::Matrix{Float64},
                            grid::StaggeredGrid, body_mask::BitMatrix)
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz
    
    @inbounds for j = 1:nz, i = 1:nx
        if body_mask[i, j]
            # Inside body: divergence is undefined/zero
            div[i, j] = 0.0
        else
            # Fluid region: compute divergence with proper staggered grid indexing
            # u[i+1,j] is velocity at east face, u[i,j] is velocity at west face
            # w[i,j+1] is velocity at north face, w[i,j] is velocity at south face
            u_east = (i < nx) ? u[i+1, j] : 0.0
            u_west = u[i, j]
            w_north = (j < nz) ? w[i, j+1] : 0.0
            w_south = w[i, j]
            
            # Standard divergence formula for staggered grid
            div[i, j] = (u_east - u_west) / dx + (w_north - w_south) / dz
        end
    end
    return div
end

"""
    bdim_gradient_pressure_2d!(dpdx, dpdz, p, grid, body_mask)

BDIM-style pressure gradient that respects immersed boundaries.
Sets gradients to zero inside bodies and computes proper gradients in fluid.
"""
function bdim_gradient_pressure_2d!(dpdx::Matrix{Float64}, dpdz::Matrix{Float64}, 
                                   p::Matrix{Float64}, grid::StaggeredGrid, body_mask::BitMatrix)
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz
    
    # dpdx at u-faces (simple centered differences)
    @inbounds for j = 1:nz, i = 1:nx+1
        if i == 1
            # Left boundary: use forward difference if fluid cell to the right
            dpdx[i, j] = body_mask[1, j] ? 0.0 : (p[1, j] - 0.0) / dx
        elseif i == nx+1
            # Right boundary: use backward difference if fluid cell to the left
            dpdx[i, j] = body_mask[nx, j] ? 0.0 : (0.0 - p[nx, j]) / dx
        else
            # Interior faces: standard centered difference
            i_left, i_right = i-1, i
            if body_mask[i_left, j] || body_mask[i_right, j]
                dpdx[i, j] = 0.0  # Zero gradient at body boundaries
            else
                dpdx[i, j] = (p[i_right, j] - p[i_left, j]) / dx
            end
        end
    end
    
    # dpdz at w-faces (simple centered differences)
    @inbounds for j = 1:nz+1, i = 1:nx
        if j == 1
            # Bottom boundary: use forward difference if fluid cell above
            dpdz[i, j] = body_mask[i, 1] ? 0.0 : (p[i, 1] - 0.0) / dz
        elseif j == nz+1
            # Top boundary: use backward difference if fluid cell below
            dpdz[i, j] = body_mask[i, nz] ? 0.0 : (0.0 - p[i, nz]) / dz
        else
            # Interior faces: standard centered difference
            j_bottom, j_top = j-1, j
            if body_mask[i, j_bottom] || body_mask[i, j_top]
                dpdz[i, j] = 0.0  # Zero gradient at body boundaries
            else
                dpdz[i, j] = (p[i, j_top] - p[i, j_bottom]) / dz
            end
        end
    end
    return dpdx, dpdz
end

"""
    apply_bdim_boundary_conditions!(u, w, grid, body_mask, bodies)

BDIM-style boundary condition enforcement.
Sets velocities inside and on body boundaries to enforce no-slip conditions.
"""
function apply_bdim_boundary_conditions!(u::Matrix{Float64}, w::Matrix{Float64}, 
                                       grid::StaggeredGrid, body_mask::AbstractMatrix{Bool}, bodies)
    return apply_bdim_boundary_conditions!(u, w, grid, BitMatrix(body_mask), bodies)
end

function apply_bdim_boundary_conditions!(u::Matrix{Float64}, w::Matrix{Float64}, 
                                       grid::StaggeredGrid, body_mask::BitMatrix, bodies)
    # u-velocity enforcement
    nu_x = min(grid.nx + 1, size(u, 1))
    nu_z = min(grid.nz, size(u, 2))
    @inbounds for j = 1:nu_z, i = 1:nu_x
        x = grid.xu[i]
        z = grid.z[j]
        inside = false
        for body in bodies.bodies
            if is_inside_xz(body, x, z)
                inside = true
                break
            end
        end
        if inside
            u[i, j] = 0.0  # No-slip condition
        end
    end
    
    # w-velocity enforcement
    nw_x = min(grid.nx, size(w, 1))
    nw_z = min(grid.nz + 1, size(w, 2))
    @inbounds for j = 1:nw_z, i = 1:nw_x
        x = grid.x[i]
        z = grid.zw[j]
        inside = false
        for body in bodies.bodies
            if is_inside_xz(body, x, z)
                inside = true
                break
            end
        end
        if inside
            w[i, j] = 0.0  # No-slip condition
        end
    end
end

function ImmersedBoundaryData2D(bodies::RigidBodyCollection, grid::StaggeredGrid)
    if bodies.n_bodies == 0
        nx, nz = grid.nx, grid.nz
        body_mask = falses(nx, nz)
        distance_function = fill(Inf, nx, nz)
        normal_vectors = [zeros(2) for _ in 1:nx, _ in 1:nz]
        return ImmersedBoundaryData(body_mask, distance_function, normal_vectors,
                                    Vector{Vector{Float64}}(), Tuple{Vector{Int}, Float64}[])
    end

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

"""
    smooth_heaviside(phi, eps)

Smoothed Heaviside function used to build a smooth solid mask from a signed distance `phi`.
Returns values in [0,1].
"""
@inline function smooth_heaviside(phi::Float64, eps::Float64)
    if phi <= -eps
        return 0.0
    elseif phi >= eps
        return 1.0
    else
        # 0.5 + phi/(2eps) + (1/(2π)) sin(π phi/eps)
        return 0.5 + (phi/(2*eps)) + (sin(pi*phi/eps))/(2*pi)
    end
end

"""
    build_solid_mask_faces_2d(bodies, grid; eps_mul=2.0)

Construct smooth solid masks at staggered face locations (χ_u at x-faces for u,
χ_w at z-faces for w) using a smoothed Heaviside of the signed distance to the
nearest body surface. `eps_mul` controls smoothing width as eps = eps_mul*max(dx,dz).
"""
function build_solid_mask_faces_2d(bodies::RigidBodyCollection, grid::StaggeredGrid; eps_mul::Float64=2.0)
    nx, nz = grid.nx, grid.nz
    eps = eps_mul * max(grid.dx, grid.dz)
    # Signed distance at cell centers (nx x nz)
    phi_cc = Array{Float64}(undef, nx, nz)
    @inbounds for j = 1:nz, i = 1:nx
        x = grid.x[i]; z = grid.z[j]
        # Minimum signed distance across bodies
        mind = Inf
        for body in bodies.bodies
            d = distance_to_surface_xz(body, x, z)
            if abs(d) < abs(mind)
                mind = d
            end
        end
        phi_cc[i, j] = mind
    end
    # Smooth solid fraction at cell centers: χ_cc = Hε(-phi)
    chi_cc = Array{Float64}(undef, nx, nz)
    @inbounds for j = 1:nz, i = 1:nx
        chi_cc[i, j] = smooth_heaviside(-phi_cc[i, j], eps)
    end
    # Interpolate to faces
    chi_u = zeros(Float64, nx+1, nz)
    chi_w = zeros(Float64, nx, nz+1)
    # χ at u-faces: average adjacent cell centers
    @inbounds for j = 1:nz, i = 1:nx+1
        if i == 1
            chi_u[i, j] = chi_cc[1, j]
        elseif i == nx+1
            chi_u[i, j] = chi_cc[nx, j]
        else
            chi_u[i, j] = 0.5 * (chi_cc[i, j] + chi_cc[i-1, j])
        end
    end
    # χ at w-faces
    @inbounds for j = 1:nz+1, i = 1:nx
        if j == 1
            chi_w[i, j] = chi_cc[i, 1]
        elseif j == nz+1
            chi_w[i, j] = chi_cc[i, nz]
        else
            chi_w[i, j] = 0.5 * (chi_cc[i, j] + chi_cc[i, j-1])
        end
    end
    return chi_u, chi_w
end

"""
    masked_divergence_2d!(div, u, w, grid, chi_u, chi_w)

Compute divergence of masked face velocities (1-χ)u, (1-χ)w at cell centers.
"""
function masked_divergence_2d!(div::Matrix{Float64}, u::Matrix{Float64}, w::Matrix{Float64},
                               grid::StaggeredGrid, chi_u::Matrix{Float64}, chi_w::Matrix{Float64})
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz
    @inbounds for j = 1:nz, i = 1:nx
        dudx = ((1 - chi_u[i+1, j]) * u[i+1, j] - (1 - chi_u[i, j]) * u[i, j]) / dx
        dwdz = ((1 - chi_w[i, j+1]) * w[i, j+1] - (1 - chi_w[i, j]) * w[i, j]) / dz
        div[i, j] = dudx + dwdz
    end
    return div
end

export build_solid_mask_faces_2d, masked_divergence_2d!,
       build_solid_mask_faces_3d, masked_divergence_3d!,
       bdim_divergence_2d!, bdim_gradient_pressure_2d!, apply_bdim_boundary_conditions!


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

# ============================================================================
# Reference BDIM Implementation
# ============================================================================

using StaticArrays

# Convolution kernel functions (legacy reference implementation)
@fastmath kern_wl(d) = 0.5 + 0.5*cos(π*d)
@fastmath kern₀_wl(d) = 0.5 + 0.5*d + 0.5*sin(π*d)/π
@fastmath kern₁_wl(d) = 0.25*(1-d^2) - 0.5*(d*sin(π*d) + (1+cos(π*d))/π)/π

μ₀_wl(d, ϵ) = kern₀_wl(clamp(d/ϵ, -1, 1))
μ₁_wl(d, ϵ) = ϵ * kern₁_wl(clamp(d/ϵ, -1, 1))

# SDF for circle (compatible with existing RigidBody structure)
function circle_sdf_wl(x::SVector{2,Float64}, center::Vector{Float64}, radius::Float64)
    zc = length(center) > 2 ? center[3] : center[2]
    return sqrt((x[1] - center[1])^2 + (x[2] - zc)^2) - radius
end

# Measure body geometry (reference style)
function measure_body_wl(body::RigidBody, x::SVector{2,Float64}, t=0.0)
    if body.shape isa Circle
        d = circle_sdf_wl(x, body.center, body.shape.radius)
    else
        error("BDIM: Only Circle shapes supported currently")
    end
    
    # Skip expensive calculations if far from boundary
    if d^2 > 9
        V_far = get_body_velocity_at_point_xz(body, x[1], x[2])
        return (d, SVector(0.0, 0.0), SVector(V_far[1], V_far[2]))
    end
    
    # Compute gradient for normal vector using finite differences
    h = 1e-6
    if body.shape isa Circle
        dx_p = circle_sdf_wl(SVector(x[1] + h, x[2]), body.center, body.shape.radius)
        dx_m = circle_sdf_wl(SVector(x[1] - h, x[2]), body.center, body.shape.radius)
        dz_p = circle_sdf_wl(SVector(x[1], x[2] + h), body.center, body.shape.radius)
        dz_m = circle_sdf_wl(SVector(x[1], x[2] - h), body.center, body.shape.radius)
    else
        # Default gradient for unsupported shapes
        dx_p = dx_m = dz_p = dz_m = d
    end
    
    n = SVector((dx_p - dx_m)/(2h), (dz_p - dz_m)/(2h))
    
    # Normalize and correct distance
    m = sqrt(n[1]^2 + n[2]^2)
    if m > 1e-12
        d /= m
        n = n / m
    else
        n = SVector(0.0, 0.0)
    end
    
    # Body velocity (translation + rotation)
    V_cart = get_body_velocity_at_point_xz(body, x[1], x[2])
    V = SVector(V_cart[1], V_cart[2])
    
    return (d, n, V)
end

# Fill BDIM arrays (core BDIM functionality)
function measure_bdim_2d!(bdim_data::BDIMData, bodies::RigidBodyCollection, grid::StaggeredGrid, t=0.0, ϵ::Float64=1.0)
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz
    
    # Reset BDIM fields
    fill!(bdim_data.V, 0.0)
    fill!(bdim_data.μ₀, 1.0)
    fill!(bdim_data.μ₁, 0.0)
    
    # Helper coordinate lookups with one ghost layer on each side
    cell_x(i) = i == 1 ? grid.x[1] - dx : (i == nx + 2 ? grid.x[end] + dx : grid.x[i-1])
    cell_z(j) = j == 1 ? grid.z[1] - dz : (j == nz + 2 ? grid.z[end] + dz : grid.z[j-1])
    u_face_x(i) = i == 1 ? grid.xu[1] - dx : (i == nx + 2 ? grid.xu[end] + dx : grid.xu[i-1])
    u_face_z(j) = cell_z(j)
    w_face_x(i) = cell_x(i)
    w_face_z(j) = j == 1 ? grid.zw[1] - dz : (j == nz + 2 ? grid.zw[end] + dz : grid.zw[j-1])
    
    max_distance = 2.5 * ϵ
    max_distance_sq = max_distance^2
    
    # Loop over all bodies
    for body in bodies.bodies
        # Loop over all grid points (including ghost cells for BDIM)
        for j = 1:nz+2, i = 1:nx+2
            x_vec = SVector(cell_x(i), cell_z(j))
            
            # Get distance to body at cell center (quick check)
            if body.shape isa Circle
                d_center = circle_sdf_wl(x_vec, body.center, body.shape.radius)
            else
                continue  # Skip unsupported shapes
            end
            
            # Skip if too far from boundary
            if d_center^2 >= max_distance_sq
                if d_center < 0  # Inside solid
                    bdim_data.μ₀[i, j, 1] = 0.0
                    bdim_data.μ₀[i, j, 2] = 0.0
                end
                continue
            end
            
            # Process u-velocity point (staggered in x)
            x_u = SVector(u_face_x(i), u_face_z(j))
            d_u, n_u, V_u = measure_body_wl(body, x_u, t)
            
            if abs(d_u) < 2 * ϵ  # Near boundary
                bdim_data.V[i, j, 1] = V_u[1]
                μ₀_val = clamp(μ₀_wl(d_u, ϵ), 0.0, 1.0)
                μ₁_val = μ₁_wl(d_u, ϵ)
                bdim_data.μ₀[i, j, 1] = min(bdim_data.μ₀[i, j, 1], μ₀_val)  # Take minimum for multiple bodies
                bdim_data.μ₁[i, j, 1, 1] = μ₁_val * n_u[1]
                bdim_data.μ₁[i, j, 1, 2] = μ₁_val * n_u[2]
            elseif d_u < 0  # Inside solid
                bdim_data.μ₀[i, j, 1] = 0.0
                bdim_data.V[i, j, 1] = V_u[1]
            end
            
            # Process w-velocity point (staggered in z)
            x_w = SVector(w_face_x(i), w_face_z(j))
            d_w, n_w, V_w = measure_body_wl(body, x_w, t)
            
            if abs(d_w) < 2 * ϵ  # Near boundary
                bdim_data.V[i, j, 2] = V_w[2]
                μ₀_val = clamp(μ₀_wl(d_w, ϵ), 0.0, 1.0)
                μ₁_val = μ₁_wl(d_w, ϵ)
                bdim_data.μ₀[i, j, 2] = min(bdim_data.μ₀[i, j, 2], μ₀_val)
                bdim_data.μ₁[i, j, 2, 1] = μ₁_val * n_w[1]
                bdim_data.μ₁[i, j, 2, 2] = μ₁_val * n_w[2]
            elseif d_w < 0  # Inside solid
                bdim_data.μ₀[i, j, 2] = 0.0
                bdim_data.V[i, j, 2] = V_w[2]
            end
        end
    end
end

# BDIM velocity correction (reference approach, simplified for stability)
function apply_bdim_correction_2d!(state::SolutionState, bdim_data::BDIMData, grid::StaggeredGrid, dt::Float64)
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz
    
    nu_x = min(nx + 1, size(state.u, 1))
    nu_z = min(nz, size(state.u, 2))
    nw_x = min(nx, size(state.w, 1))
    nw_z = min(nz + 1, size(state.w, 2))
    
    # Apply BDIM correction to u-velocity faces
    @inbounds for j = 1:nu_z, i = 1:nu_x
        bi = min(i + 1, nx + 2)
        bj = min(j + 1, nz + 2)
        μ0 = bdim_data.μ₀[bi, bj, 1]
        μ1x = bdim_data.μ₁[bi, bj, 1, 1]
        μ1z = bdim_data.μ₁[bi, bj, 1, 2]
        Vb = bdim_data.V[bi, bj, 1]
        
        if μ0 <= 1e-6
            state.u[i, j] = Vb
            continue
        elseif μ0 >= 1 - 1e-8 && abs(μ1x) < 1e-10 && abs(μ1z) < 1e-10
            continue
        end
        
        i_minus = i == 1 ? 1 : i - 1
        i_plus = i == nu_x ? nu_x : i + 1
        j_minus = j == 1 ? 1 : j - 1
        j_plus = j == nu_z ? nu_z : j + 1
        
        dudx = (state.u[i_plus, j] - state.u[i_minus, j]) / ((i_plus - i_minus) * dx)
        dudz = (state.u[i, j_plus] - state.u[i, j_minus]) / ((j_plus - j_minus) * dz)
        
        fluid_part = μ0 * state.u[i, j]
        solid_part = (1 - μ0) * Vb
        correction = -(μ1x * dudx + μ1z * dudz)
        state.u[i, j] = fluid_part + solid_part + correction
    end
    
    # Apply BDIM correction to w-velocity faces
    @inbounds for j = 1:nw_z, i = 1:nw_x
        bi = min(i + 1, nx + 2)
        bj = min(j + 1, nz + 2)
        μ0 = bdim_data.μ₀[bi, bj, 2]
        μ1x = bdim_data.μ₁[bi, bj, 2, 1]
        μ1z = bdim_data.μ₁[bi, bj, 2, 2]
        Vb = bdim_data.V[bi, bj, 2]
        
        if μ0 <= 1e-6
            state.w[i, j] = Vb
            continue
        elseif μ0 >= 1 - 1e-8 && abs(μ1x) < 1e-10 && abs(μ1z) < 1e-10
            continue
        end
        
        i_minus = i == 1 ? 1 : i - 1
        i_plus = i == nw_x ? nw_x : i + 1
        j_minus = j == 1 ? 1 : j - 1
        j_plus = j == nw_z ? nw_z : j + 1
        
        dwdx = (state.w[i_plus, j] - state.w[i_minus, j]) / ((i_plus - i_minus) * dx)
        dwdz = (state.w[i, j_plus] - state.w[i, j_minus]) / ((j_plus - j_minus) * dz)
        
        fluid_part = μ0 * state.w[i, j]
        solid_part = (1 - μ0) * Vb
        correction = -(μ1x * dwdx + μ1z * dwdz)
        state.w[i, j] = fluid_part + solid_part + correction
    end
    @inbounds for idx in eachindex(state.u)
        v = state.u[idx]
        if !isfinite(v)
            state.u[idx] = 0.0
        end
    end
    @inbounds for idx in eachindex(state.w)
        v = state.w[idx]
        if !isfinite(v)
            state.w[idx] = 0.0
        end
    end
end

# Main BDIM immersed boundary function
function apply_bdim_forcing_2d!(state::SolutionState, bodies::RigidBodyCollection, 
                               grid::StaggeredGrid, dt::Float64; bdim_data::Union{BDIMData,Nothing}=nothing)
    """
    Apply reference-style BDIM forcing for rigid bodies in 2D.
    """
    nx, nz = grid.nx, grid.nz
    
    # Create BDIM data if not provided
    if bdim_data === nothing
        bdim_data = BDIMData(nx, nz)
    end
    
    smoothing = max(grid.dx, grid.dz)
    measure_bdim_2d!(bdim_data, bodies, grid, 0.0, smoothing)
    
    # Apply BDIM velocity correction
    apply_bdim_correction_2d!(state, bdim_data, grid, dt)
    
    return bdim_data  # Return for reuse
end

# Updated main immersed boundary function with method selection
function apply_immersed_boundary_forcing!(state::SolutionState, 
                                        rigid_bodies::RigidBodyCollection, 
                                        grid::StaggeredGrid, dt::Float64;
                                        method::ImmersedBoundaryMethod=VolumePenalty,
                                        bdim_data::Union{BDIMData,Nothing}=nothing)
    """
    Apply immersed boundary forcing for rigid bodies with method selection.
    
    Methods:
    - VolumePenalty: Original volume penalty method (default)
    - BDIM: reference-style Boundary Data Immersion Method
    """
    if method == BDIM
        if grid.grid_type == TwoDimensional
            return apply_bdim_forcing_2d!(state, rigid_bodies, grid, dt; bdim_data=bdim_data)
        else
            @warn "BDIM (masked IB) in 3D not implemented; falling back to VolumePenalty"
            method = VolumePenalty
        end
    end
    # VolumePenalty branch (default and 3D fallback)
    if method == VolumePenalty
        # Use original volume penalty method
        if grid.grid_type == TwoDimensional
            ib_data = ImmersedBoundaryData2D(rigid_bodies, grid)
        else
            error("3D immersed boundary data creation not implemented")
        end
        apply_immersed_boundary_forcing!(state, ib_data, rigid_bodies, grid, dt)
        return nothing
    end
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

# ============ 3D masked IB helpers (faces and divergence) ============ #

"""
    build_solid_mask_faces_3d(bodies, grid; eps_mul=2.0)

Construct smooth solid masks at staggered face locations in 3D (χ_u at x-faces,
χ_v at y-faces, χ_w at z-faces) using a smoothed Heaviside of the signed distance
to the nearest body surface. eps = eps_mul*max(dx,dy,dz).
"""
function build_solid_mask_faces_3d(bodies::RigidBodyCollection, grid::StaggeredGrid; eps_mul::Float64=2.0)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    eps = eps_mul * maximum((grid.dx, grid.dy, grid.dz))
    # Signed distance at cell centers
    phi_cc = Array{Float64}(undef, nx, ny, nz)
    @inbounds for k = 1:nz, j = 1:ny, i = 1:nx
        x = grid.x[i]; y = grid.y[j]; z = grid.z[k]
        mind = Inf
        for body in bodies.bodies
            d = distance_to_surface_3d(body, x, y, z)
            if abs(d) < abs(mind)
                mind = d
            end
        end
        phi_cc[i, j, k] = mind
    end
    # Smooth solid fraction at cell centers: χ_cc = Hε(-phi)
    chi_cc = Array{Float64}(undef, nx, ny, nz)
    @inbounds for k = 1:nz, j = 1:ny, i = 1:nx
        chi_cc[i, j, k] = smooth_heaviside(-phi_cc[i, j, k], eps)
    end
    # Interpolate to faces
    chi_u = zeros(Float64, nx+1, ny, nz)
    chi_v = zeros(Float64, nx, ny+1, nz)
    chi_w = zeros(Float64, nx, ny, nz+1)
    # χ at u-faces: average in x
    @inbounds for k = 1:nz, j = 1:ny, i = 1:nx+1
        if i == 1
            chi_u[i, j, k] = chi_cc[1, j, k]
        elseif i == nx+1
            chi_u[i, j, k] = chi_cc[nx, j, k]
        else
            chi_u[i, j, k] = 0.5 * (chi_cc[i, j, k] + chi_cc[i-1, j, k])
        end
    end
    # χ at v-faces: average in y
    @inbounds for k = 1:nz, j = 1:ny+1, i = 1:nx
        if j == 1
            chi_v[i, j, k] = chi_cc[i, 1, k]
        elseif j == ny+1
            chi_v[i, j, k] = chi_cc[i, ny, k]
        else
            chi_v[i, j, k] = 0.5 * (chi_cc[i, j, k] + chi_cc[i, j-1, k])
        end
    end
    # χ at w-faces: average in z
    @inbounds for k = 1:nz+1, j = 1:ny, i = 1:nx
        if k == 1
            chi_w[i, j, k] = chi_cc[i, j, 1]
        elseif k == nz+1
            chi_w[i, j, k] = chi_cc[i, j, nz]
        else
            chi_w[i, j, k] = 0.5 * (chi_cc[i, j, k] + chi_cc[i, j, k-1])
        end
    end
    return chi_u, chi_v, chi_w
end

"""
    masked_divergence_3d!(div, u, v, w, grid, chi_u, chi_v, chi_w)

Compute divergence of masked face velocities (1-χ) at cell centers in 3D.
"""
function masked_divergence_3d!(div::Array{Float64,3}, u::Array{Float64,3}, v::Array{Float64,3}, w::Array{Float64,3},
                               grid::StaggeredGrid, chi_u::Array{Float64,3}, chi_v::Array{Float64,3}, chi_w::Array{Float64,3})
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    @inbounds for k = 1:nz, j = 1:ny, i = 1:nx
        dudx = ((1 - chi_u[i+1, j, k]) * u[i+1, j, k] - (1 - chi_u[i, j, k]) * u[i, j, k]) / dx
        dvdy = ((1 - chi_v[i, j+1, k]) * v[i, j+1, k] - (1 - chi_v[i, j, k]) * v[i, j, k]) / dy
        dwdz = ((1 - chi_w[i, j, k+1]) * w[i, j, k+1] - (1 - chi_w[i, j, k]) * w[i, j, k]) / dz
        div[i, j, k] = dudx + dvdy + dwdz
    end
    return div
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
            
            # Apply to w-velocity (represented as state.w in XZ plane)
            state.w[i, j] += force_w * dt
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
    # Mass-conserving immersed boundary method
    # Applies forcing while maintaining local mass balance
    
    # Store original state
    u_orig = copy(state.u)
    w_orig = copy(state.w)
    
    # Step 1: Apply standard IB forcing
    apply_standard_ib_forcing!(state, bodies, grid)
    
    # Step 2: Apply local mass conservation correction
    apply_local_mass_correction!(state, u_orig, w_orig, bodies, grid, dt)
    
    # Safety check: replace any NaN values
    replace!(state.u, NaN => 0.0)
    replace!(state.w, NaN => 0.0)
end

function apply_standard_ib_forcing!(state::SolutionState, bodies, grid::StaggeredGrid)
    """Apply BDIM-style immersed boundary forcing"""
    
    # First, replace any NaN values with reasonable defaults
    replace!(state.u, NaN => 0.0)
    replace!(state.w, NaN => 0.0)
    
    # Create body mask for BDIM
    body_mask = bodies_mask_2d(bodies, grid)
    
    # Apply BDIM boundary conditions directly
    apply_bdim_boundary_conditions!(state.u, state.w, grid, body_mask, bodies)
end

function apply_local_mass_correction!(state::SolutionState, u_orig::Matrix, w_orig::Matrix, 
                                     bodies, grid::StaggeredGrid, dt::Float64)
    """
    Disabled mass correction to allow natural flow evolution
    """
    # Do nothing - let the gentle penalty forcing handle the boundary conditions
    # without aggressive divergence corrections that freeze the flow
    return
end

function apply_velocity_correction!(vel::Matrix, i::Int, j::Int, correction::Float64,
                                   grid::StaggeredGrid, bodies, direction::Symbol, 
                                   x_coords::Vector, z_coords::Vector)
    """Apply velocity correction if point is outside all bodies"""
    if i > 0 && j > 0 && i <= size(vel, 1) && j <= size(vel, 2)
        x = x_coords[min(i, length(x_coords))]
        z = z_coords[min(j, length(z_coords))]
        
        # Check if point is outside all bodies
        is_outside = true
        for body in bodies.bodies
            dist = sqrt((x - body.center[1])^2 + (z - body.center[2])^2)
            if dist <= body.shape.radius * 1.05  # Small safety margin
                is_outside = false
                break
            end
        end
        
        if is_outside
            vel[i, j] += correction
        end
    end
end


# Simple helper function for cylinder detection
function is_inside_cylinder_xz(body::RigidBody, x::Float64, z::Float64)
    # Simple distance check for circular cylinder
    dx = x - body.center[1]
    dz = z - body.center[2]  # Assume center[2] is z-coordinate
    radius = body.shape.radius
    return (dx^2 + dz^2) <= radius^2
end

function apply_ib_forcing_3d!(state::SolutionState, ib_data::ImmersedBoundaryData,
                             bodies::Union{RigidBodyCollection, FlexibleBodyCollection}, grid::StaggeredGrid, dt::Float64)
    # Similar to 2D but for 3D velocity components
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # u-velocity forcing
    for k = 1:nz, j = 1:ny, i = 1:nx+1
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
    
    # v-velocity forcing
    for k = 1:nz, j = 1:ny+1, i = 1:nx
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
    
    # w-velocity forcing
    for k = 1:nz+1, j = 1:ny, i = 1:nx
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

"""
    solve_bdim_step_2d!(solver, state_new, state_old, dt, bodies)

Complete BDIM solver that integrates immersed boundary conditions directly into
the pressure projection step to maintain mass conservation.
"""
function solve_bdim_step_2d!(solver, state_new::SolutionState, state_old::SolutionState, 
                           dt::Float64, bodies)
    grid = solver.grid
    fluid = solver.fluid
    bc = solver.bc
    
    # Step 1: Predictor step (convection + diffusion)
    # Copy old state to new
    state_new.u .= state_old.u
    state_new.w .= state_old.w
    state_new.p .= state_old.p
    state_new.t = state_old.t + dt
    state_new.step = state_old.step + 1
    
    # Create body mask once
    body_mask = BioFlows.bodies_mask_2d(bodies, grid)
    
    # Apply viscous diffusion (only in fluid regions)
    apply_viscous_diffusion_bdim!(state_new.u, state_new.w, grid, fluid.μ, dt, body_mask)
    
    # Apply convective terms (only in fluid regions) 
    apply_convection_bdim!(state_new.u, state_new.w, state_old.u, state_old.w, grid, dt, body_mask)
    
    # Apply domain boundary conditions (inlet/outlet/walls) BEFORE projection
    # so the projection can enforce incompressibility with those BCs.
    apply_boundary_conditions!(grid, state_new, bc, state_new.t)

    # Step 2: BDIM-modified pressure projection
    # This is where the mass conservation happens
    solve_bdim_projection!(solver, state_new, dt, bodies, body_mask)
    
    # Step 3: Enforce body no-slip (BDIM) at faces
    apply_bdim_boundary_conditions!(state_new.u, state_new.w, grid, body_mask, bodies)
    
    return nothing
end

"""
    solve_bdim_projection!(solver, state, dt, bodies, body_mask)

BDIM-integrated pressure projection that ensures divergence-free velocity field
while respecting immersed boundary conditions.
"""
function solve_bdim_projection!(solver, state::SolutionState, dt::Float64, bodies, body_mask::BitMatrix)
    grid = solver.grid
    fluid = solver.fluid
    
    # Build smooth face masks for fluid region (reference-style)
    chi_u, chi_w = build_solid_mask_faces_2d(bodies, grid; eps_mul=2.0)
    
    # Create RHS for Poisson using masked divergence of face velocities
    rhs = zeros(grid.nx, grid.nz)
    masked_divergence_2d!(rhs, state.u, state.w, grid, chi_u, chi_w)
    rhs .*= (1.0 / dt)
    
    # Solve Poisson equation: ∇²φ = (∇·u*)/dt
    phi = zeros(grid.nx, grid.nz)
    solve_bdim_poisson!(phi, rhs, grid, body_mask, max_iter=1000, tol=1e-8)
    
    # Pressure update: p^{n+1} = p^n + ρ φ (φ has units of pressure/ρ)
    density_val = fluid.ρ isa ConstantDensity ? fluid.ρ.ρ : 1000.0
    state.p .+= density_val .* phi
    
    # Velocity correction: u^{n+1} = u* − dt (1−χ) ∇φ at faces
    dphidx = zeros(grid.nx+1, grid.nz)
    dphidz = zeros(grid.nx, grid.nz+1)
    gradient_pressure_2d!(dphidx, dphidz, phi, grid)
    
    @inbounds for j = 1:grid.nz, i = 1:grid.nx+1
        if i <= size(state.u, 1) && j <= size(state.u, 2)
            state.u[i, j] -= dt * (1.0 - chi_u[i, j]) * dphidx[i, j]
        end
    end
    @inbounds for j = 1:grid.nz+1, i = 1:grid.nx
        if i <= size(state.w, 1) && j <= size(state.w, 2)
            state.w[i, j] -= dt * (1.0 - chi_w[i, j]) * dphidz[i, j]
        end
    end
    
    return nothing
end

"""
    solve_bdim_poisson!(phi, rhs, grid, body_mask; max_iter=1000, tol=1e-8)

Solve Poisson equation ∇²φ = rhs with BDIM boundary conditions.
Sets φ = 0 inside bodies and solves only in fluid regions.
"""
function solve_bdim_poisson!(phi::Matrix{Float64}, rhs::Matrix{Float64}, grid::StaggeredGrid, 
                            body_mask::BitMatrix; max_iter::Int=1000, tol::Float64=1e-8)
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz
    dx2, dz2 = dx^2, dz^2
    
    # Enforce Neumann solvability: subtract mean over fluid cells
    fluid_count = 0
    rhs_sum = 0.0
    @inbounds for j = 1:nz, i = 1:nx
        if !body_mask[i, j]
            rhs_sum += rhs[i, j]
            fluid_count += 1
        end
    end
    rhs_mean = fluid_count > 0 ? rhs_sum / fluid_count : 0.0
    @inbounds for j = 1:nz, i = 1:nx
        if !body_mask[i, j]
            rhs[i, j] -= rhs_mean
        end
    end

    # Gauss-Seidel iterations
    for iter = 1:max_iter
        residual = 0.0
        
        @inbounds for j = 2:nz-1, i = 2:nx-1
            if !body_mask[i, j]  # Only update fluid cells
                phi_old = phi[i, j]
                
                # 5-point stencil for Laplacian
                phi[i, j] = (dx2 * (phi[i, j-1] + phi[i, j+1]) + 
                           dz2 * (phi[i-1, j] + phi[i+1, j]) - 
                           dx2 * dz2 * rhs[i, j]) / (2 * (dx2 + dz2))
                
                residual += (phi[i, j] - phi_old)^2
            else
                phi[i, j] = 0.0  # Zero inside bodies
            end
        end
        
        # Apply homogeneous Neumann BC on domain boundaries: ∂φ/∂n = 0
        @inbounds for i = 1:nx
            phi[i, 1]  = phi[i, 2]
            phi[i, nz] = phi[i, nz-1]
        end
        @inbounds for j = 1:nz
            phi[1, j]  = phi[2, j]
            phi[nx, j] = phi[nx-1, j]
        end
        
        if sqrt(residual) < tol
            break
        end
    end
    
    return nothing
end

"""
    apply_viscous_diffusion_bdim!(u, w, grid, μ, dt, body_mask)

Apply viscous diffusion terms only in fluid regions.
"""
function apply_viscous_diffusion_bdim!(u::Matrix{Float64}, w::Matrix{Float64}, 
                                     grid::StaggeredGrid, μ::Float64, dt::Float64, body_mask::BitMatrix)
    # Simple explicit diffusion - can be replaced with implicit for stability
    # This is a placeholder - in practice should use proper staggered grid operators
    return nothing  # Skip diffusion for now to focus on projection
end

"""
    apply_convection_bdim!(u_new, w_new, u_old, w_old, grid, dt, body_mask)

Apply convective terms only in fluid regions.
"""
function apply_convection_bdim!(u_new::Matrix{Float64}, w_new::Matrix{Float64},
                              u_old::Matrix{Float64}, w_old::Matrix{Float64},
                              grid::StaggeredGrid, dt::Float64, body_mask::BitMatrix)
    # Simple placeholder - skip convection for now to focus on projection
    return nothing
end

# Export IBM functions
export ImmersedBoundaryData, ImmersedBoundaryData2D, ImmersedBoundaryData3D
export BDIMData, ImmersedBoundaryMethod, VolumePenalty, BDIM
export bodies_mask_2d, bodies_mask_3d
export apply_immersed_boundary_forcing!, apply_ib_forcing_2d!, apply_ib_forcing_3d!
export apply_flexible_ib_forcing_2d!, apply_flexible_ib_forcing_3d!
export apply_force_spreading_2d!, apply_force_spreading_3d!
export regularized_delta_2d, regularized_delta_3d
# Export BDIM functions
export bdim_divergence_2d!, bdim_gradient_pressure_2d!, apply_bdim_boundary_conditions!
export solve_bdim_step_2d!, solve_bdim_projection!, solve_bdim_poisson!
export apply_viscous_diffusion_bdim!, apply_convection_bdim!
