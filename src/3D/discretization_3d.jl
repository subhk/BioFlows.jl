function divergence_3d!(div::Array{T,3}, u::Array{T,3}, v::Array{T,3}, w::Array{T,3},
                        grid::StaggeredGrid{T}) where T<:Real
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    
    @inbounds for k = 1:nz, j = 1:ny, i = 1:nx
        div[i, j, k] = (u[i+1, j, k] - u[i, j, k]) / dx + 
                       (v[i, j+1, k] - v[i, j, k]) / dy + 
                       (w[i, j, k+1] - w[i, j, k]) / dz
    end
end

function gradient_pressure_3d!(dpdx::Array{T,3}, dpdy::Array{T,3}, dpdz::Array{T,3}, 
                               p::Array{T,3}, grid::StaggeredGrid{T}) where T<:Real
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    
    # 2nd order accurate pressure gradient at u-velocity points (x-faces)
    @inbounds for k = 1:nz, j = 1:ny, i = 1:nx+1
        if i == 1
            # Left boundary: 2nd order one-sided difference
            dpdx[i, j, k] = (-3*p[1, j, k] + 4*p[2, j, k] - p[3, j, k]) / (2*dx)
        elseif i == nx+1
            # Right boundary: 2nd order one-sided difference
            dpdx[i, j, k] = (3*p[nx, j, k] - 4*p[nx-1, j, k] + p[nx-2, j, k]) / (2*dx)
        else
            # Interior: 2nd order central difference
            dpdx[i, j, k] = (p[i, j, k] - p[i-1, j, k]) / dx
        end
    end
    
    # 2nd order accurate pressure gradient at v-velocity points (y-faces)
    @inbounds for k = 1:nz, j = 1:ny+1, i = 1:nx
        if j == 1
            # Bottom boundary: 2nd order one-sided difference
            dpdy[i, j, k] = (-3*p[i, 1, k] + 4*p[i, 2, k] - p[i, 3, k]) / (2*dy)
        elseif j == ny+1
            # Top boundary: 2nd order one-sided difference
            dpdy[i, j, k] = (3*p[i, ny, k] - 4*p[i, ny-1, k] + p[i, ny-2, k]) / (2*dy)
        else
            # Interior: 2nd order central difference
            dpdy[i, j, k] = (p[i, j, k] - p[i, j-1, k]) / dy
        end
    end
    
    # 2nd order accurate pressure gradient at w-velocity points (z-faces)
    @inbounds for k = 1:nz+1, j = 1:ny, i = 1:nx
        if k == 1
            # Front boundary: 2nd order one-sided difference
            dpdz[i, j, k] = (-3*p[i, j, 1] + 4*p[i, j, 2] - p[i, j, 3]) / (2*dz)
        elseif k == nz+1
            # Back boundary: 2nd order one-sided difference
            dpdz[i, j, k] = (3*p[i, j, nz] - 4*p[i, j, nz-1] + p[i, j, nz-2]) / (2*dz)
        else
            # Interior: 2nd order central difference
            dpdz[i, j, k] = (p[i, j, k] - p[i, j, k-1]) / dz
        end
    end
end

# Minmod slope limiter for 2nd order accuracy with TVD property
function minmod_3d(a::T, b::T) where T<:Real
    if a * b <= 0
        return zero(T)
    elseif abs(a) < abs(b)
        return a
    else
        return b
    end
end

function advection_3d!(adv_u::Array{T,3}, adv_v::Array{T,3}, adv_w::Array{T,3},
                      u::Array{T,3}, v::Array{T,3}, w::Array{T,3},
                      grid::StaggeredGrid{T}) where T<:Real
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    
    # Advection term for u-momentum: u∂u/∂x + v∂u/∂y + w∂u/∂z
    @inbounds for k = 2:nz-1, j = 2:ny-1, i = 2:nx
        # u∂u/∂x
        u_face = 0.5 * (u[i, j, k] + u[i-1, j, k])
        dudx = (u[i, j, k] - u[i-1, j, k]) / dx
        
        # v∂u/∂y
        v_face = 0.25 * (v[i-1, j, k] + v[i, j, k] + v[i-1, j+1, k] + v[i, j+1, k])
        dudy = (u[i, j+1, k] - u[i, j-1, k]) / (2*dy)
        
        # w∂u/∂z
        w_face = 0.25 * (w[i-1, j, k] + w[i, j, k] + w[i-1, j, k+1] + w[i, j, k+1])
        dudz = (u[i, j, k+1] - u[i, j, k-1]) / (2*dz)
        
        adv_u[i, j, k] = u_face * dudx + v_face * dudy + w_face * dudz
    end
    
    # Advection term for v-momentum: u∂v/∂x + v∂v/∂y + w∂v/∂z
    @inbounds for k = 2:nz-1, j = 2:ny, i = 2:nx-1
        # u∂v/∂x
        u_face = 0.25 * (u[i, j-1, k] + u[i+1, j-1, k] + u[i, j, k] + u[i+1, j, k])
        dvdx = (v[i+1, j, k] - v[i-1, j, k]) / (2*dx)
        
        # v∂v/∂y
        v_face = 0.5 * (v[i, j, k] + v[i, j-1, k])
        dvdy = (v[i, j, k] - v[i, j-1, k]) / dy
        
        # w∂v/∂z
        w_face = 0.25 * (w[i, j-1, k] + w[i, j, k] + w[i, j-1, k+1] + w[i, j, k+1])
        dvdz = (v[i, j, k+1] - v[i, j, k-1]) / (2*dz)
        
        adv_v[i, j, k] = u_face * dvdx + v_face * dvdy + w_face * dvdz
    end
    
    # Advection term for w-momentum: u∂w/∂x + v∂w/∂y + w∂w/∂z
    @inbounds for k = 2:nz, j = 2:ny-1, i = 2:nx-1
        # u∂w/∂x
        u_face = 0.25 * (u[i, j, k-1] + u[i+1, j, k-1] + u[i, j, k] + u[i+1, j, k])
        dwdx = (w[i+1, j, k] - w[i-1, j, k]) / (2*dx)
        
        # v∂w/∂y
        v_face = 0.25 * (v[i, j, k-1] + v[i, j+1, k-1] + v[i, j, k] + v[i, j+1, k])
        dwdy = (w[i, j+1, k] - w[i, j-1, k]) / (2*dy)
        
        # w∂w/∂z
        w_face = 0.5 * (w[i, j, k] + w[i, j, k-1])
        dwdz = (w[i, j, k] - w[i, j, k-1]) / dz
        
        adv_w[i, j, k] = u_face * dwdx + v_face * dwdy + w_face * dwdz
    end
end

function compute_diffusion_3d!(diff_u::Array{T,3}, diff_v::Array{T,3}, diff_w::Array{T,3},
                              u::Array{T,3}, v::Array{T,3}, w::Array{T,3},
                              fluid::FluidProperties, grid::StaggeredGrid{T}) where T<:Real
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    μ = fluid.μ
    
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
        ν = μ / ρ
    else
        error("Variable density not implemented for diffusion term")
    end
    
    # Diffusion term for u-momentum: ν∇²u
    @inbounds for k = 2:nz-1, j = 2:ny-1, i = 2:nx
        d2udx2 = (u[i+1, j, k] - 2*u[i, j, k] + u[i-1, j, k]) / dx^2
        d2udy2 = (u[i, j+1, k] - 2*u[i, j, k] + u[i, j-1, k]) / dy^2
        d2udz2 = (u[i, j, k+1] - 2*u[i, j, k] + u[i, j, k-1]) / dz^2
        diff_u[i, j, k] = ν * (d2udx2 + d2udy2 + d2udz2)
    end
    
    # Diffusion term for v-momentum: ν∇²v
    @inbounds for k = 2:nz-1, j = 2:ny, i = 2:nx-1
        d2vdx2 = (v[i+1, j, k] - 2*v[i, j, k] + v[i-1, j, k]) / dx^2
        d2vdy2 = (v[i, j+1, k] - 2*v[i, j, k] + v[i, j-1, k]) / dy^2
        d2vdz2 = (v[i, j, k+1] - 2*v[i, j, k] + v[i, j, k-1]) / dz^2
        diff_v[i, j, k] = ν * (d2vdx2 + d2vdy2 + d2vdz2)
    end
    
    # Diffusion term for w-momentum: ν∇²w
    @inbounds for k = 2:nz, j = 2:ny-1, i = 2:nx-1
        d2wdx2 = (w[i+1, j, k] - 2*w[i, j, k] + w[i-1, j, k]) / dx^2
        d2wdy2 = (w[i, j+1, k] - 2*w[i, j, k] + w[i, j-1, k]) / dy^2
        d2wdz2 = (w[i, j, k+1] - 2*w[i, j, k] + w[i, j, k-1]) / dz^2
        diff_w[i, j, k] = ν * (d2wdx2 + d2wdy2 + d2wdz2)
    end
end

function interpolate_to_cell_center_3d(u::Array{T,3}, v::Array{T,3}, w::Array{T,3},
                                      grid::StaggeredGrid{T}) where T<:Real
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    u_cc = zeros(T, nx, ny, nz)
    v_cc = zeros(T, nx, ny, nz)
    w_cc = zeros(T, nx, ny, nz)
    
    @inbounds for k = 1:nz, j = 1:ny, i = 1:nx
        u_cc[i, j, k] = 0.5 * (u[i, j, k] + u[i+1, j, k])
        v_cc[i, j, k] = 0.5 * (v[i, j, k] + v[i, j+1, k])
        w_cc[i, j, k] = 0.5 * (w[i, j, k] + w[i, j, k+1])
    end
    
    return u_cc, v_cc, w_cc
end

function compute_cfl_3d(u::Array{T,3}, v::Array{T,3}, w::Array{T,3}, 
                       grid::StaggeredGrid{T}, dt::T) where T<:Real
    max_u = maximum(abs.(u))
    max_v = maximum(abs.(v))
    max_w = maximum(abs.(w))
    
    cfl_x = max_u * dt / grid.dx
    cfl_y = max_v * dt / grid.dy
    cfl_z = max_w * dt / grid.dz
    
    return max(cfl_x, cfl_y, cfl_z)
end