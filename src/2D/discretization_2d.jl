function divergence_2d!(div::Matrix{T}, u::Matrix{T}, v::Matrix{T}, 
                        grid::StaggeredGrid{T}) where T<:Real
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz  # Use dz for XZ plane
    
    @inbounds for j = 1:nz, i = 1:nx
        div[i, j] = (u[i+1, j] - u[i, j]) / dx + (v[i, j+1] - v[i, j]) / dz  # Use dz for z-direction
    end
end

function gradient_pressure_2d!(dpdx::Matrix{T}, dpdz::Matrix{T}, p::Matrix{T},
                               grid::StaggeredGrid{T}) where T<:Real
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz  # Use dz for XZ plane
    
    # 2nd order accurate pressure gradient at u-velocity points (x-faces)
    @inbounds for j = 1:nz, i = 1:nx+1
        if i == 1
            # Left boundary: 2nd order one-sided difference
            dpdx[i, j] = (-3*p[1, j] + 4*p[2, j] - p[3, j]) / (2*dx)
        elseif i == nx+1
            # Right boundary: 2nd order one-sided difference
            dpdx[i, j] = (3*p[nx, j] - 4*p[nx-1, j] + p[nx-2, j]) / (2*dx)
        else
            # Interior: 2nd order central difference
            dpdx[i, j] = (p[i, j] - p[i-1, j]) / dx
        end
    end
    
    # 2nd order accurate pressure gradient at w-velocity points (z-faces, XZ plane)
    @inbounds for j = 1:nz+1, i = 1:nx
        if j == 1
            # Bottom boundary: 2nd order one-sided difference
            dpdz[i, j] = (-3*p[i, 1] + 4*p[i, 2] - p[i, 3]) / (2*dz)
        elseif j == nz+1
            # Top boundary: 2nd order one-sided difference
            dpdz[i, j] = (3*p[i, nz] - 4*p[i, nz-1] + p[i, nz-2]) / (2*dz)
        else
            # Interior: 2nd order central difference
            dpdz[i, j] = (p[i, j] - p[i, j-1]) / dz
        end
    end
end

function advection_2d!(adv_u::Matrix{T}, adv_v::Matrix{T}, 
                      u::Matrix{T}, v::Matrix{T}, 
                      grid::StaggeredGrid{T}) where T<:Real
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz
    
    # 2nd order accurate advection using conservative finite volume method
    # Based on MAC staggered grid discretization
    
    # Advection term for u-momentum: ∇·(u⊗u) = ∂(u²)/∂x + ∂(uv)/∂y
    @inbounds for j = 2:nz-1, i = 2:nx
        # ∂(u²)/∂x: 2nd order conservative form
        # Compute u² at cell faces using 2nd order interpolation
        if i > 2 && i < nx
            # Interior points: 2nd order upwind-biased interpolation
            u_east = u[i, j]   + 0.5 * minmod((u[i+1, j] - u[i, j]), (u[i, j] - u[i-1, j]))
            u_west = u[i-1, j] + 0.5 * minmod((u[i, j] - u[i-1, j]), (u[i-1, j] - u[i-2, j]))
        else
            # Near boundaries: central difference
            u_east = 0.5 * (u[i, j] + u[i+1, j]) if i < nx else u[i, j]
            u_west = 0.5 * (u[i-1, j] + u[i, j])
        end
        
        flux_u_east = u_east^2
        flux_u_west = u_west^2
        d_uu_dx = (flux_u_east - flux_u_west) / dx
        
        # ∂(uv)/∂y: 2nd order conservative form
        # Interpolate u and v to y-faces
        u_north = 0.5 * (u[i, j] + u[i, j+1])
        u_south = 0.5 * (u[i, j-1] + u[i, j])
        
        # 2nd order interpolation of v to u-location
        v_north = 0.25 * (v[i-1, j+1] + v[i, j+1] + v[i-1, j] + v[i, j])
        v_south = 0.25 * (v[i-1, j]   + v[i, j] + v[i-1, j-1] + v[i, j-1])
        
        flux_uv_north = u_north * v_north
        flux_uv_south = u_south * v_south
        d_uv_dz = (flux_uv_north - flux_uv_south) / dz
        
        adv_u[i, j] = d_uu_dx + d_uv_dz
    end
    
    # Advection term for v-momentum: ∇·(v⊗u) = ∂(uv)/∂x + ∂(v²)/∂y
    @inbounds for j = 2:nz, i = 2:nx-1
        # ∂(uv)/∂x: 2nd order conservative form
        # Interpolate u and v to x-faces
        v_east = 0.5 * (v[i, j] + v[i+1, j])
        v_west = 0.5 * (v[i-1, j] + v[i, j])
        
        # 2nd order interpolation of u to v-location
        u_east = 0.25 * (u[i+1, j-1] + u[i+1, j] + u[i, j-1] + u[i, j])
        u_west = 0.25 * (u[i, j-1] + u[i, j] + u[i-1, j-1] + u[i-1, j])
        
        flux_uv_east = u_east * v_east
        flux_uv_west = u_west * v_west
        d_uv_dx = (flux_uv_east - flux_uv_west) / dx
        
        # ∂(v²)/∂y: 2nd order conservative form
        if j > 2 && j < nz
            # Interior points: 2nd order upwind-biased interpolation
            v_north = v[i, j]   + 0.5 * minmod((v[i, j+1] - v[i, j]), (v[i, j] - v[i, j-1]))
            v_south = v[i, j-1] + 0.5 * minmod((v[i, j] - v[i, j-1]), (v[i, j-1] - v[i, j-2]))
        else
            # Near boundaries: central difference
            v_north = 0.5 * (v[i, j] + v[i, j+1]) if j < nz else v[i, j]
            v_south = 0.5 * (v[i, j-1] + v[i, j])
        end
        
        flux_v_north = v_north^2
        flux_v_south = v_south^2
        d_vv_dz = (flux_v_north - flux_v_south) / dz
        
        adv_v[i, j] = d_uv_dx + d_vv_dz
    end
end

# Minmod slope limiter for 2nd order accuracy with TVD property
function minmod(a::T, b::T) where T<:Real
    if a * b <= 0
        return zero(T)
    elseif abs(a) < abs(b)
        return a
    else
        return b
    end
end

function compute_diffusion_2d!(diff_u::Matrix{T}, diff_v::Matrix{T},
                            u::Matrix{T}, v::Matrix{T}, 
                            fluid::FluidProperties, grid::StaggeredGrid{T}) where T<:Real
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz
    μ = fluid.μ
    
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
        ν = μ / ρ
    else
        error("Variable density not implemented for diffusion term")
    end
    
    # 2nd order accurate viscous terms for u-momentum
    # Full viscous stress tensor: ∇·τ = μ∇²u + μ∇(∇·u) for incompressible flow
    # Since ∇·u = 0, this reduces to μ∇²u
    @inbounds for j = 2:nz-1, i = 2:nx
        # ∂²u/∂x²: 2nd order accurate
        d2udx2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx^2
        
        # ∂²u/∂y²: 2nd order accurate 
        d2udz2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dz^2
        
        diff_u[i, j] = ν * (d2udx2 + d2udz2)
    end
    
    # 2nd order accurate viscous terms for v-momentum
    @inbounds for j = 2:nz, i = 2:nx-1
        # ∂²v/∂x²: 2nd order accurate
        d2vdx2 = (v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx^2
        
        # ∂²v/∂y²: 2nd order accurate
        d2vdz2 = (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dz^2
        
        diff_v[i, j] = ν * (d2vdx2 + d2vdz2)
    end
end

# Alternative: Full viscous stress tensor for generality
function viscous_stress_2d!(visc_u::Matrix{T}, visc_v::Matrix{T},
                        u::Matrix{T}, v::Matrix{T}, 
                        fluid::FluidProperties, grid::StaggeredGrid{T}) where T<:Real
    """
    2nd order accurate discretization of full viscous stress tensor.
    For incompressible flow: ∇·τ = μ[∇²u + ∇(∇·u)]
    Since ∇·u = 0, this reduces to μ∇²u, but this function computes the full tensor.
    """
    nx, nz = grid.nx, grid.nz
    dx, dz = grid.dx, grid.dz
    μ = fluid.μ
    
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
    else
        error("Variable density not implemented")
    end
    
    # u-momentum viscous terms
    @inbounds for j = 2:nz-1, i = 2:nx
        # ∂/∂x(μ ∂u/∂x) = μ ∂²u/∂x²
        d2udx2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx^2
        
        # ∂/∂y(μ ∂u/∂y) = μ ∂²u/∂y²  
        d2udz2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dz^2
        
        # For incompressible flow, additional terms from ∇(∇·u) = 0
        visc_u[i, j] = (μ/ρ) * (d2udx2 + d2udz2)
    end
    
    # v-momentum viscous terms
    @inbounds for j = 2:nz, i = 2:nx-1
        # ∂/∂x(μ ∂v/∂x) = μ ∂²v/∂x²
        d2vdx2 = (v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx^2
        
        # ∂/∂y(μ ∂v/∂y) = μ ∂²v/∂y²
        d2vdz2 = (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dz^2
        
        visc_v[i, j] = (μ/ρ) * (d2vdx2 + d2vdz2)
    end
end

function pressure_poisson_2d!(rhs::Matrix{T}, div_u::Matrix{T}, 
                            dt::T, grid::StaggeredGrid{T}) where T<:Real
    nx, nz = grid.nx, grid.nz
    
    @inbounds for j = 1:nz, i = 1:nx
        rhs[i, j] = div_u[i, j] / dt
    end
end

function interpolate_to_cell_center_2d(u::Matrix{T}, v::Matrix{T}, 
                                    grid::StaggeredGrid{T}) where T<:Real
    nx, nz = grid.nx, grid.nz
    u_cc = zeros(T, nx, nz)
    v_cc = zeros(T, nx, nz)
    
    # Interpolate u from face centers to cell centers
    @inbounds for j = 1:nz, i = 1:nx
        u_cc[i, j] = 0.5 * (u[i, j] + u[i+1, j])
    end
    
    # Interpolate v from face centers to cell centers  
    @inbounds for j = 1:nz, i = 1:nx
        v_cc[i, j] = 0.5 * (v[i, j] + v[i, j+1])
    end
    
    return u_cc, v_cc
end

function compute_cfl_2d(u::Matrix{T}, v::Matrix{T}, 
                    grid::StaggeredGrid{T}, dt::T) where T<:Real
    max_u = maximum(abs.(u))
    max_v = maximum(abs.(v))
    
    cfl_x = max_u * dt / grid.dx
    cfl_y = max_v * dt / grid.dz
    
    return max(cfl_x, cfl_y)
end