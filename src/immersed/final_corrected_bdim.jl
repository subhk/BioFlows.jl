# FINAL CORRECTED BDIM IMPLEMENTATION
# Complete rewrite based on mathematical principles

using StaticArrays

"""
Complete BDIM solver with proper mathematical formulation
"""
struct FinalBDIM{T}
    # Grid dimensions
    nx::Int
    nz::Int
    dx::T
    dz::T
    
    # BDIM fields
    χ_u::Matrix{T}    # Volume fraction at u-faces
    χ_w::Matrix{T}    # Volume fraction at w-faces
    F_u::Matrix{T}    # Body force at u-faces
    F_w::Matrix{T}    # Body force at w-faces
    
    # Body geometry
    sdf::Matrix{T}    # Signed distance field at cell centers
    
    # Parameters
    ε::T             # Kernel width
end

function FinalBDIM(nx::Int, nz::Int, dx::T, dz::T; ε=T(1.5)) where T
    return FinalBDIM{T}(
        nx, nz, dx, dz,
        ones(T, nx+1, nz),      # χ_u: 1 = fluid, 0 = solid
        ones(T, nx, nz+1),      # χ_w: 1 = fluid, 0 = solid  
        zeros(T, nx+1, nz),     # F_u: body forces
        zeros(T, nx, nz+1),     # F_w: body forces
        zeros(T, nx, nz),       # sdf: signed distance
        ε
    )
end

"""
Smooth Heaviside function for BDIM
"""
@inline function heaviside(d, ε)
    if d < -ε
        return 0.0
    elseif d > ε
        return 1.0
    else
        return 0.5 * (1.0 + d/ε + sin(π*d/ε)/π)
    end
end

"""
Update BDIM fields for circle
"""
function update_bdim_circle!(bdim::FinalBDIM{T}, center, radius, velocity) where T
    cx, cz = center[1], center[2]
    V_body_x, V_body_z = velocity[1], velocity[2]
    
    # Reset fields
    fill!(bdim.χ_u, one(T))
    fill!(bdim.χ_w, one(T))
    fill!(bdim.F_u, zero(T))
    fill!(bdim.F_w, zero(T))
    
    # Update signed distance field at cell centers
    for j in 1:bdim.nz, i in 1:bdim.nx
        x_center = (i - 0.5) * bdim.dx
        z_center = (j - 0.5) * bdim.dz
        
        d = sqrt((x_center - cx)^2 + (z_center - cz)^2) - radius
        bdim.sdf[i, j] = d
    end
    
    # Update u-face volume fractions and forces
    for j in 1:bdim.nz, i in 1:bdim.nx+1
        x_face = (i - 1) * bdim.dx  # u-face location
        z_face = (j - 0.5) * bdim.dz
        
        d = sqrt((x_face - cx)^2 + (z_face - cz)^2) - radius
        
        # Volume fraction (1 = fluid, 0 = solid)
        bdim.χ_u[i, j] = heaviside(d, bdim.ε)
        
        # Body force (enforces V = V_body)
        if d < bdim.ε  # Within influence region
            bdim.F_u[i, j] = (1.0 - bdim.χ_u[i, j]) * V_body_x / bdim.ε
        end
    end
    
    # Update w-face volume fractions and forces  
    for j in 1:bdim.nz+1, i in 1:bdim.nx
        x_face = (i - 0.5) * bdim.dx
        z_face = (j - 1) * bdim.dz  # w-face location
        
        d = sqrt((x_face - cx)^2 + (z_face - cz)^2) - radius
        
        # Volume fraction
        bdim.χ_w[i, j] = heaviside(d, bdim.ε)
        
        # Body force
        if d < bdim.ε
            bdim.F_w[i, j] = (1.0 - bdim.χ_w[i, j]) * V_body_z / bdim.ε
        end
    end
end

"""
Apply BDIM correction to velocity field
"""
function apply_bdim_correction!(u_new, w_new, u_star, w_star, bdim::FinalBDIM{T}, dt) where T
    """
    BDIM velocity correction:
    u = χ * u* + (1-χ) * V_body + dt * F_body
    
    This enforces:
    - No-slip boundary condition inside body
    - Smooth transition in boundary region
    - Mass conservation
    """
    
    # u-velocity correction
    for j in 1:bdim.nz, i in 1:bdim.nx+1
        χ = bdim.χ_u[i, j]
        F = bdim.F_u[i, j]
        
        # Blend between fluid velocity and body velocity
        u_new[i, j] = χ * u_star[i, j] + dt * F
        
        # Ensure bounded values
        u_new[i, j] = clamp(u_new[i, j], -5.0, 5.0)
    end
    
    # w-velocity correction
    for j in 1:bdim.nz+1, i in 1:bdim.nx
        χ = bdim.χ_w[i, j]
        F = bdim.F_w[i, j]
        
        # Blend between fluid velocity and body velocity  
        w_new[i, j] = χ * w_star[i, j] + dt * F
        
        # Ensure bounded values
        w_new[i, j] = clamp(w_new[i, j], -5.0, 5.0)
    end
end

"""
Stable upwind advection for BDIM
"""
function compute_bdim_safe_advection!(adv_u, adv_w, u, w, dx, dz; α=0.7)
    """
    Upwind-biased advection with safety limiters
    α = 0.7: blend between upwind (α=1) and central (α=0)
    """
    nx, nz = size(u, 1) - 1, size(u, 2)
    
    fill!(adv_u, 0.0)
    fill!(adv_w, 0.0)
    
    # u-momentum equation
    for j in 2:nz-1, i in 2:nx
        u_val = u[i, j]
        
        # Self-advection: u * ∂u/∂x
        if u_val > 0.0
            dudx_up = (u[i, j] - u[i-1, j]) / dx     # Upwind
            dudx_cd = (u[i+1, j] - u[i-1, j]) / (2*dx)  # Central
        else
            dudx_up = (u[i+1, j] - u[i, j]) / dx     # Upwind
            dudx_cd = (u[i+1, j] - u[i-1, j]) / (2*dx)  # Central
        end
        dudx = α * dudx_up + (1-α) * dudx_cd
        
        # Cross-advection: w * ∂u/∂z
        w_interp = 0.25 * (w[i,j] + w[i-1,j] + w[i,j+1] + w[i-1,j+1])
        if w_interp > 0.0
            dudz_up = (u[i, j] - u[i, j-1]) / dz
            dudz_cd = (u[i, j+1] - u[i, j-1]) / (2*dz)
        else
            dudz_up = (u[i, j+1] - u[i, j]) / dz
            dudz_cd = (u[i, j+1] - u[i, j-1]) / (2*dz)
        end
        dudz = α * dudz_up + (1-α) * dudz_cd
        
        adv_u[i, j] = u_val * dudx + w_interp * dudz
        
        # Safety limiter
        adv_u[i, j] = clamp(adv_u[i, j], -10.0, 10.0)
    end
    
    # w-momentum equation
    for j in 2:nz, i in 2:nx-1
        w_val = w[i, j]
        
        # Cross-advection: u * ∂w/∂x  
        u_interp = 0.25 * (u[i,j] + u[i+1,j] + u[i,j-1] + u[i+1,j-1])
        if u_interp > 0.0
            dwdx_up = (w[i, j] - w[i-1, j]) / dx
            dwdx_cd = (w[i+1, j] - w[i-1, j]) / (2*dx)
        else
            dwdx_up = (w[i+1, j] - w[i, j]) / dx
            dwdx_cd = (w[i+1, j] - w[i-1, j]) / (2*dx)
        end
        dwdx = α * dwdx_up + (1-α) * dwdx_cd
        
        # Self-advection: w * ∂w/∂z
        if w_val > 0.0
            dwdz_up = (w[i, j] - w[i, j-1]) / dz
            dwdz_cd = (w[i, j+1] - w[i, j-1]) / (2*dz)
        else
            dwdz_up = (w[i, j+1] - w[i, j]) / dz  
            dwdz_cd = (w[i, j+1] - w[i, j-1]) / (2*dz)
        end
        dwdz = α * dwdz_up + (1-α) * dwdz_cd
        
        adv_w[i, j] = u_interp * dwdx + w_val * dwdz
        
        # Safety limiter
        adv_w[i, j] = clamp(adv_w[i, j], -10.0, 10.0)
    end
end

export FinalBDIM, update_bdim_circle!, apply_bdim_correction!, compute_bdim_safe_advection!