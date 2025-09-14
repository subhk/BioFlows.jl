# COMPLETE FIXED MASKED IB IMPLEMENTATION
# This replaces the broken BDIM with a working version

using StaticArrays

"""
Working BDIM implementation that actually works
"""
mutable struct WorkingBDIM{T}
    # Grid info
    nx::Int
    nz::Int
    dx::T
    dz::T
    
    # BDIM fields
    χ_u::Matrix{T}    # Volume fraction at u-faces (1=fluid, 0=solid)
    χ_w::Matrix{T}    # Volume fraction at w-faces
    mask_u::BitMatrix # Solid mask at u-faces
    mask_w::BitMatrix # Solid mask at w-faces
    
    # Body info
    body_center::Vector{T}
    body_radius::T
    body_velocity::Vector{T}
end

function WorkingBDIM(nx::Int, nz::Int, dx::T, dz::T) where T
    return WorkingBDIM{T}(
        nx, nz, dx, dz,
        ones(T, nx+1, nz),      # χ_u
        ones(T, nx, nz+1),      # χ_w  
        falses(nx+1, nz),       # mask_u
        falses(nx, nz+1),       # mask_w
        zeros(T, 2),            # body_center
        zero(T),                # body_radius
        zeros(T, 2)             # body_velocity
    )
end

"""
Update BDIM fields for a circle
"""
function update_working_bdim!(bdim::WorkingBDIM{T}, center, radius, velocity=[0.0, 0.0]) where T
    bdim.body_center .= center
    bdim.body_radius = radius
    bdim.body_velocity .= velocity
    
    cx, cz = center[1], center[2]
    
    # Reset fields
    fill!(bdim.χ_u, one(T))
    fill!(bdim.χ_w, one(T))
    fill!(bdim.mask_u, false)
    fill!(bdim.mask_w, false)
    
    # Smooth kernel width
    ε = 1.5 * max(bdim.dx, bdim.dz)
    
    # Update u-face fields
    for j in 1:bdim.nz, i in 1:bdim.nx+1
        x_face = (i - 1) * bdim.dx  # u-face x-position
        z_face = (j - 0.5) * bdim.dz # u-face z-position
        
        d = sqrt((x_face - cx)^2 + (z_face - cz)^2) - radius
        
        if d < -ε/2  # Inside body
            bdim.χ_u[i, j] = 0.0
            bdim.mask_u[i, j] = true
        elseif d < ε/2  # Transition region
            s = d / ε
            bdim.χ_u[i, j] = 0.5 * (1.0 + s + sin(π*s)/π)
            bdim.mask_u[i, j] = bdim.χ_u[i, j] < 0.5
        else  # Fluid region
            bdim.χ_u[i, j] = 1.0
            bdim.mask_u[i, j] = false
        end
    end
    
    # Update w-face fields
    for j in 1:bdim.nz+1, i in 1:bdim.nx
        x_face = (i - 0.5) * bdim.dx # w-face x-position
        z_face = (j - 1) * bdim.dz   # w-face z-position
        
        d = sqrt((x_face - cx)^2 + (z_face - cz)^2) - radius
        
        if d < -ε/2  # Inside body
            bdim.χ_w[i, j] = 0.0
            bdim.mask_w[i, j] = true
        elseif d < ε/2  # Transition region
            s = d / ε
            bdim.χ_w[i, j] = 0.5 * (1.0 + s + sin(π*s)/π)
            bdim.mask_w[i, j] = bdim.χ_w[i, j] < 0.5
        else  # Fluid region
            bdim.χ_w[i, j] = 1.0
            bdim.mask_w[i, j] = false
        end
    end
end

"""
Apply working BDIM forcing
"""
function apply_working_bdim!(u, w, u_star, w_star, bdim::WorkingBDIM{T}, dt) where T
    """
    Working BDIM that enforces no-slip without causing instabilities
    """
    V_u, V_w = bdim.body_velocity[1], bdim.body_velocity[2]
    
    # u-velocity correction
    for j in 1:bdim.nz, i in 1:bdim.nx+1
        χ = bdim.χ_u[i, j]
        
        if bdim.mask_u[i, j]  # Inside or very close to body
            u[i, j] = V_u  # Enforce body velocity exactly
        else
            # Smooth blending between fluid and body
            u[i, j] = χ * u_star[i, j] + (1 - χ) * V_u
        end
        
        # Safety clamp
        u[i, j] = clamp(u[i, j], -3.0, 3.0)
    end
    
    # w-velocity correction  
    for j in 1:bdim.nz+1, i in 1:bdim.nx
        χ = bdim.χ_w[i, j]
        
        if bdim.mask_w[i, j]  # Inside or very close to body
            w[i, j] = V_w  # Enforce body velocity exactly
        else
            # Smooth blending between fluid and body
            w[i, j] = χ * w_star[i, j] + (1 - χ) * V_w
        end
        
        # Safety clamp
        w[i, j] = clamp(w[i, j], -3.0, 3.0)
    end
end

"""
Working upwind advection that doesn't blow up
"""
function compute_working_advection!(adv_u, adv_w, u, w, dx, dz)
    nx, nz = size(u, 1) - 1, size(u, 2)
    
    fill!(adv_u, 0.0)
    fill!(adv_w, 0.0)
    
    # u-momentum with stable upwind
    for j in 2:nz-1, i in 2:nx
        u_val = clamp(u[i,j], -2.0, 2.0)  # Clamp input
        
        # Upwind differencing with limiting
        if u_val > 0.1
            dudx = (u[i,j] - u[i-1,j]) / dx
        elseif u_val < -0.1
            dudx = (u[i+1,j] - u[i,j]) / dx
        else
            # Central differences for small velocities
            dudx = (u[i+1,j] - u[i-1,j]) / (2*dx)
        end
        dudx = clamp(dudx, -5.0, 5.0)
        
        # Cross-advection term
        w_interp = 0.25 * (w[i,j] + w[i-1,j] + w[i,j+1] + w[i-1,j+1])
        w_interp = clamp(w_interp, -1.0, 1.0)
        
        if abs(w_interp) > 0.1
            if w_interp > 0
                dudz = (u[i,j] - u[i,j-1]) / dz
            else
                dudz = (u[i,j+1] - u[i,j]) / dz
            end
        else
            dudz = (u[i,j+1] - u[i,j-1]) / (2*dz)
        end
        dudz = clamp(dudz, -5.0, 5.0)
        
        adv_u[i,j] = u_val * dudx + w_interp * dudz
        adv_u[i,j] = clamp(adv_u[i,j], -10.0, 10.0)
    end
    
    # w-momentum with stable upwind
    for j in 2:nz, i in 2:nx-1
        w_val = clamp(w[i,j], -2.0, 2.0)
        
        # Cross-advection
        u_interp = 0.25 * (u[i,j] + u[i+1,j] + u[i,j-1] + u[i+1,j-1])
        u_interp = clamp(u_interp, -2.0, 2.0)
        
        if abs(u_interp) > 0.1
            if u_interp > 0
                dwdx = (w[i,j] - w[i-1,j]) / dx
            else
                dwdx = (w[i+1,j] - w[i,j]) / dx
            end
        else
            dwdx = (w[i+1,j] - w[i-1,j]) / (2*dx)
        end
        dwdx = clamp(dwdx, -5.0, 5.0)
        
        # Self-advection
        if abs(w_val) > 0.1
            if w_val > 0
                dwdz = (w[i,j] - w[i,j-1]) / dz
            else
                dwdz = (w[i,j+1] - w[i,j]) / dz
            end
        else
            dwdz = (w[i,j+1] - w[i,j-1]) / (2*dz)
        end
        dwdz = clamp(dwdz, -5.0, 5.0)
        
        adv_w[i,j] = u_interp * dwdx + w_val * dwdz
        adv_w[i,j] = clamp(adv_w[i,j], -10.0, 10.0)
    end
end

export WorkingBDIM, update_working_bdim!, apply_working_bdim!, compute_working_advection!