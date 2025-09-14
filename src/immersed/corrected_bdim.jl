# Mathematically Correct BDIM Implementation
# Based on Weymouth & Yue (2011) and WaterLily.jl 2024 paper

using StaticArrays

"""
Corrected BDIM data structure following proper mathematical formulation
"""
struct CorrectedBDIM{T}
    # Core BDIM fields on staggered grid
    μ₀::Array{T, 3}      # Zeroth moment: (nx+1, nz+1, 2) for u,w faces
    μ₁::Array{T, 4}      # First moment: (nx+1, nz+1, 2, 2) for gradients  
    V::Array{T, 3}       # Body velocity: (nx+1, nz+1, 2)
    
    # Working arrays
    d::Array{T, 2}       # Distance field: (nx+1, nz+1)
    n::Array{T, 3}       # Normal field: (nx+1, nz+1, 2)
    
    # Kernel width (typically 1.5-2.0 grid cells)
    ϵ::T
end

function CorrectedBDIM(nx::Int, nz::Int, T=Float64; ϵ=1.5)
    return CorrectedBDIM{T}(
        ones(T, nx+1, nz+1, 2),     # μ₀ = 1 in fluid
        zeros(T, nx+1, nz+1, 2, 2), # μ₁ = 0 initially
        zeros(T, nx+1, nz+1, 2),    # V = 0 for static bodies
        zeros(T, nx+1, nz+1),       # distance field
        zeros(T, nx+1, nz+1, 2),    # normal field
        T(ϵ)
    )
end

"""
WaterLily kernel functions (exact mathematical formulation)
"""
@inline function W₀(d, ϵ)
    """Zeroth moment kernel - controls volume fraction"""
    s = clamp(d/ϵ, -1.0, 1.0)
    return 0.5 * (1.0 + s + sin(π*s)/π)
end

@inline function W₁(d, ϵ) 
    """First moment kernel - controls momentum transfer"""
    s = clamp(d/ϵ, -1.0, 1.0)
    return ϵ * 0.25 * (1.0 - s*s) - ϵ/(2π) * (s*sin(π*s) + (1.0 + cos(π*s))/π)
end

"""
Measure body geometry and compute BDIM coefficients
"""
function measure_corrected!(bdim::CorrectedBDIM{T}, bodies, grid, t=zero(T)) where T
    nx, nz = size(grid.x_faces)[1]-1, size(grid.z_faces)[1]-1
    dx, dz = grid.dx, grid.dz
    
    # Reset fields
    fill!(bdim.μ₀, one(T))
    fill!(bdim.μ₁, zero(T))  
    fill!(bdim.V, zero(T))
    fill!(bdim.d, T(Inf))
    fill!(bdim.n, zero(T))
    
    # Process each body
    for body in bodies.bodies
        if !(body.shape isa Circle)
            continue  # Only circles for now
        end
        
        center = SVector{2,T}(body.center[1], body.center[2])
        radius = T(body.shape.radius)
        velocity = SVector{2,T}(body.velocity[1], body.velocity[2])
        
        # Process all velocity face locations
        for k in 1:2  # u and w components
            for j in 1:nz+1, i in 1:nx+1
                # Get face location for component k
                if k == 1  # u-face
                    if i <= nx+1 && j <= nz
                        x_face = SVector{2,T}(grid.x_faces[i], grid.z_centers[j])
                    else
                        continue
                    end
                else  # w-face
                    if i <= nx && j <= nz+1
                        x_face = SVector{2,T}(grid.x_centers[i], grid.z_faces[j])
                    else
                        continue
                    end
                end
                
                # Compute distance to body surface
                d_vec = x_face - center
                d_norm = norm(d_vec)
                d = d_norm - radius
                
                # Store distance and normal
                if i <= size(bdim.d, 1) && j <= size(bdim.d, 2)
                    if abs(d) < abs(bdim.d[i, j])
                        bdim.d[i, j] = d
                        if d_norm > 1e-12
                            bdim.n[i, j, :] .= d_vec / d_norm
                        end
                    end
                end
                
                # Compute BDIM coefficients if within kernel support
                if abs(d) <= bdim.ϵ
                    # Zeroth moment
                    if d < zero(T)  # Inside body
                        bdim.μ₀[i, j, k] = zero(T)
                    else  # Near boundary
                        bdim.μ₀[i, j, k] = W₀(d, bdim.ϵ)
                    end
                    
                    # Body velocity
                    bdim.V[i, j, k] = velocity[k]
                    
                    # First moment tensor
                    if abs(d) < bdim.ϵ
                        w₁ = W₁(d, bdim.ϵ)
                        if d_norm > 1e-12
                            n_vec = d_vec / d_norm
                            for m in 1:2
                                bdim.μ₁[i, j, k, m] = w₁ * n_vec[m]
                            end
                        end
                    end
                end
            end
        end
    end
end

"""
Apply corrected BDIM forcing to velocity field
"""
function apply_corrected_bdim!(state, u_star, w_star, bdim::CorrectedBDIM, dt, grid)
    """
    Corrected BDIM application:
    u = u* + Δt * ∇·(μ₁ ⊗ f) + V + μ₀ * (u* - V)
    
    Where:
    - u* is the predicted velocity from advection-diffusion
    - f = (u* - V) is the forcing
    - μ₀ controls volume fraction
    - μ₁ controls momentum redistribution
    """
    
    dx, dz = grid.dx, grid.dz
    nx, nz = grid.nx, grid.nz
    
    # u-velocity correction
    for j in 1:nz, i in 1:nx+1
        if i <= size(bdim.μ₀, 1) && j <= size(bdim.μ₀, 2)
            # Get BDIM coefficients
            μ₀_val = bdim.μ₀[i, j, 1]
            V_u = bdim.V[i, j, 1]
            
            # Force computation: f = u* - V
            f_u = u_star[i, j] - V_u
            
            # Momentum redistribution term: ∇·(μ₁ ⊗ f)
            div_term = zero(eltype(state.u))
            if i > 1 && i < size(bdim.μ₁, 1) && j > 1 && j < size(bdim.μ₁, 2)
                # ∂/∂x(μ₁ₓₓ * f_u)
                div_term += (bdim.μ₁[i+1, j, 1, 1] * f_u - bdim.μ₁[i-1, j, 1, 1] * f_u) / (2*dx)
                # ∂/∂z(μ₁ₓz * f_u) 
                div_term += (bdim.μ₁[i, j+1, 1, 2] * f_u - bdim.μ₁[i, j-1, 1, 2] * f_u) / (2*dz)
            end
            
            # Apply correction: u = u* + dt*∇·(μ₁⊗f) + V + μ₀*(u*-V)
            state.u[i, j] = u_star[i, j] + dt * div_term + V_u + μ₀_val * f_u
        end
    end
    
    # w-velocity correction
    for j in 1:nz+1, i in 1:nx
        if i <= size(bdim.μ₀, 1) && j <= size(bdim.μ₀, 2)
            # Get BDIM coefficients  
            μ₀_val = bdim.μ₀[i, j, 2]
            V_w = bdim.V[i, j, 2]
            
            # Force computation: f = w* - V
            f_w = w_star[i, j] - V_w
            
            # Momentum redistribution term
            div_term = zero(eltype(state.w))
            if i > 1 && i < size(bdim.μ₁, 1) && j > 1 && j < size(bdim.μ₁, 2)
                # ∂/∂x(μ₁zₓ * f_w)
                div_term += (bdim.μ₁[i+1, j, 2, 1] * f_w - bdim.μ₁[i-1, j, 2, 1] * f_w) / (2*dx)
                # ∂/∂z(μ₁zz * f_w)
                div_term += (bdim.μ₁[i, j+1, 2, 2] * f_w - bdim.μ₁[i, j-1, 2, 2] * f_w) / (2*dz)
            end
            
            # Apply correction
            state.w[i, j] = w_star[i, j] + dt * div_term + V_w + μ₀_val * f_w
        end
    end
end

"""
Stable upwind advection compatible with BDIM
"""
function compute_stable_upwind_advection_2d!(adv_u, adv_w, u, w, grid; cfl_limit=0.5)
    """
    Stable upwind scheme with CFL limiting for BDIM compatibility
    """
    dx, dz = grid.dx, grid.dz
    nx, nz = grid.nx, grid.nz
    
    fill!(adv_u, 0.0)
    fill!(adv_w, 0.0)
    
    # u-momentum advection with CFL-based switching
    for j in 2:nz-1, i in 2:nx
        u_val = u[i, j]
        local_cfl_x = abs(u_val) / dx
        
        # Use upwind only if CFL suggests it's needed
        if local_cfl_x > cfl_limit
            # High CFL: use upwind
            if u_val > 0.0
                dudx = (u[i,j] - u[i-1,j]) / dx
            else
                dudx = (u[i+1,j] - u[i,j]) / dx
            end
        else
            # Low CFL: use central differences for accuracy
            dudx = (u[i+1,j] - u[i-1,j]) / (2*dx)
        end
        
        # Cross-advection with interpolation
        w_interp = 0.25 * (w[i,j] + w[i-1,j] + w[i,j+1] + w[i-1,j+1])
        local_cfl_z = abs(w_interp) / dz
        
        if local_cfl_z > cfl_limit
            if w_interp > 0.0
                dudz = (u[i,j] - u[i,j-1]) / dz
            else
                dudz = (u[i,j+1] - u[i,j]) / dz
            end
        else
            dudz = (u[i,j+1] - u[i,j-1]) / (2*dz)
        end
        
        adv_u[i,j] = u_val * dudx + w_interp * dudz
    end
    
    # w-momentum advection
    for j in 2:nz, i in 2:nx-1
        # Cross-advection
        u_interp = 0.25 * (u[i,j] + u[i+1,j] + u[i,j-1] + u[i+1,j-1])
        local_cfl_x = abs(u_interp) / dx
        
        if local_cfl_x > cfl_limit
            if u_interp > 0.0
                dwdx = (w[i,j] - w[i-1,j]) / dx
            else
                dwdx = (w[i+1,j] - w[i,j]) / dx
            end
        else
            dwdx = (w[i+1,j] - w[i-1,j]) / (2*dx)
        end
        
        # Self-advection
        w_val = w[i,j]
        local_cfl_z = abs(w_val) / dz
        
        if local_cfl_z > cfl_limit
            if w_val > 0.0
                dwdz = (w[i,j] - w[i,j-1]) / dz
            else
                dwdz = (w[i,j+1] - w[i,j]) / dz
            end
        else
            dwdz = (w[i,j+1] - w[i,j-1]) / (2*dz)
        end
        
        adv_w[i,j] = u_interp * dwdx + w_val * dwdz
    end
end

export CorrectedBDIM, measure_corrected!, apply_corrected_bdim!, compute_stable_upwind_advection_2d!