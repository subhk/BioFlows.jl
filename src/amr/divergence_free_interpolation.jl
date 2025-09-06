"""
Divergence-Free Velocity Interpolation for AMR

This module implements conservative velocity interpolation between AMR levels
that preserves the divergence-free condition ∇·u = 0.
"""

using LinearAlgebra

"""
    interpolate_velocity_conservative!(u_fine, v_fine, u_coarse, v_coarse, amr_level)

Conservative velocity interpolation that preserves divergence-free condition.

For staggered MAC grids, this function:
1. Interpolates face-centered velocities using area-weighted averaging
2. Applies divergence correction to ensure ∇·u = 0
3. Handles 2D (XZ plane) and 3D configurations

# Arguments
- `u_fine::Array`: Fine grid u-velocity (x-faces) - modified in place
- `v_fine::Array`: Fine grid v-velocity (z-faces for 2D, y-faces for 3D) - modified in place  
- `u_coarse::Array`: Coarse grid u-velocity (x-faces)
- `v_coarse::Array`: Coarse grid v-velocity
- `amr_level::AMRLevel`: AMR level information containing grid spacing
"""
function interpolate_velocity_conservative!(u_fine::Matrix{T}, v_fine::Matrix{T}, 
                                           u_coarse::Matrix{T}, v_coarse::Matrix{T}, 
                                           amr_level::AMRLevel) where T<:Real
    
    if amr_level.grid_type == TwoDimensional
        interpolate_velocity_conservative_2d!(u_fine, v_fine, u_coarse, v_coarse, amr_level)
    else
        error("3D velocity interpolation not yet implemented")
    end
end

"""
    interpolate_velocity_conservative_2d!(u_fine, v_fine, u_coarse, v_coarse, amr_level)

2D divergence-free velocity interpolation for XZ plane configuration.
"""
function interpolate_velocity_conservative_2d!(u_fine::Matrix{T}, v_fine::Matrix{T}, 
                                              u_coarse::Matrix{T}, v_coarse::Matrix{T}, 
                                              amr_level::AMRLevel) where T<:Real
    
    # Grid dimensions
    nx_coarse, nz_coarse = size(u_coarse, 1) - 1, size(u_coarse, 2)  # u has nx+1 points
    nx_fine = 2 * nx_coarse
    nz_fine = 2 * nz_coarse
    
    # Refinement ratio (typically 2)
    ratio = 2
    
    # Step 1: Area-weighted interpolation for u-velocity (x-faces)
    # u is located at x-faces, so preserve flux across faces
    interpolate_u_faces_2d!(u_fine, u_coarse, ratio)
    
    # Step 2: Area-weighted interpolation for v-velocity (z-faces) 
    # v represents w-velocity in XZ plane, located at z-faces
    interpolate_v_faces_2d!(v_fine, v_coarse, ratio)
    
    # Step 3: Apply divergence correction to ensure ∇·u = 0
    # Use projection method to enforce discrete divergence-free condition
    apply_divergence_correction_2d!(u_fine, v_fine, amr_level)
end

"""
    interpolate_u_faces_2d!(u_fine, u_coarse, ratio)

Conservative interpolation of u-velocity at x-faces.
Preserves mass flux across faces.
"""
function interpolate_u_faces_2d!(u_fine::Matrix{T}, u_coarse::Matrix{T}, ratio::Int) where T<:Real
    
    nx_coarse_plus1, nz_coarse = size(u_coarse)
    nx_fine_plus1 = 2 * (nx_coarse_plus1 - 1) + 1
    nz_fine = 2 * nz_coarse
    
    # For each coarse u-face, create refined faces
    for j_c = 1:nz_coarse, i_c = 1:nx_coarse_plus1
        
        # Map to fine grid indices
        # Each coarse z-cell maps to 2 fine z-cells
        j_f_start = 2 * (j_c - 1) + 1
        j_f_end = j_f_start + 1
        
        # x-face positions remain the same relative to x-coordinates
        i_f = 2 * (i_c - 1) + 1
        
        # Conservative flux interpolation
        # The flux through a coarse face equals sum of fluxes through fine faces
        coarse_flux = u_coarse[i_c, j_c]
        
        # For conservation, flux per fine face should account for area difference
        # Each coarse face maps to 2 fine faces in the z-direction
        # Since fine faces have half the area, they should have the same velocity
        # to conserve mass flux
        for j_f = j_f_start:min(j_f_end, nz_fine)
            if i_f <= nx_fine_plus1
                u_fine[i_f, j_f] = coarse_flux  # Preserve velocity (mass flux conserved by geometry)
            end
        end
        
        # Add intermediate x-faces for interior refinement
        if i_c < nx_coarse_plus1
            i_f_mid = i_f + 1
            if i_f_mid <= nx_fine_plus1
                # Interpolate velocity at intermediate x-face
                u_interp = 0.5 * (u_coarse[i_c, j_c] + u_coarse[min(i_c + 1, nx_coarse_plus1), j_c])
                for j_f = j_f_start:min(j_f_end, nz_fine)
                    u_fine[i_f_mid, j_f] = u_interp
                end
            end
        end
    end
end

"""
    interpolate_v_faces_2d!(v_fine, v_coarse, ratio)

Conservative interpolation of v-velocity at z-faces (w-velocity in XZ plane).
"""
function interpolate_v_faces_2d!(v_fine::Matrix{T}, v_coarse::Matrix{T}, ratio::Int) where T<:Real
    
    nx_coarse, nz_coarse_plus1 = size(v_coarse)
    nx_fine = 2 * nx_coarse  
    nz_fine_plus1 = 2 * (nz_coarse_plus1 - 1) + 1
    
    # For each coarse v-face, create refined faces
    for j_c = 1:nz_coarse_plus1, i_c = 1:nx_coarse
        
        # Map to fine grid indices  
        i_f_start = 2 * (i_c - 1) + 1
        i_f_end = i_f_start + 1
        
        # z-face positions
        j_f = 2 * (j_c - 1) + 1
        
        # Conservative flux interpolation
        coarse_flux = v_coarse[i_c, j_c]
        
        # For conservation: each coarse face maps to 2 fine faces in x-direction
        # Fine faces have half the area, so preserve velocity for mass flux conservation
        for i_f = i_f_start:min(i_f_end, nx_fine)
            if j_f <= nz_fine_plus1
                v_fine[i_f, j_f] = coarse_flux  # Preserve velocity (mass flux conserved by geometry)
            end
        end
        
        # Add intermediate z-faces
        if j_c < nz_coarse_plus1
            j_f_mid = j_f + 1
            if j_f_mid <= nz_fine_plus1
                v_interp = 0.5 * (v_coarse[i_c, j_c] + v_coarse[i_c, min(j_c + 1, nz_coarse_plus1)])
                for i_f = i_f_start:min(i_f_end, nx_fine)
                    v_fine[i_f, j_f_mid] = v_interp
                end
            end
        end
    end
end

"""
    apply_divergence_correction_2d!(u_fine, v_fine, amr_level)

Apply projection to ensure discrete divergence-free condition.
Solves ∇²φ = ∇·u, then corrects u := u - ∇φ.
"""
function apply_divergence_correction_2d!(u_fine::Matrix{T}, v_fine::Matrix{T}, 
                                        amr_level::AMRLevel) where T<:Real
    
    nx, nz = size(u_fine, 1) - 1, size(u_fine, 2)  # Cell-centered dimensions
    dx, dz = amr_level.dx, amr_level.dz
    
    # Compute divergence at cell centers
    div_u = zeros(T, nx, nz)
    compute_divergence_2d!(div_u, u_fine, v_fine, dx, dz)
    
    # Check if correction is needed
    max_div = maximum(abs.(div_u))
    if max_div < 1e-12
        return  # Already divergence-free
    end
    
    # Solve Poisson equation ∇²φ = ∇·u for correction potential
    phi = zeros(T, nx, nz)
    solve_poisson_correction_2d!(phi, div_u, dx, dz)
    
    # Apply correction: u := u - ∇φ
    apply_gradient_correction_2d!(u_fine, v_fine, phi, dx, dz)
end

"""
    compute_divergence_2d!(div_u, u, v, dx, dz)

Compute discrete divergence using finite volume method.
"""
function compute_divergence_2d!(div_u::Matrix{T}, u::Matrix{T}, v::Matrix{T}, 
                                dx::T, dz::T) where T<:Real
    
    nx, nz = size(div_u)
    
    @inbounds for j = 1:nz, i = 1:nx
        # Finite volume divergence: ∇·u = (u_{i+1/2} - u_{i-1/2})/dx + (v_{j+1/2} - v_{j-1/2})/dz
        div_u[i, j] = (u[i+1, j] - u[i, j]) / dx + (v[i, j+1] - v[i, j]) / dz
    end
end

"""
    solve_poisson_correction_2d!(phi, rhs, dx, dz)

Solve ∇²φ = rhs using simple iterative method.
For production code, this should use the multigrid solver.
"""
function solve_poisson_correction_2d!(phi::Matrix{T}, rhs::Matrix{T}, 
                                     dx::T, dz::T, max_iter::Int=100, 
                                     tol::T=1e-10) where T<:Real
    
    nx, nz = size(phi)
    dx2, dz2 = dx^2, dz^2
    factor = 1.0 / (2.0 * (1.0/dx2 + 1.0/dz2))
    
    # Gauss-Seidel iteration
    for iter = 1:max_iter
        residual = 0.0
        
        for j = 2:nz-1, i = 2:nx-1
            old_phi = phi[i, j]
            phi[i, j] = factor * ((phi[i+1,j] + phi[i-1,j]) / dx2 + 
                                 (phi[i,j+1] + phi[i,j-1]) / dz2 - rhs[i,j])
            residual += (phi[i,j] - old_phi)^2
        end
        
        # Apply Neumann boundary conditions
        phi[1, :] .= phi[2, :]      # ∂φ/∂x = 0
        phi[nx, :] .= phi[nx-1, :]  # ∂φ/∂x = 0
        phi[:, 1] .= phi[:, 2]      # ∂φ/∂z = 0
        phi[:, nz] .= phi[:, nz-1]  # ∂φ/∂z = 0
        
        if sqrt(residual) < tol
            break
        end
    end
end

"""
    apply_gradient_correction_2d!(u, v, phi, dx, dz)

Apply velocity correction: u := u - ∇φ
"""
function apply_gradient_correction_2d!(u::Matrix{T}, v::Matrix{T}, phi::Matrix{T}, 
                                      dx::T, dz::T) where T<:Real
    
    nx, nz = size(phi)
    
    # Correct u-velocity at x-faces
    @inbounds for j = 1:nz, i = 1:nx+1
        if i == 1
            # Left boundary
            dphidx = (phi[1, j] - 0.0) / (0.5 * dx)  # One-sided difference
        elseif i == nx+1
            # Right boundary  
            dphidx = (0.0 - phi[nx, j]) / (0.5 * dx)  # One-sided difference
        else
            # Interior: central difference
            dphidx = (phi[i, j] - phi[i-1, j]) / dx
        end
        u[i, j] -= dphidx
    end
    
    # Correct v-velocity at z-faces  
    @inbounds for j = 1:nz+1, i = 1:nx
        if j == 1
            # Bottom boundary
            dphidz = (phi[i, 1] - 0.0) / (0.5 * dz)
        elseif j == nz+1
            # Top boundary
            dphidz = (0.0 - phi[i, nz]) / (0.5 * dz)
        else
            # Interior
            dphidz = (phi[i, j] - phi[i, j-1]) / dz
        end
        v[i, j] -= dphidz
    end
end

"""
    verify_divergence_free(u, v, dx, dz, tolerance=1e-10)

Verify that interpolated velocity field is divergence-free.
"""
function verify_divergence_free(u::Matrix{T}, v::Matrix{T}, dx::T, dz::T, 
                               tolerance::T=1e-10) where T<:Real
    
    nx, nz = size(u, 1) - 1, size(u, 2)
    div_u = zeros(T, nx, nz)
    compute_divergence_2d!(div_u, u, v, dx, dz)
    
    max_div = maximum(abs.(div_u))
    mean_div = sum(abs.(div_u)) / length(div_u)
    
    is_divergence_free = max_div < tolerance
    
    return is_divergence_free, max_div, mean_div
end