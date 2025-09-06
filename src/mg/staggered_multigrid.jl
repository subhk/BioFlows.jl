"""
Staggered Grid-Aware Multigrid Solver

This module provides a multigrid solver specifically designed for staggered grids
where pressure is at cell centers and velocities are at cell faces.

The pressure Poisson equation ∇²φ = ∇·u* is solved with proper staggered grid handling:
- φ (pressure correction) lives at cell centers
- RHS (∇·u*) is computed from face-centered velocities
- Gradients of φ are computed to face centers for velocity correction
"""

"""
    StaggeredMultiLevelPoisson{T}

Multigrid solver specifically designed for staggered grids.
Properly handles the fact that pressure lives at cell centers while
velocities live at cell faces.
"""
mutable struct StaggeredMultiLevelPoisson{T}
    levels::Int
    # Pressure arrays at cell centers for each level
    φ::Vector{Matrix{T}}      # Pressure correction (cell centers)
    r::Vector{Matrix{T}}      # Residual (cell centers)
    b::Vector{Matrix{T}}      # RHS from divergence (cell centers)
    # Grid information for each level
    nx::Vector{Int}           # Number of cells in x
    ny::Vector{Int}           # Number of cells in y  
    dx::Vector{T}             # Cell spacing in x
    dy::Vector{T}             # Cell spacing in y
    # Staggered grid coordinates for each level
    x_centers::Vector{Vector{T}}  # Cell center coordinates
    y_centers::Vector{Vector{T}}
    x_faces::Vector{Vector{T}}    # x-face coordinates (for u)
    y_faces::Vector{Vector{T}}    # y-face coordinates (for v)
    # Solver parameters
    n_smooth::Int
    tol::T
    
    function StaggeredMultiLevelPoisson{T}(grid::StaggeredGrid, levels::Int=4; 
                                          n_smooth::Int=3, tol::T=1e-6) where T
        
        # Extract grid information (handle 2D XZ by mapping z→y)
        if grid.grid_type == TwoDimensional
            nx_fine = grid.nx
            ny_fine = grid.nz
            dx_fine = grid.dx
            dy_fine = grid.dz
        else
            nx_fine = grid.nx
            ny_fine = grid.ny
            dx_fine = grid.dx
            dy_fine = grid.dy
        end
        
        # Initialize arrays for all levels
        φ_levels = Vector{Matrix{T}}(undef, levels)
        r_levels = Vector{Matrix{T}}(undef, levels)  
        b_levels = Vector{Matrix{T}}(undef, levels)
        nx_levels = Vector{Int}(undef, levels)
        ny_levels = Vector{Int}(undef, levels)
        dx_levels = Vector{T}(undef, levels)
        dy_levels = Vector{T}(undef, levels)
        x_centers_levels = Vector{Vector{T}}(undef, levels)
        y_centers_levels = Vector{Vector{T}}(undef, levels)
        x_faces_levels = Vector{Vector{T}}(undef, levels)
        y_faces_levels = Vector{Vector{T}}(undef, levels)
        
        # Create staggered grid hierarchy
        curr_nx, curr_ny = nx_fine, ny_fine
        curr_dx, curr_dy = dx_fine, dy_fine
        origin_x, origin_y = 0.0, 0.0  # Assume origin at (0,0)
        
        for level = 1:levels
            nx_levels[level] = curr_nx
            ny_levels[level] = curr_ny
            dx_levels[level] = curr_dx
            dy_levels[level] = curr_dy
            
            # Create staggered coordinates for this level
            # Cell centers (where pressure lives)
            x_centers_levels[level] = collect(origin_x .+ (0.5:curr_nx-0.5) * curr_dx)
            y_centers_levels[level] = collect(origin_y .+ (0.5:curr_ny-0.5) * curr_dy)
            
            # Cell faces (where velocities live) 
            x_faces_levels[level] = collect(origin_x .+ (0:curr_nx) * curr_dx)
            y_faces_levels[level] = collect(origin_y .+ (0:curr_ny) * curr_dy)
            
            # Allocate pressure arrays (all at cell centers)
            φ_levels[level] = zeros(T, curr_nx, curr_ny)
            r_levels[level] = zeros(T, curr_nx, curr_ny)
            b_levels[level] = zeros(T, curr_nx, curr_ny)
            
            # Coarsen grid for next level
            if level < levels
                curr_nx = max(3, curr_nx ÷ 2)
                curr_ny = max(3, curr_ny ÷ 2)  
                curr_dx *= 2
                curr_dy *= 2
            end
        end
        
        new(levels, φ_levels, r_levels, b_levels, nx_levels, ny_levels,
            dx_levels, dy_levels, x_centers_levels, y_centers_levels,
            x_faces_levels, y_faces_levels, n_smooth, tol)
    end
end

StaggeredMultiLevelPoisson(grid::StaggeredGrid, levels::Int=4; kwargs...) = 
    StaggeredMultiLevelPoisson{Float64}(grid, levels; kwargs...)

"""
    solve_staggered_poisson!(φ, rhs, mg)

Solve ∇²φ = rhs where φ is at cell centers and rhs comes from velocity divergence.
"""
function solve_staggered_poisson!(φ::Matrix{T}, rhs::Matrix{T}, 
                                 mg::StaggeredMultiLevelPoisson{T}; 
                                 max_iter::Int=50) where T
    
    # Initialize finest level with staggered grid data
    mg.φ[1] .= φ
    mg.b[1] .= rhs
    
    initial_residual = compute_staggered_residual_norm(mg, 1)
    
    for iter = 1:max_iter
        # Perform staggered V-cycle
        staggered_v_cycle!(mg, 1)
        
        # Check convergence
        residual_norm = compute_staggered_residual_norm(mg, 1)
        
        if residual_norm < mg.tol * initial_residual
            φ .= mg.φ[1]
            return residual_norm, iter
        end
    end
    
    φ .= mg.φ[1]
    return compute_staggered_residual_norm(mg, 1), max_iter
end

"""
    staggered_v_cycle!(mg, level)

V-cycle specifically for staggered grids with pressure at cell centers.
"""
function staggered_v_cycle!(mg::StaggeredMultiLevelPoisson, level::Int)
    if level == mg.levels
        # Coarsest level: exact solve
        staggered_exact_solve!(mg, level)
    else
        # Pre-smoothing with staggered grid awareness
        for _ = 1:mg.n_smooth
            staggered_gauss_seidel_smooth!(mg, level)
        end
        
        # Compute residual and restrict (both at cell centers)
        compute_staggered_residual!(mg, level)
        staggered_restrict!(mg, level)
        
        # Initialize coarser level
        mg.φ[level+1] .= 0.0
        
        # Recursively solve
        staggered_v_cycle!(mg, level+1)
        
        # Prolongate and correct (cell center to cell center)
        staggered_prolongate_and_correct!(mg, level)
        
        # Post-smoothing
        for _ = 1:mg.n_smooth
            staggered_gauss_seidel_smooth!(mg, level)
        end
    end
end

"""
    staggered_gauss_seidel_smooth!(mg, level)

Gauss-Seidel smoothing for pressure at cell centers.
Uses proper staggered grid boundary conditions.
"""
function staggered_gauss_seidel_smooth!(mg::StaggeredMultiLevelPoisson{T}, level::Int) where T
    φ = mg.φ[level]
    b = mg.b[level]
    nx, ny = mg.nx[level], mg.ny[level]
    dx, dy = mg.dx[level], mg.dy[level]
    
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    factor = 1.0 / (2.0 * (dx2_inv + dy2_inv))
    
    # Red-black Gauss-Seidel on cell centers
    for color = 0:1
        for j = 2:ny-1, i = 2:nx-1
            if (i + j) % 2 == color
                # Standard 5-point Laplacian at cell center
                φ[i, j] = factor * (
                    dx2_inv * (φ[i-1, j] + φ[i+1, j]) +
                    dy2_inv * (φ[i, j-1] + φ[i, j+1]) -
                    b[i, j]
                )
            end
        end
    end
    
    # Apply staggered grid boundary conditions for pressure
    apply_staggered_pressure_bc!(φ, nx, ny)
end

"""
    compute_staggered_residual!(mg, level)

Compute residual for pressure equation at cell centers.
"""
function compute_staggered_residual!(mg::StaggeredMultiLevelPoisson{T}, level::Int) where T
    φ = mg.φ[level]
    b = mg.b[level]
    r = mg.r[level]
    nx, ny = mg.nx[level], mg.ny[level]
    dx, dy = mg.dx[level], mg.dy[level]
    
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    
    # Compute residual at cell centers: r = b - L(φ)
    for j = 2:ny-1, i = 2:nx-1
        laplacian_φ = dx2_inv * (φ[i-1, j] - 2*φ[i, j] + φ[i+1, j]) +
                      dy2_inv * (φ[i, j-1] - 2*φ[i, j] + φ[i, j+1])
        r[i, j] = b[i, j] - laplacian_φ
    end
    
    # Boundary residuals are zero for Neumann BC
    r[1, :] .= 0.0
    r[nx, :] .= 0.0
    r[:, 1] .= 0.0
    r[:, ny] .= 0.0
end

"""
    staggered_restrict!(mg, level)

Restriction operator for cell-centered pressure data.
"""
function staggered_restrict!(mg::StaggeredMultiLevelPoisson{T}, level::Int) where T
    r_fine = mg.r[level]
    b_coarse = mg.b[level+1]
    
    nx_fine, ny_fine = mg.nx[level], mg.ny[level]
    nx_coarse, ny_coarse = mg.nx[level+1], mg.ny[level+1]
    
    # Full weighting restriction from cell centers to cell centers
    for j = 2:ny_coarse-1, i = 2:nx_coarse-1
        # Map to fine grid indices
        i_fine = 2*i - 1
        j_fine = 2*j - 1
        
        if i_fine <= nx_fine && j_fine <= ny_fine
            # Full weighting: 1/16 * [1 2 1; 2 4 2; 1 2 1] stencil
            b_coarse[i, j] = 0.0625 * (
                1.0 * (r_fine[i_fine-1, j_fine-1] + r_fine[i_fine+1, j_fine-1] + 
                       r_fine[i_fine-1, j_fine+1] + r_fine[i_fine+1, j_fine+1]) +
                2.0 * (r_fine[i_fine, j_fine-1] + r_fine[i_fine, j_fine+1] + 
                       r_fine[i_fine-1, j_fine] + r_fine[i_fine+1, j_fine]) +
                4.0 * r_fine[i_fine, j_fine]
            )
        end
    end
    
    # Boundary handling
    b_coarse[1, :] .= 0.0
    b_coarse[nx_coarse, :] .= 0.0
    b_coarse[:, 1] .= 0.0
    b_coarse[:, ny_coarse] .= 0.0
end

"""
    staggered_prolongate_and_correct!(mg, level)

Prolongation from coarse cell centers to fine cell centers.
"""
function staggered_prolongate_and_correct!(mg::StaggeredMultiLevelPoisson{T}, level::Int) where T
    φ_fine = mg.φ[level]
    φ_coarse = mg.φ[level+1]
    
    nx_fine, ny_fine = mg.nx[level], mg.ny[level]
    nx_coarse, ny_coarse = mg.nx[level+1], mg.ny[level+1]
    
    # Bilinear prolongation from cell centers to cell centers
    for j = 2:ny_fine-1, i = 2:nx_fine-1
        # Find corresponding coarse grid indices
        i_coarse_real = (i + 1.0) / 2.0
        j_coarse_real = (j + 1.0) / 2.0
        
        i_coarse = Int(floor(i_coarse_real))
        j_coarse = Int(floor(j_coarse_real))
        
        # Bilinear weights
        α = i_coarse_real - i_coarse
        β = j_coarse_real - j_coarse
        
        # Bounds checking
        i1 = max(1, min(nx_coarse, i_coarse))
        i2 = max(1, min(nx_coarse, i_coarse + 1))
        j1 = max(1, min(ny_coarse, j_coarse))
        j2 = max(1, min(ny_coarse, j_coarse + 1))
        
        # Bilinear interpolation
        correction = (1-α)*(1-β)*φ_coarse[i1, j1] + α*(1-β)*φ_coarse[i2, j1] +
                    (1-α)*β*φ_coarse[i1, j2] + α*β*φ_coarse[i2, j2]
                    
        φ_fine[i, j] += correction
    end
end

"""
    apply_staggered_pressure_bc!(φ, nx, ny)

Apply boundary conditions appropriate for pressure in staggered grid.
For incompressible flow, pressure typically has Neumann BC: ∂φ/∂n = 0.
"""
function apply_staggered_pressure_bc!(φ::Matrix{T}, nx::Int, ny::Int) where T
    # Homogeneous Neumann boundary conditions for pressure
    # ∂φ/∂x = 0 at x-boundaries
    φ[1, :] .= φ[2, :]        # Left boundary
    φ[nx, :] .= φ[nx-1, :]    # Right boundary
    
    # ∂φ/∂y = 0 at y-boundaries  
    φ[:, 1] .= φ[:, 2]        # Bottom boundary
    φ[:, ny] .= φ[:, ny-1]    # Top boundary
end

"""
    staggered_exact_solve!(mg, level)

Exact solve on coarsest level using many iterations.
"""
function staggered_exact_solve!(mg::StaggeredMultiLevelPoisson{T}, level::Int) where T
    for _ = 1:100
        staggered_gauss_seidel_smooth!(mg, level)
    end
end

"""
    compute_staggered_residual_norm(mg, level)

Compute L2 norm of residual for convergence checking.
"""
function compute_staggered_residual_norm(mg::StaggeredMultiLevelPoisson{T}, level::Int) where T
    compute_staggered_residual!(mg, level)
    r = mg.r[level]
    nx, ny = mg.nx[level], mg.ny[level]
    
    # L2 norm of interior residual
    norm_sq = 0.0
    for j = 2:ny-1, i = 2:nx-1
        norm_sq += r[i, j]^2
    end
    
    return sqrt(norm_sq / ((nx-2) * (ny-2)))
end

"""
    compute_pressure_gradient_to_faces!(dpdx_faces, dpdy_faces, φ, grid)

Compute pressure gradient from cell centers to face centers for velocity correction.
This is the key operation that connects pressure (cell centers) to velocities (faces).
"""
function compute_pressure_gradient_to_faces!(dpdx_faces::Matrix{T}, dpdy_faces::Matrix{T}, 
                                           φ::Matrix{T}, grid::StaggeredGrid) where T
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    
    # Pressure gradient to x-faces (for u-velocity correction)
    # dpdx_faces[i,j] is gradient at face between cells (i-1,j) and (i,j)
    for j = 1:ny
        for i = 1:nx+1
            if i == 1
                # Left boundary: use one-sided difference
                dpdx_faces[i, j] = (φ[1, j] - 0.0) / (0.5 * dx)  # Assuming φ = 0 outside
            elseif i == nx+1
                # Right boundary: use one-sided difference  
                dpdx_faces[i, j] = (0.0 - φ[nx, j]) / (0.5 * dx)
            else
                # Interior: central difference between adjacent cell centers
                dpdx_faces[i, j] = (φ[i, j] - φ[i-1, j]) / dx
            end
        end
    end
    
    # Pressure gradient to y-faces (for v-velocity correction)
    # dpdy_faces[i,j] is gradient at face between cells (i,j-1) and (i,j)
    for j = 1:ny+1
        for i = 1:nx
            if j == 1
                # Bottom boundary
                dpdy_faces[i, j] = (φ[i, 1] - 0.0) / (0.5 * dy)
            elseif j == ny+1
                # Top boundary
                dpdy_faces[i, j] = (0.0 - φ[i, ny]) / (0.5 * dy)
            else
                # Interior: central difference
                dpdy_faces[i, j] = (φ[i, j] - φ[i, j-1]) / dy
            end
        end
    end
end

"""
    compute_velocity_divergence_from_faces!(div_u, u_faces, v_faces, grid)

Compute velocity divergence at cell centers from face-centered velocities.
This is the RHS computation for the pressure Poisson equation.
"""
function compute_velocity_divergence_from_faces!(div_u::Matrix{T}, u_faces::Matrix{T}, 
                                               v_faces::Matrix{T}, grid::StaggeredGrid) where T
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    
    # Divergence at cell centers from face-centered velocities
    for j = 1:ny, i = 1:nx
        # ∇·u = (u_{i+1/2,j} - u_{i-1/2,j})/dx + (v_{i,j+1/2} - v_{i,j-1/2})/dy
        dudx = (u_faces[i+1, j] - u_faces[i, j]) / dx
        dvdy = (v_faces[i, j+1] - v_faces[i, j]) / dy
        div_u[i, j] = dudx + dvdy
    end
end
