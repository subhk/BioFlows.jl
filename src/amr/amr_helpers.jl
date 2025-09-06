"""
AMR Helper Functions for BioFlow.jl

This module provides helper functions for adaptive mesh refinement that integrate
properly with the existing codebase structure and coordinate systems.
"""

"""
    distance_to_surface_2d(body, x, z)

Compute distance to surface for 2D rigid body in XZ plane.
"""
function distance_to_surface_2d(body, x::Float64, z::Float64)
    # This is a placeholder - actual implementation would depend on body geometry
    # For now, assume circular body for testing
    if hasfield(typeof(body), :center) && hasfield(typeof(body), :radius)
        center_x = body.center[1]
        center_z = body.center[2]  # In XZ plane, second coordinate is z
        radius = body.radius
        
        dist_to_center = sqrt((x - center_x)^2 + (z - center_z)^2)
        return dist_to_center - radius
    else
        # Fallback: return large distance if body structure unknown
        return 1000.0
    end
end

"""
    distance_to_surface_3d(body, x, y, z)

Compute distance to surface for 3D rigid body.
"""
function distance_to_surface_3d(body, x::Float64, y::Float64, z::Float64)
    # This is a placeholder - actual implementation would depend on body geometry
    # For now, assume spherical body for testing
    if hasfield(typeof(body), :center) && hasfield(typeof(body), :radius)
        center_x = body.center[1]
        center_y = body.center[2]
        center_z = body.center[3]
        radius = body.radius
        
        dist_to_center = sqrt((x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2)
        return dist_to_center - radius
    else
        # Fallback: return large distance if body structure unknown
        return 1000.0
    end
end

"""
    d2dx2(field, grid)

Compute second derivative in x-direction using 2nd order finite differences.
"""
function d2dx2(field::Matrix{Float64}, grid::StaggeredGrid)
    nx, nz = size(field)
    d2fdx2 = zeros(nx, nz)
    dx = grid.dx
    
    # Interior points using centered differences
    @inbounds for j = 1:nz, i = 2:nx-1
        d2fdx2[i, j] = (field[i+1, j] - 2*field[i, j] + field[i-1, j]) / (dx^2)
    end
    
    # Boundary conditions - use one-sided differences
    @inbounds for j = 1:nz
        # Left boundary
        d2fdx2[1, j] = (field[3, j] - 2*field[2, j] + field[1, j]) / (dx^2)
        # Right boundary
        d2fdx2[nx, j] = (field[nx, j] - 2*field[nx-1, j] + field[nx-2, j]) / (dx^2)
    end
    
    return d2fdx2
end

"""
    d2dy2(field, grid)

Compute second derivative in y-direction (or z-direction for XZ plane).
"""
function d2dy2(field::Matrix{Float64}, grid::StaggeredGrid)
    if grid.grid_type == TwoDimensional
        # For XZ plane, this is actually d2/dz2
        return d2dz2_2d(field, grid)
    else
        # For 3D, this is d2/dy2
        return d2dy2_3d(field, grid)
    end
end

"""
    d2dz2_2d(field, grid)

Compute second derivative in z-direction for 2D XZ plane.
"""
function d2dz2_2d(field::Matrix{Float64}, grid::StaggeredGrid)
    nx, nz = size(field)
    d2fdz2 = zeros(nx, nz)
    dz = grid.dz
    
    # Interior points using centered differences
    @inbounds for j = 2:nz-1, i = 1:nx
        d2fdz2[i, j] = (field[i, j+1] - 2*field[i, j] + field[i, j-1]) / (dz^2)
    end
    
    # Boundary conditions - use one-sided differences
    @inbounds for i = 1:nx
        # Bottom boundary
        d2fdz2[i, 1] = (field[i, 3] - 2*field[i, 2] + field[i, 1]) / (dz^2)
        # Top boundary
        d2fdz2[i, nz] = (field[i, nz] - 2*field[i, nz-1] + field[i, nz-2]) / (dz^2)
    end
    
    return d2fdz2
end

"""
    d2dy2_3d(field, grid)

Compute second derivative in y-direction for 3D.
"""
function d2dy2_3d(field::Array{Float64,3}, grid::StaggeredGrid)
    nx, ny, nz = size(field)
    d2fdy2 = zeros(nx, ny, nz)
    dy = grid.dy
    
    # Interior points using centered differences
    @inbounds for k = 1:nz, j = 2:ny-1, i = 1:nx
        d2fdy2[i, j, k] = (field[i, j+1, k] - 2*field[i, j, k] + field[i, j-1, k]) / (dy^2)
    end
    
    # Boundary conditions - use one-sided differences
    @inbounds for k = 1:nz, i = 1:nx
        # Bottom boundary
        d2fdy2[i, 1, k] = (field[i, 3, k] - 2*field[i, 2, k] + field[i, 1, k]) / (dy^2)
        # Top boundary
        d2fdy2[i, ny, k] = (field[i, ny, k] - 2*field[i, ny-1, k] + field[i, ny-2, k]) / (dy^2)
    end
    
    return d2fdy2
end

"""
    create_mg_solver_for_level(amr_level)

Create a multigrid solver for a specific AMR level.
"""
function create_mg_solver_for_level(amr_level)
    # Create staggered grid for this AMR level
    if amr_level.ny == 0  # This indicates 2D case (nx, nz dimensions)
        # This is actually a 2D XZ plane case
        Lx = amr_level.x_max - amr_level.x_min
        Lz = amr_level.y_max - amr_level.y_min  # y maps to z in XZ plane
        
        local_grid = StaggeredGrid2D(amr_level.nx, amr_level.nx,  # Assuming square cells for simplicity
                                    Lx, Lz;
                                    origin_x=amr_level.x_min,
                                    origin_z=amr_level.y_min)
    else
        # This is a 3D case
        Lx = amr_level.x_max - amr_level.x_min
        Ly = amr_level.y_max - amr_level.y_min
        Lz = amr_level.x_max - amr_level.x_min  # Placeholder - need actual z dimension
        
        local_grid = StaggeredGrid3D(amr_level.nx, amr_level.ny, amr_level.nx,  # Placeholder dimensions
                                    Lx, Ly, Lz;
                                    origin_x=amr_level.x_min,
                                    origin_y=amr_level.y_min,
                                    origin_z=0.0)
    end
    
    # Create staggered-aware multigrid solver
    return MultigridPoissonSolver(local_grid; smoother=:staggered, tolerance=1e-8)
end

# Export AMR helper functions
export distance_to_surface_2d, distance_to_surface_3d
export d2dx2, d2dy2, d2dz2_2d, d2dy2_3d
export create_mg_solver_for_level
