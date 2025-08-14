"""
Differential Operators for Staggered Grid Finite Volume Method

This module provides clean, readable differential operators for 2nd order accurate
finite volume discretization on staggered grids. All operators are designed to be
intuitive and easy to debug.

Naming Convention:
- ddx, ddy, ddz: First derivatives
- d2dx2, d2dy2, d2dz2: Second derivatives  
- div: Divergence operator
- grad: Gradient operator
- laplacian: Laplacian operator (∇²)

Grid Layout:
- p, scalars: Cell centers
- u: x-faces (staggered in x)
- v: y-faces (staggered in y)  
- w: z-faces (staggered in z)
"""

# =============================================================================
# First Derivative Operators (2nd Order Accurate)
# =============================================================================

"""
    ddx(field, grid; boundary_order=2)

Compute ∂field/∂x with 2nd order accuracy.
Automatically detects field location (cell-center or face) and applies appropriate stencil.
"""
function ddx(field::Matrix{T}, grid::StaggeredGrid; boundary_order::Int=2) where T
    nx, ny = size(field)
    result = zeros(T, nx, ny)
    dx = grid.dx
    
    if boundary_order == 2
        # 2nd order everywhere
        for j = 1:ny, i = 1:nx
            if i == 1
                # Left boundary: 2nd order forward difference
                result[i, j] = (-3*field[i, j] + 4*field[i+1, j] - field[i+2, j]) / (2*dx)
            elseif i == nx
                # Right boundary: 2nd order backward difference
                result[i, j] = (3*field[i, j] - 4*field[i-1, j] + field[i-2, j]) / (2*dx)
            else
                # Interior: 2nd order central difference
                result[i, j] = (field[i+1, j] - field[i-1, j]) / (2*dx)
            end
        end
    else
        # 1st order for comparison/debugging
        for j = 1:ny, i = 1:nx-1
            result[i, j] = (field[i+1, j] - field[i, j]) / dx
        end
    end
    
    return result
end

"""
    ddx_at_faces(field, grid)

Compute ∂field/∂x at x-faces (for u-velocity locations).
field should be at cell centers, result is at u-velocity points.
"""
function ddx_at_faces(field::Matrix{T}, grid::StaggeredGrid) where T
    nx, ny = size(field)
    result = zeros(T, nx+1, ny)
    dx = grid.dx
    
    for j = 1:ny
        # First face (left boundary)
        result[1, j] = (-3*field[1, j] + 4*field[2, j] - field[3, j]) / (2*dx)
        
        # Interior faces
        for i = 2:nx
            result[i, j] = (field[i, j] - field[i-1, j]) / dx
        end
        
        # Last face (right boundary)
        result[nx+1, j] = (3*field[nx, j] - 4*field[nx-1, j] + field[nx-2, j]) / (2*dx)
    end
    
    return result
end

"""
    ddy(field, grid; boundary_order=2)

Compute ∂field/∂y with 2nd order accuracy.
"""
function ddy(field::Matrix{T}, grid::StaggeredGrid; boundary_order::Int=2) where T
    nx, ny = size(field)
    result = zeros(T, nx, ny)
    dy = grid.dy
    
    if boundary_order == 2
        for j = 1:ny, i = 1:nx
            if j == 1
                # Bottom boundary: 2nd order forward difference
                result[i, j] = (-3*field[i, j] + 4*field[i, j+1] - field[i, j+2]) / (2*dy)
            elseif j == ny
                # Top boundary: 2nd order backward difference
                result[i, j] = (3*field[i, j] - 4*field[i, j-1] + field[i, j-2]) / (2*dy)
            else
                # Interior: 2nd order central difference
                result[i, j] = (field[i, j+1] - field[i, j-1]) / (2*dy)
            end
        end
    else
        for j = 1:ny-1, i = 1:nx
            result[i, j] = (field[i, j+1] - field[i, j]) / dy
        end
    end
    
    return result
end

"""
    ddy_at_faces(field, grid)

Compute ∂field/∂y at y-faces (for v-velocity locations).
"""
function ddy_at_faces(field::Matrix{T}, grid::StaggeredGrid) where T
    nx, ny = size(field)
    result = zeros(T, nx, ny+1)
    dy = grid.dy
    
    for i = 1:nx
        # First face (bottom boundary)
        result[i, 1] = (-3*field[i, 1] + 4*field[i, 2] - field[i, 3]) / (2*dy)
        
        # Interior faces
        for j = 2:ny
            result[i, j] = (field[i, j] - field[i, j-1]) / dy
        end
        
        # Last face (top boundary)
        result[i, ny+1] = (3*field[i, ny] - 4*field[i, ny-1] + field[i, ny-2]) / (2*dy)
    end
    
    return result
end

# =============================================================================
# 3D First Derivative Operators
# =============================================================================

function ddx(field::Array{T,3}, grid::StaggeredGrid; boundary_order::Int=2) where T
    nx, ny, nz = size(field)
    result = zeros(T, nx, ny, nz)
    dx = grid.dx
    
    if boundary_order == 2
        for k = 1:nz, j = 1:ny, i = 1:nx
            if i == 1
                result[i, j, k] = (-3*field[i, j, k] + 4*field[i+1, j, k] - field[i+2, j, k]) / (2*dx)
            elseif i == nx
                result[i, j, k] = (3*field[i, j, k] - 4*field[i-1, j, k] + field[i-2, j, k]) / (2*dx)
            else
                result[i, j, k] = (field[i+1, j, k] - field[i-1, j, k]) / (2*dx)
            end
        end
    end
    
    return result
end

function ddy(field::Array{T,3}, grid::StaggeredGrid; boundary_order::Int=2) where T
    nx, ny, nz = size(field)
    result = zeros(T, nx, ny, nz)
    dy = grid.dy
    
    if boundary_order == 2
        for k = 1:nz, j = 1:ny, i = 1:nx
            if j == 1
                result[i, j, k] = (-3*field[i, j, k] + 4*field[i, j+1, k] - field[i, j+2, k]) / (2*dy)
            elseif j == ny
                result[i, j, k] = (3*field[i, j, k] - 4*field[i, j-1, k] + field[i, j-2, k]) / (2*dy)
            else
                result[i, j, k] = (field[i, j+1, k] - field[i, j-1, k]) / (2*dy)
            end
        end
    end
    
    return result
end

function ddz(field::Array{T,3}, grid::StaggeredGrid; boundary_order::Int=2) where T
    nx, ny, nz = size(field)
    result = zeros(T, nx, ny, nz)
    dz = grid.dz
    
    if boundary_order == 2
        for k = 1:nz, j = 1:ny, i = 1:nx
            if k == 1
                result[i, j, k] = (-3*field[i, j, k] + 4*field[i, j, k+1] - field[i, j, k+2]) / (2*dz)
            elseif k == nz
                result[i, j, k] = (3*field[i, j, k] - 4*field[i, j, k-1] + field[i, j, k-2]) / (2*dz)
            else
                result[i, j, k] = (field[i, j, k+1] - field[i, j, k-1]) / (2*dz)
            end
        end
    end
    
    return result
end

# =============================================================================
# Second Derivative Operators (2nd Order Accurate)
# =============================================================================

"""
    d2dx2(field, grid)

Compute ∂²field/∂x² with 2nd order accuracy.
"""
function d2dx2(field::Matrix{T}, grid::StaggeredGrid) where T
    nx, ny = size(field)
    result = zeros(T, nx, ny)
    dx = grid.dx
    
    for j = 1:ny, i = 2:nx-1
        result[i, j] = (field[i+1, j] - 2*field[i, j] + field[i-1, j]) / dx^2
    end
    
    # Boundary conditions (homogeneous Neumann for demonstration)
    for j = 1:ny
        result[1, j] = result[2, j]      # ∂²f/∂x² at boundary = interior value
        result[nx, j] = result[nx-1, j]
    end
    
    return result
end

"""
    d2dy2(field, grid)

Compute ∂²field/∂y² with 2nd order accuracy.
"""
function d2dy2(field::Matrix{T}, grid::StaggeredGrid) where T
    nx, ny = size(field)
    result = zeros(T, nx, ny)
    dy = grid.dy
    
    for j = 2:ny-1, i = 1:nx
        result[i, j] = (field[i, j+1] - 2*field[i, j] + field[i, j-1]) / dy^2
    end
    
    # Boundary conditions
    for i = 1:nx
        result[i, 1] = result[i, 2]
        result[i, ny] = result[i, ny-1]
    end
    
    return result
end

function d2dx2(field::Array{T,3}, grid::StaggeredGrid) where T
    nx, ny, nz = size(field)
    result = zeros(T, nx, ny, nz)
    dx = grid.dx
    
    for k = 1:nz, j = 1:ny, i = 2:nx-1
        result[i, j, k] = (field[i+1, j, k] - 2*field[i, j, k] + field[i-1, j, k]) / dx^2
    end
    
    return result
end

function d2dy2(field::Array{T,3}, grid::StaggeredGrid) where T
    nx, ny, nz = size(field)
    result = zeros(T, nx, ny, nz)
    dy = grid.dy
    
    for k = 1:nz, j = 2:ny-1, i = 1:nx
        result[i, j, k] = (field[i, j+1, k] - 2*field[i, j, k] + field[i, j-1, k]) / dy^2
    end
    
    return result
end

function d2dz2(field::Array{T,3}, grid::StaggeredGrid) where T
    nx, ny, nz = size(field)
    result = zeros(T, nx, ny, nz)
    dz = grid.dz
    
    for k = 2:nz-1, j = 1:ny, i = 1:nx
        result[i, j, k] = (field[i, j, k+1] - 2*field[i, j, k] + field[i, j, k-1]) / dz^2
    end
    
    return result
end

# =============================================================================
# Vector Operators
# =============================================================================

"""
    div(u, v, grid)
    div(u, v, w, grid)

Compute divergence ∇·u with 2nd order accuracy on staggered grid.
Returns divergence at cell centers.
"""
function div(u::Matrix{T}, v::Matrix{T}, grid::StaggeredGrid) where T
    nx, ny = grid.nx, grid.ny  # Cell-centered grid size
    result = zeros(T, nx, ny)
    dx, dy = grid.dx, grid.dy
    
    # Divergence: ∂u/∂x + ∂v/∂y
    # u is at x-faces, v is at y-faces, result at cell centers
    for j = 1:ny, i = 1:nx
        dudx = (u[i+1, j] - u[i, j]) / dx      # u[i+1,j] is right face, u[i,j] is left face
        dvdy = (v[i, j+1] - v[i, j]) / dy      # v[i,j+1] is top face, v[i,j] is bottom face
        result[i, j] = dudx + dvdy
    end
    
    return result
end

function div(u::Array{T,3}, v::Array{T,3}, w::Array{T,3}, grid::StaggeredGrid) where T
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    result = zeros(T, nx, ny, nz)
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    
    for k = 1:nz, j = 1:ny, i = 1:nx
        dudx = (u[i+1, j, k] - u[i, j, k]) / dx
        dvdy = (v[i, j+1, k] - v[i, j, k]) / dy
        dwdz = (w[i, j, k+1] - w[i, j, k]) / dz
        result[i, j, k] = dudx + dvdy + dwdz
    end
    
    return result
end

"""
    grad(p, grid)

Compute gradient ∇p with 2nd order accuracy.
Returns gradient components at staggered locations (u-faces, v-faces, w-faces).
"""
function grad(p::Matrix{T}, grid::StaggeredGrid) where T
    dpdx = ddx_at_faces(p, grid)  # At u-velocity points
    dpdy = ddy_at_faces(p, grid)  # At v-velocity points
    
    return dpdx, dpdy
end

function grad(p::Array{T,3}, grid::StaggeredGrid) where T
    nx, ny, nz = size(p)
    
    # Pressure gradient at u-velocity points (x-faces)
    dpdx = zeros(T, nx+1, ny, nz)
    for k = 1:nz, j = 1:ny
        for i = 1:nx+1
            if i == 1
                dpdx[i, j, k] = (-3*p[1, j, k] + 4*p[2, j, k] - p[3, j, k]) / (2*grid.dx)
            elseif i == nx+1
                dpdx[i, j, k] = (3*p[nx, j, k] - 4*p[nx-1, j, k] + p[nx-2, j, k]) / (2*grid.dx)
            else
                dpdx[i, j, k] = (p[i, j, k] - p[i-1, j, k]) / grid.dx
            end
        end
    end
    
    # Pressure gradient at v-velocity points (y-faces)
    dpdy = zeros(T, nx, ny+1, nz)
    for k = 1:nz, j = 1:ny+1, i = 1:nx
        if j == 1
            dpdy[i, j, k] = (-3*p[i, 1, k] + 4*p[i, 2, k] - p[i, 3, k]) / (2*grid.dy)
        elseif j == ny+1
            dpdy[i, j, k] = (3*p[i, ny, k] - 4*p[i, ny-1, k] + p[i, ny-2, k]) / (2*grid.dy)
        else
            dpdy[i, j, k] = (p[i, j, k] - p[i, j-1, k]) / grid.dy
        end
    end
    
    # Pressure gradient at w-velocity points (z-faces)
    dpdz = zeros(T, nx, ny, nz+1)
    for k = 1:nz+1, j = 1:ny, i = 1:nx
        if k == 1
            dpdz[i, j, k] = (-3*p[i, j, 1] + 4*p[i, j, 2] - p[i, j, 3]) / (2*grid.dz)
        elseif k == nz+1
            dpdz[i, j, k] = (3*p[i, j, nz] - 4*p[i, j, nz-1] + p[i, j, nz-2]) / (2*grid.dz)
        else
            dpdz[i, j, k] = (p[i, j, k] - p[i, j, k-1]) / grid.dz
        end
    end
    
    return dpdx, dpdy, dpdz
end

"""
    laplacian(field, grid)

Compute Laplacian ∇²field = ∂²field/∂x² + ∂²field/∂y² + ∂²field/∂z².
"""
function laplacian(field::Matrix{T}, grid::StaggeredGrid) where T
    return d2dx2(field, grid) + d2dy2(field, grid)
end

function laplacian(field::Array{T,3}, grid::StaggeredGrid) where T
    return d2dx2(field, grid) + d2dy2(field, grid) + d2dz2(field, grid)
end

# =============================================================================
# Interpolation Operators
# =============================================================================

"""
    interpolate_u_to_cell_center(u, grid)

Interpolate u-velocity from x-faces to cell centers.
"""
function interpolate_u_to_cell_center(u::Matrix{T}, grid::StaggeredGrid) where T
    nx, ny = grid.nx, grid.ny
    u_cc = zeros(T, nx, ny)
    
    for j = 1:ny, i = 1:nx
        u_cc[i, j] = 0.5 * (u[i, j] + u[i+1, j])
    end
    
    return u_cc
end

"""
    interpolate_v_to_cell_center(v, grid)

Interpolate v-velocity from y-faces to cell centers.
"""
function interpolate_v_to_cell_center(v::Matrix{T}, grid::StaggeredGrid) where T
    nx, ny = grid.nx, grid.ny
    v_cc = zeros(T, nx, ny)
    
    for j = 1:ny, i = 1:nx
        v_cc[i, j] = 0.5 * (v[i, j] + v[i, j+1])
    end
    
    return v_cc
end

"""
    interpolate_to_cell_centers(u, v, grid)
    interpolate_to_cell_centers(u, v, w, grid)

Interpolate all velocity components to cell centers for post-processing.
"""
function interpolate_to_cell_centers(u::Matrix{T,2}, v::Matrix{T,2}, grid::StaggeredGrid) where T
    u_cc = interpolate_u_to_cell_center(u, grid)
    v_cc = interpolate_v_to_cell_center(v, grid)
    return u_cc, v_cc
end

function interpolate_to_cell_centers(u::Array{T,3}, v::Array{T,3}, w::Array{T,3}, grid::StaggeredGrid) where T
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    u_cc = zeros(T, nx, ny, nz)
    v_cc = zeros(T, nx, ny, nz)
    w_cc = zeros(T, nx, ny, nz)
    
    for k = 1:nz, j = 1:ny, i = 1:nx
        u_cc[i, j, k] = 0.5 * (u[i, j, k] + u[i+1, j, k])
        v_cc[i, j, k] = 0.5 * (v[i, j, k] + v[i, j+1, k])
        w_cc[i, j, k] = 0.5 * (w[i, j, k] + w[i, j, k+1])
    end
    
    return u_cc, v_cc, w_cc
end

# =============================================================================
# Debug and Verification Functions
# =============================================================================

"""
    verify_operator_accuracy(grid)

Test differential operators against known analytical functions.
Useful for debugging and verification.
"""
function verify_operator_accuracy(grid::StaggeredGrid)
    println("Testing differential operator accuracy...")
    
    # Test function: f(x,y) = sin(πx/Lx) * cos(πy/Ly)
    nx, ny = grid.nx, grid.ny
    
    # Create test field at cell centers
    f = zeros(nx, ny)
    for j = 1:ny, i = 1:nx
        x = grid.x[i]
        y = grid.y[j]
        f[i, j] = sin(π * x / grid.Lx) * cos(π * y / grid.Ly)
    end
    
    # Analytical derivatives
    dfdx_analytical = zeros(nx, ny)
    dfdy_analytical = zeros(nx, ny)
    d2fdx2_analytical = zeros(nx, ny)
    
    for j = 1:ny, i = 1:nx
        x = grid.x[i]
        y = grid.y[j]
        dfdx_analytical[i, j] = (π / grid.Lx) * cos(π * x / grid.Lx) * cos(π * y / grid.Ly)
        dfdy_analytical[i, j] = -(π / grid.Ly) * sin(π * x / grid.Lx) * sin(π * y / grid.Ly)
        d2fdx2_analytical[i, j] = -(π / grid.Lx)^2 * sin(π * x / grid.Lx) * cos(π * y / grid.Ly)
    end
    
    # Numerical derivatives
    dfdx_numerical = ddx(f, grid)
    dfdy_numerical = ddy(f, grid)
    d2fdx2_numerical = d2dx2(f, grid)
    
    # Compute errors (exclude boundaries for fair comparison)
    error_ddx = maximum(abs.(dfdx_numerical[2:end-1, 2:end-1] - dfdx_analytical[2:end-1, 2:end-1]))
    error_ddy = maximum(abs.(dfdy_numerical[2:end-1, 2:end-1] - dfdy_analytical[2:end-1, 2:end-1]))
    error_d2dx2 = maximum(abs.(d2fdx2_numerical[2:end-1, 2:end-1] - d2fdx2_analytical[2:end-1, 2:end-1]))
    
    println("  ddx error: $(error_ddx)")
    println("  ddy error: $(error_ddy)")
    println("  d2dx2 error: $(error_d2dx2)")
    
    # Expected error scaling: O(h²) for 2nd order
    expected_error = (grid.dx^2 + grid.dy^2)
    println("  Expected error scale: $(expected_error)")
    
    return error_ddx, error_ddy, error_d2dx2
end

"""
    check_staggered_grid_consistency(u, v, p, grid)

Verify that arrays are properly sized for staggered grid.
"""
function check_staggered_grid_consistency(u::Matrix, v::Matrix, p::Matrix, grid::StaggeredGrid)
    println("Checking staggered grid array consistency...")
    
    # Expected sizes
    expected_u_size = (grid.nx + 1, grid.ny)
    expected_v_size = (grid.nx, grid.ny + 1)
    expected_p_size = (grid.nx, grid.ny)
    
    u_ok = size(u) == expected_u_size
    v_ok = size(v) == expected_v_size
    p_ok = size(p) == expected_p_size
    
    println("  u size: $(size(u)), expected: $expected_u_size, OK: $u_ok")
    println("  v size: $(size(v)), expected: $expected_v_size, OK: $v_ok")
    println("  p size: $(size(p)), expected: $expected_p_size, OK: $p_ok")
    
    return u_ok && v_ok && p_ok
end