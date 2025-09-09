"""
Differential Operators for Staggered Grid Finite Volume Method

This module provides clean, readable differential operators for 2nd order accurate
finite volume discretization on staggered grids. All operators are designed to be
intuitive and easy to debug.

Grid Layout:
- p, scalars: Cell centers
- u: x-faces (staggered in x), v: y-faces (staggered in y), w: z-faces (staggered in z)
"""

# =============================================================================
# Core Derivative Operators (2nd Order Accurate)
# =============================================================================

"""
    derivative_1d(field, h, dim, boundary_order=2)

Generic 1D derivative computation along specified dimension.
"""
function derivative_1d(field::AbstractArray{T,N}, h::T, dim::Int; boundary_order::Int=2) where {T,N}
    result = similar(field)
    
    if boundary_order == 2
        # 2nd order central/forward/backward differences
        for idx in CartesianIndices(field)
            i = idx[dim]
            n = size(field, dim)
            
            if i == 1
                # Forward difference
                idx_p1 = CartesianIndex(Tuple(idx[j] == i && j == dim ? i+1 : idx[j] for j in 1:N))
                idx_p2 = CartesianIndex(Tuple(idx[j] == i && j == dim ? i+2 : idx[j] for j in 1:N))
                result[idx] = (-3*field[idx] + 4*field[idx_p1] - field[idx_p2]) / (2*h)
            elseif i == n
                # Backward difference
                idx_m1 = CartesianIndex(Tuple(idx[j] == i && j == dim ? i-1 : idx[j] for j in 1:N))
                idx_m2 = CartesianIndex(Tuple(idx[j] == i && j == dim ? i-2 : idx[j] for j in 1:N))
                result[idx] = (3*field[idx] - 4*field[idx_m1] + field[idx_m2]) / (2*h)
            else
                # Central difference
                idx_p1 = CartesianIndex(Tuple(idx[j] == i && j == dim ? i+1 : idx[j] for j in 1:N))
                idx_m1 = CartesianIndex(Tuple(idx[j] == i && j == dim ? i-1 : idx[j] for j in 1:N))
                result[idx] = (field[idx_p1] - field[idx_m1]) / (2*h)
            end
        end
    else
        # 1st order forward difference for debugging
        for idx in CartesianIndices(field)
            i = idx[dim]
            if i < size(field, dim)
                idx_p1 = CartesianIndex(Tuple(idx[j] == i && j == dim ? i+1 : idx[j] for j in 1:N))
                result[idx] = (field[idx_p1] - field[idx]) / h
            end
        end
    end
    
    return result
end

"""
    ddx(field, grid; boundary_order=2)
    ddy(field, grid; boundary_order=2) 
    ddz(field, grid; boundary_order=2)

Compute first derivatives with 2nd order accuracy.
"""
ddx(field::AbstractArray, grid::StaggeredGrid; boundary_order::Int=2) = derivative_1d(field, grid.dx, 1; boundary_order)
ddy(field::AbstractArray, grid::StaggeredGrid; boundary_order::Int=2) = derivative_1d(field, grid.dy, 2; boundary_order)
function ddz(field::AbstractArray, grid::StaggeredGrid; boundary_order::Int=2)
    # For 2D XZ-plane fields, z corresponds to dimension 2
    if ndims(field) == 2
        return derivative_1d(field, grid.dz, 2; boundary_order)
    else
        return derivative_1d(field, grid.dz, 3; boundary_order)
    end
end

"""
    derivative_at_faces(field, grid, dim)

Compute derivative at staggered faces for specified dimension.
"""
function derivative_at_faces(field::AbstractArray{T,N}, grid::StaggeredGrid, dim::Int) where {T,N}
    input_size = size(field)
    output_size = Tuple(i == dim ? input_size[i] + 1 : input_size[i] for i in 1:N)
    result = zeros(T, output_size)
    
    h = (dim == 1) ? grid.dx : (dim == 2) ? grid.dy : grid.dz
    n = input_size[dim]
    
    for idx in CartesianIndices(result)
        i = idx[dim]
        
        if i == 1
            # First face - forward difference from cell centers
            base_idx = CartesianIndex(Tuple(idx[j] == i && j == dim ? 1 : idx[j] for j in 1:N))
            idx_p1 = CartesianIndex(Tuple(base_idx[j] == 1 && j == dim ? 2 : base_idx[j] for j in 1:N))
            idx_p2 = CartesianIndex(Tuple(base_idx[j] == 1 && j == dim ? 3 : base_idx[j] for j in 1:N))
            result[idx] = (-3*field[base_idx] + 4*field[idx_p1] - field[idx_p2]) / (2*h)
        elseif i == n + 1
            # Last face - backward difference from cell centers
            base_idx = CartesianIndex(Tuple(idx[j] == i && j == dim ? n : idx[j] for j in 1:N))
            idx_m1 = CartesianIndex(Tuple(base_idx[j] == n && j == dim ? n-1 : base_idx[j] for j in 1:N))
            idx_m2 = CartesianIndex(Tuple(base_idx[j] == n && j == dim ? n-2 : base_idx[j] for j in 1:N))
            result[idx] = (3*field[base_idx] - 4*field[idx_m1] + field[idx_m2]) / (2*h)
        else
            # Interior faces - simple difference
            curr_idx = CartesianIndex(Tuple(idx[j] == i && j == dim ? i : idx[j] for j in 1:N))
            prev_idx = CartesianIndex(Tuple(idx[j] == i && j == dim ? i-1 : idx[j] for j in 1:N))
            result[idx] = (field[curr_idx] - field[prev_idx]) / h
        end
    end
    
    return result
end

"""
    ddx_at_faces(field, grid)
    ddy_at_faces(field, grid)

Compute derivatives at face locations for staggered grid.
"""
ddx_at_faces(field::AbstractArray, grid::StaggeredGrid) = derivative_at_faces(field, grid, 1)
ddy_at_faces(field::AbstractArray, grid::StaggeredGrid) = derivative_at_faces(field, grid, 2)

# =============================================================================
# Second Derivative Operators
# =============================================================================

"""
    second_derivative_1d(field, h, dim)

Generic 2nd derivative computation along specified dimension.
"""
# Simplified implementations to avoid LLVM compilation issues
function second_derivative_1d(field::Matrix{T}, h::T, dim::Int) where {T}
    result = zeros(T, size(field))
    nx, ny = size(field)
    
    if dim == 1  # d/dx direction
        @inbounds for j in 1:ny, i in 2:nx-1
            result[i,j] = (field[i+1,j] - 2*field[i,j] + field[i-1,j]) / h^2
        end
        # Neumann BC: copy interior values to boundaries
        @inbounds for j in 1:ny
            result[1,j] = result[2,j]
            result[nx,j] = result[nx-1,j]
        end
    elseif dim == 2  # d/dy direction  
        @inbounds for j in 2:ny-1, i in 1:nx
            result[i,j] = (field[i,j+1] - 2*field[i,j] + field[i,j-1]) / h^2
        end
        # Neumann BC: copy interior values to boundaries
        @inbounds for i in 1:nx
            result[i,1] = result[i,2]
            result[i,ny] = result[i,ny-1]
        end
    end
    
    return result
end

function second_derivative_1d(field::Array{T,3}, h::T, dim::Int) where {T}
    result = zeros(T, size(field))
    nx, ny, nz = size(field)
    
    if dim == 1  # d/dx direction
        @inbounds for k in 1:nz, j in 1:ny, i in 2:nx-1
            result[i,j,k] = (field[i+1,j,k] - 2*field[i,j,k] + field[i-1,j,k]) / h^2
        end
        # Neumann BC
        @inbounds for k in 1:nz, j in 1:ny
            result[1,j,k] = result[2,j,k]
            result[nx,j,k] = result[nx-1,j,k]
        end
    elseif dim == 2  # d/dy direction
        @inbounds for k in 1:nz, j in 2:ny-1, i in 1:nx
            result[i,j,k] = (field[i,j+1,k] - 2*field[i,j,k] + field[i,j-1,k]) / h^2
        end
        # Neumann BC
        @inbounds for k in 1:nz, i in 1:nx
            result[i,1,k] = result[i,2,k]
            result[i,ny,k] = result[i,ny-1,k]
        end
    elseif dim == 3  # d/dz direction
        @inbounds for k in 2:nz-1, j in 1:ny, i in 1:nx
            result[i,j,k] = (field[i,j,k+1] - 2*field[i,j,k] + field[i,j,k-1]) / h^2
        end
        # Neumann BC
        @inbounds for j in 1:ny, i in 1:nx
            result[i,j,1] = result[i,j,2]
            result[i,j,nz] = result[i,j,nz-1]
        end
    end
    
    return result
end

"""
    d2dx2(field, grid)
    d2dy2(field, grid)
    d2dz2(field, grid)

Compute second derivatives with 2nd order accuracy.
"""
d2dx2(field::AbstractArray, grid::StaggeredGrid) = second_derivative_1d(field, grid.dx, 1)
d2dy2(field::AbstractArray, grid::StaggeredGrid) = second_derivative_1d(field, grid.dy, 2)
d2dz2(field::AbstractArray, grid::StaggeredGrid) = second_derivative_1d(field, grid.dz, 3)

# =============================================================================
# Vector Operators
# =============================================================================

"""
    div(u, v, grid)
    div(u, v, w, grid)

Compute divergence ∇·u with 2nd order accuracy on staggered grid.
"""
function div(u::Matrix{T}, v::Matrix{T}, grid::StaggeredGrid) where T
    # For 2D grids, check if this is XY or XZ plane
    if grid.grid_type == TwoDimensional
        # XZ plane: use nx, nz dimensions
        nx, nz = grid.nx, grid.nz
        result = zeros(T, nx, nz)
        dx, dz = grid.dx, grid.dz
        
        @inbounds for j = 1:nz, i = 1:nx
            result[i, j] = (u[i+1, j] - u[i, j]) / dx + (v[i, j+1] - v[i, j]) / dz
        end
    else
        # XY plane: use nx, ny dimensions
        nx, ny = grid.nx, grid.ny
        result = zeros(T, nx, ny)
        dx, dy = grid.dx, grid.dy
        
        @inbounds for j = 1:ny, i = 1:nx
            result[i, j] = (u[i+1, j] - u[i, j]) / dx + (v[i, j+1] - v[i, j]) / dy
        end
    end
    
    return result
end

function div(u::Array{T,3}, v::Array{T,3}, w::Array{T,3}, grid::StaggeredGrid) where T
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    result = zeros(T, nx, ny, nz)
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    
    @inbounds for k = 1:nz, j = 1:ny, i = 1:nx
        result[i, j, k] = (u[i+1, j, k] - u[i, j, k]) / dx + 
                         (v[i, j+1, k] - v[i, j, k]) / dy + 
                         (w[i, j, k+1] - w[i, j, k]) / dz
    end
    
    return result
end

"""
    grad(p, grid)

Compute gradient ∇p at staggered face locations.
"""
grad(p::Matrix, grid::StaggeredGrid) = (ddx_at_faces(p, grid), ddy_at_faces(p, grid))

function grad(p::Array{T,3}, grid::StaggeredGrid) where T
    dpdx = derivative_at_faces(p, grid, 1)
    dpdy = derivative_at_faces(p, grid, 2)
    dpdz = derivative_at_faces(p, grid, 3)
    return dpdx, dpdy, dpdz
end

"""
    laplacian(field, grid)

Compute Laplacian ∇²field.
"""
function laplacian(field::Matrix{T}, grid::StaggeredGrid) where T
    # Optimized 2D laplacian to avoid function call overhead
    result = zeros(T, size(field))
    nx, ny = size(field)
    dx, dy = grid.dx, grid.dy
    inv_dx2, inv_dy2 = 1.0/dx^2, 1.0/dy^2
    
    # Interior points
    @inbounds for j in 2:ny-1, i in 2:nx-1
        result[i,j] = inv_dx2 * (field[i+1,j] - 2*field[i,j] + field[i-1,j]) +
                      inv_dy2 * (field[i,j+1] - 2*field[i,j] + field[i,j-1])
    end
    
    # Boundary conditions: Neumann (zero gradient)
    # Left and right boundaries
    @inbounds for j in 2:ny-1
        result[1,j] = inv_dx2 * (field[2,j] - field[1,j]) +
                      inv_dy2 * (field[1,j+1] - 2*field[1,j] + field[1,j-1])
        result[nx,j] = inv_dx2 * (field[nx-1,j] - field[nx,j]) +
                       inv_dy2 * (field[nx,j+1] - 2*field[nx,j] + field[nx,j-1])
    end
    
    # Top and bottom boundaries
    @inbounds for i in 2:nx-1
        result[i,1] = inv_dx2 * (field[i+1,1] - 2*field[i,1] + field[i-1,1]) +
                      inv_dy2 * (field[i,2] - field[i,1])
        result[i,ny] = inv_dx2 * (field[i+1,ny] - 2*field[i,ny] + field[i-1,ny]) +
                       inv_dy2 * (field[i,ny-1] - field[i,ny])
    end
    
    # Corner points
    result[1,1] = inv_dx2 * (field[2,1] - field[1,1]) + inv_dy2 * (field[1,2] - field[1,1])
    result[nx,1] = inv_dx2 * (field[nx-1,1] - field[nx,1]) + inv_dy2 * (field[nx,2] - field[nx,1])
    result[1,ny] = inv_dx2 * (field[2,ny] - field[1,ny]) + inv_dy2 * (field[1,ny-1] - field[1,ny])
    result[nx,ny] = inv_dx2 * (field[nx-1,ny] - field[nx,ny]) + inv_dy2 * (field[nx,ny-1] - field[nx,ny])
    
    return result
end

function laplacian(field::Array{T,3}, grid::StaggeredGrid) where T
    d2dx2(field, grid) .+ d2dy2(field, grid) .+ d2dz2(field, grid)
end

# =============================================================================
# Interpolation Operators
# =============================================================================

"""
    interpolate_to_cell_centers(u, v, grid)
    interpolate_to_cell_centers(u, v, w, grid)

Interpolate velocity components from faces to cell centers.
"""
function interpolate_to_cell_centers(u::Matrix{T}, v::Matrix{T}, grid::StaggeredGrid) where T
    nx, ny = grid.nx, grid.ny
    u_cc = zeros(T, nx, ny)
    v_cc = zeros(T, nx, ny)
    
    @inbounds for j = 1:ny, i = 1:nx
        u_cc[i, j] = 0.5 * (u[i, j] + u[i+1, j])
        v_cc[i, j] = 0.5 * (v[i, j] + v[i, j+1])
    end
    
    return u_cc, v_cc
end

# Specialized version for 2D XZ-plane (u, w components)
function interpolate_to_cell_centers_xz(u::Matrix{T}, w::Matrix{T}, grid::StaggeredGrid) where T
    nx, nz = grid.nx, grid.nz
    u_cc = zeros(T, nx, nz)
    w_cc = zeros(T, nx, nz)
    
    # Interpolate u from x-faces to cell centers
    @inbounds for j = 1:nz, i = 1:nx
        u_cc[i, j] = 0.5 * (u[i, j] + u[i+1, j])
    end
    
    # Interpolate w from z-faces to cell centers  
    @inbounds for j = 1:nz, i = 1:nx
        w_cc[i, j] = 0.5 * (w[i, j] + w[i, j+1])
    end
    
    return u_cc, w_cc
end

function interpolate_to_cell_centers(u::Array{T,3}, v::Array{T,3}, w::Array{T,3}, grid::StaggeredGrid) where T
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

# Convenience functions for individual components - optimized versions
function interpolate_u_to_cell_center(u::Matrix{T}, grid::StaggeredGrid) where T
    if grid.grid_type == TwoDimensional
        nx, nz = grid.nx, grid.nz
        u_cc = zeros(T, nx, nz)
        @inbounds for j = 1:nz, i = 1:nx
            u_cc[i, j] = 0.5 * (u[i, j] + u[i+1, j])
        end
        return u_cc
    else
        nx, ny = grid.nx, grid.ny
        u_cc = zeros(T, nx, ny)
        @inbounds for j = 1:ny, i = 1:nx
            u_cc[i, j] = 0.5 * (u[i, j] + u[i+1, j])
        end
        return u_cc
    end
end

function interpolate_v_to_cell_center(v::Matrix{T}, grid::StaggeredGrid) where T
    if grid.grid_type == TwoDimensional
        nx, nz = grid.nx, grid.nz
        v_cc = zeros(T, nx, nz)
        @inbounds for j = 1:nz, i = 1:nx
            v_cc[i, j] = 0.5 * (v[i, j] + v[i, j+1])
        end
        return v_cc
    else
        nx, ny = grid.nx, grid.ny
        v_cc = zeros(T, nx, ny)
        @inbounds for j = 1:ny, i = 1:nx
            v_cc[i, j] = 0.5 * (v[i, j] + v[i, j+1])
        end
        return v_cc
    end
end

function interpolate_u_to_cell_center(u::Array{T,3}, grid::StaggeredGrid) where T
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    u_cc = zeros(T, nx, ny, nz)
    @inbounds for k = 1:nz, j = 1:ny, i = 1:nx
        u_cc[i, j, k] = 0.5 * (u[i, j, k] + u[i+1, j, k])
    end
    return u_cc
end

function interpolate_v_to_cell_center(v::Array{T,3}, grid::StaggeredGrid) where T
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    v_cc = zeros(T, nx, ny, nz)
    @inbounds for k = 1:nz, j = 1:ny, i = 1:nx
        v_cc[i, j, k] = 0.5 * (v[i, j, k] + v[i, j+1, k])
    end
    return v_cc
end

# =============================================================================
# Verification Functions
# =============================================================================

"""
    verify_operator_accuracy(grid)

Test differential operators against analytical functions.
"""
function verify_operator_accuracy(grid::StaggeredGrid)
    println("Testing differential operator accuracy...")
    
    nx, ny = grid.nx, grid.ny
    
    # Test function: f(x,y) = sin(πx/Lx) * cos(πy/Ly)
    f = [sin(π * grid.x[i] / grid.Lx) * cos(π * grid.y[j] / grid.Ly) for i = 1:nx, j = 1:ny]
    
    # Analytical derivatives
    dfdx_analytical = [(π / grid.Lx) * cos(π * grid.x[i] / grid.Lx) * cos(π * grid.y[j] / grid.Ly) for i = 1:nx, j = 1:ny]
    dfdy_analytical = [-(π / grid.Ly) * sin(π * grid.x[i] / grid.Lx) * sin(π * grid.y[j] / grid.Ly) for i = 1:nx, j = 1:ny]
    d2fdx2_analytical = [-(π / grid.Lx)^2 * sin(π * grid.x[i] / grid.Lx) * cos(π * grid.y[j] / grid.Ly) for i = 1:nx, j = 1:ny]
    
    # Numerical derivatives
    dfdx_numerical = ddx(f, grid)
    dfdy_numerical = ddy(f, grid)
    d2fdx2_numerical = d2dx2(f, grid)
    
    # Compute errors (exclude boundaries)
    interior = 2:nx-1, 2:ny-1
    error_ddx = maximum(abs.(dfdx_numerical[interior...] - dfdx_analytical[interior...]))
    error_ddy = maximum(abs.(dfdy_numerical[interior...] - dfdy_analytical[interior...]))
    error_d2dx2 = maximum(abs.(d2fdx2_numerical[interior...] - d2fdx2_analytical[interior...]))
    
    println("  ddx error: $(error_ddx)")
    println("  ddy error: $(error_ddy)")
    println("  d2dx2 error: $(error_d2dx2)")
    println("  Expected error scale: $((grid.dx^2 + grid.dy^2))")
    
    return error_ddx, error_ddy, error_d2dx2
end

"""
    check_staggered_grid_consistency(u, v, p, grid)

Verify array sizes for staggered grid.
"""
function check_staggered_grid_consistency(u::Matrix, v::Matrix, p::Matrix, grid::StaggeredGrid)
    println("Checking staggered grid array consistency...")
    
    expected_sizes = ((grid.nx + 1, grid.ny), (grid.nx, grid.ny + 1), (grid.nx, grid.ny))
    actual_sizes = (size(u), size(v), size(p))
    names = ("u", "v", "p")
    
    all_ok = true
    for (name, actual, expected) in zip(names, actual_sizes, expected_sizes)
        ok = actual == expected
        all_ok &= ok
        println("  $name size: $actual, expected: $expected, OK: $ok")
    end
    
    return all_ok
end
