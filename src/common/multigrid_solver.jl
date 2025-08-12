using GeometricMultigrid

struct MultigridPoissonSolver
    mg_solver::GeometricMultigrid.Multigrid
    levels::Int
    max_iterations::Int
    tolerance::Float64
    smoother::Symbol  # :jacobi, :gauss_seidel, :sor
    cycle_type::Symbol  # :V, :W, :F
end

function MultigridPoissonSolver(grid::StaggeredGrid; 
                               levels::Int=4,
                               max_iterations::Int=100,
                               tolerance::Float64=1e-10,
                               smoother::Symbol=:gauss_seidel,
                               cycle_type::Symbol=:V)
    
    # Set up multigrid hierarchy based on grid
    if grid.grid_type == TwoDimensional
        mg_solver = setup_multigrid_2d(grid, levels, smoother, cycle_type)
    elseif grid.grid_type == ThreeDimensional
        mg_solver = setup_multigrid_3d(grid, levels, smoother, cycle_type)
    else
        error("Multigrid not implemented for grid type: $(grid.grid_type)")
    end
    
    MultigridPoissonSolver(mg_solver, levels, max_iterations, tolerance, smoother, cycle_type)
end

function setup_multigrid_2d(grid::StaggeredGrid, levels::Int, smoother::Symbol, cycle_type::Symbol)
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    
    # Create Poisson operator for 2D with uniform grid spacing
    # This creates the discrete Laplacian operator ∇²φ = rhs
    function laplacian_2d(x::AbstractVector)
        phi = reshape(x, nx, ny)
        result = zeros(nx, ny)
        
        for j = 2:ny-1, i = 2:nx-1
            result[i,j] = (phi[i+1,j] - 2*phi[i,j] + phi[i-1,j]) / dx^2 + 
                         (phi[i,j+1] - 2*phi[i,j] + phi[i,j-1]) / dy^2
        end
        
        # Apply homogeneous Neumann boundary conditions
        # ∂φ/∂n = 0 on boundaries
        result[1, :] = result[2, :] - result[1, :] # left
        result[nx, :] = result[nx-1, :] - result[nx, :] # right  
        result[:, 1] = result[:, 2] - result[:, 1] # bottom
        result[:, ny] = result[:, ny-1] - result[:, ny] # top
        
        return vec(result)
    end
    
    # Select smoother
    if smoother == :jacobi
        smoother_func = GeometricMultigrid.Jacobi()
    elseif smoother == :gauss_seidel
        smoother_func = GeometricMultigrid.GaussSeidel()
    else
        smoother_func = GeometricMultigrid.GaussSeidel() # default
    end
    
    # Create restriction and prolongation operators
    restrict_op = GeometricMultigrid.LinearRestriction()
    prolong_op = GeometricMultigrid.BilinearProlongation()
    
    # Set up multigrid
    mg = GeometricMultigrid.Multigrid(
        operator = laplacian_2d,
        levels = levels,
        smoother = smoother_func,
        restriction = restrict_op,
        prolongation = prolong_op,
        coarse_solver = GeometricMultigrid.DirectSolver(),
        cycle = (cycle_type == :V) ? GeometricMultigrid.VCycle() : GeometricMultigrid.WCycle()
    )
    
    return mg
end

function setup_multigrid_3d(grid::StaggeredGrid, levels::Int, smoother::Symbol, cycle_type::Symbol)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    
    # Create Poisson operator for 3D
    function laplacian_3d(x::AbstractVector)
        phi = reshape(x, nx, ny, nz)
        result = zeros(nx, ny, nz)
        
        for k = 2:nz-1, j = 2:ny-1, i = 2:nx-1
            result[i,j,k] = (phi[i+1,j,k] - 2*phi[i,j,k] + phi[i-1,j,k]) / dx^2 + 
                           (phi[i,j+1,k] - 2*phi[i,j,k] + phi[i,j-1,k]) / dy^2 +
                           (phi[i,j,k+1] - 2*phi[i,j,k] + phi[i,j,k-1]) / dz^2
        end
        
        # Apply homogeneous Neumann boundary conditions
        result[1, :, :] = result[2, :, :] - result[1, :, :]
        result[nx, :, :] = result[nx-1, :, :] - result[nx, :, :]
        result[:, 1, :] = result[:, 2, :] - result[:, 1, :]
        result[:, ny, :] = result[:, ny-1, :] - result[:, ny, :]
        result[:, :, 1] = result[:, :, 2] - result[:, :, 1]
        result[:, :, nz] = result[:, :, nz-1] - result[:, :, nz]
        
        return vec(result)
    end
    
    # Select smoother
    if smoother == :jacobi
        smoother_func = GeometricMultigrid.Jacobi()
    elseif smoother == :gauss_seidel
        smoother_func = GeometricMultigrid.GaussSeidel()
    else
        smoother_func = GeometricMultigrid.GaussSeidel()
    end
    
    # Create restriction and prolongation operators for 3D
    restrict_op = GeometricMultigrid.LinearRestriction()
    prolong_op = GeometricMultigrid.TrilinearProlongation()
    
    # Set up multigrid
    mg = GeometricMultigrid.Multigrid(
        operator = laplacian_3d,
        levels = levels,
        smoother = smoother_func,
        restriction = restrict_op,
        prolongation = prolong_op,
        coarse_solver = GeometricMultigrid.DirectSolver(),
        cycle = (cycle_type == :V) ? GeometricMultigrid.VCycle() : GeometricMultigrid.WCycle()
    )
    
    return mg
end

function solve_poisson!(solver::MultigridPoissonSolver, phi::Array, rhs::Array, 
                       grid::StaggeredGrid, bc::BoundaryConditions)
    
    if grid.grid_type == TwoDimensional
        solve_poisson_2d_mg!(solver, phi, rhs, grid, bc)
    elseif grid.grid_type == ThreeDimensional
        solve_poisson_3d_mg!(solver, phi, rhs, grid, bc)
    else
        error("Unsupported grid type for multigrid: $(grid.grid_type)")
    end
end

function solve_poisson_2d_mg!(solver::MultigridPoissonSolver, phi::Matrix, rhs::Matrix,
                             grid::StaggeredGrid, bc::BoundaryConditions)
    
    # Apply boundary conditions to right-hand side
    rhs_bc = copy(rhs)
    apply_poisson_rhs_bc_2d!(rhs_bc, bc, grid)
    
    # Convert to vector format for GeometricMultigrid.jl
    phi_vec = vec(phi)
    rhs_vec = vec(rhs_bc)
    
    # Solve using GeometricMultigrid.jl
    solution = GeometricMultigrid.solve(solver.mg_solver, rhs_vec, phi_vec;
                                       maxiter=solver.max_iterations,
                                       tolerance=solver.tolerance)
    
    # Reshape back to matrix and update phi
    phi .= reshape(solution, size(phi))
    
    # Apply boundary conditions to solution
    apply_poisson_bc_2d!(phi, bc, grid)
end

function solve_poisson_3d_mg!(solver::MultigridPoissonSolver, phi::Array{T,3}, rhs::Array{T,3},
                             grid::StaggeredGrid, bc::BoundaryConditions) where T
    
    # Apply boundary conditions to right-hand side
    rhs_bc = copy(rhs)
    apply_poisson_rhs_bc_3d!(rhs_bc, bc, grid)
    
    # Convert to vector format for GeometricMultigrid.jl
    phi_vec = vec(phi)
    rhs_vec = vec(rhs_bc)
    
    # Solve using GeometricMultigrid.jl
    solution = GeometricMultigrid.solve(solver.mg_solver, rhs_vec, phi_vec;
                                       maxiter=solver.max_iterations,
                                       tolerance=solver.tolerance)
    
    # Reshape back to 3D array and update phi
    phi .= reshape(solution, size(phi))
    
    # Apply boundary conditions to solution
    apply_poisson_bc_3d!(phi, bc, grid)
end

# Helper functions for boundary condition application
function apply_poisson_rhs_bc_2d!(rhs::Matrix, bc::BoundaryConditions, grid::StaggeredGrid)
    # Modify RHS to incorporate boundary conditions
    # For homogeneous Neumann BC (∂φ/∂n = 0), no modification to RHS needed
    # This function can be extended for other BC types
    nx, ny = grid.nx, grid.ny
    
    # For Neumann BC, ensure compatibility condition: ∫rhs dV = 0
    rhs_mean = sum(rhs) / (nx * ny)
    rhs .-= rhs_mean
end

function apply_poisson_rhs_bc_3d!(rhs::Array{T,3}, bc::BoundaryConditions, grid::StaggeredGrid) where T
    # Similar to 2D case
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # Ensure compatibility for Neumann BC
    rhs_mean = sum(rhs) / (nx * ny * nz)
    rhs .-= rhs_mean
end

# Utility functions for direct solving (fallback when GeometricMultigrid.jl is not available)
function gauss_seidel_2d!(phi::Matrix{T}, rhs::Matrix{T}, 
                         dx::T, dy::T, nx::Int, ny::Int) where T
    factor = 1.0 / (2.0/dx^2 + 2.0/dy^2)
    
    for j = 2:ny-1, i = 2:nx-1
        phi[i,j] = factor * (
            (phi[i+1,j] + phi[i-1,j]) / dx^2 + 
            (phi[i,j+1] + phi[i,j-1]) / dy^2 - 
            rhs[i,j]
        )
    end
end

function gauss_seidel_3d!(phi::Array{T,3}, rhs::Array{T,3},
                         dx::T, dy::T, dz::T, nx::Int, ny::Int, nz::Int) where T
    factor = 1.0 / (2.0/dx^2 + 2.0/dy^2 + 2.0/dz^2)
    
    for k = 2:nz-1, j = 2:ny-1, i = 2:nx-1
        phi[i,j,k] = factor * (
            (phi[i+1,j,k] + phi[i-1,j,k]) / dx^2 + 
            (phi[i,j+1,k] + phi[i,j-1,k]) / dy^2 +
            (phi[i,j,k+1] + phi[i,j,k-1]) / dz^2 - 
            rhs[i,j,k]
        )
    end
end

function compute_residual_2d(phi::Matrix{T}, rhs::Matrix{T}, 
                            dx::T, dy::T, nx::Int, ny::Int) where T
    residual = zeros(T, nx, ny)
    
    for j = 2:ny-1, i = 2:nx-1
        laplacian = (phi[i+1,j] - 2*phi[i,j] + phi[i-1,j]) / dx^2 + 
                   (phi[i,j+1] - 2*phi[i,j] + phi[i,j-1]) / dy^2
        residual[i,j] = rhs[i,j] - laplacian
    end
    
    return residual
end

function compute_residual_3d(phi::Array{T,3}, rhs::Array{T,3},
                            dx::T, dy::T, dz::T, nx::Int, ny::Int, nz::Int) where T
    residual = zeros(T, nx, ny, nz)
    
    for k = 2:nz-1, j = 2:ny-1, i = 2:nx-1
        laplacian = (phi[i+1,j,k] - 2*phi[i,j,k] + phi[i-1,j,k]) / dx^2 + 
                   (phi[i,j+1,k] - 2*phi[i,j,k] + phi[i,j-1,k]) / dy^2 +
                   (phi[i,j,k+1] - 2*phi[i,j,k] + phi[i,j,k-1]) / dz^2
        residual[i,j,k] = rhs[i,j,k] - laplacian
    end
    
    return residual
end

# Alternative solver when GeometricMultigrid.jl is not available or as fallback
function solve_poisson_iterative!(phi::Array, rhs::Array, grid::StaggeredGrid, 
                                 bc::BoundaryConditions; 
                                 max_iter::Int=1000, tolerance::Float64=1e-10)
    """
    Fallback iterative Poisson solver using Gauss-Seidel iteration.
    This can be used when GeometricMultigrid.jl is not available.
    """
    
    if ndims(phi) == 2
        solve_poisson_2d_iterative!(phi, rhs, grid, bc, max_iter, tolerance)
    elseif ndims(phi) == 3
        solve_poisson_3d_iterative!(phi, rhs, grid, bc, max_iter, tolerance)
    else
        error("Unsupported dimensionality: $(ndims(phi))")
    end
end

function solve_poisson_2d_iterative!(phi::Matrix, rhs::Matrix, grid::StaggeredGrid, 
                                    bc::BoundaryConditions, max_iter::Int, tolerance::Float64)
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    
    for iter = 1:max_iter
        phi_old = copy(phi)
        
        # Gauss-Seidel iterations
        gauss_seidel_2d!(phi, rhs, dx, dy, nx, ny)
        
        # Apply boundary conditions
        apply_poisson_bc_2d!(phi, bc, grid)
        
        # Check convergence
        residual_norm = maximum(abs.(phi - phi_old))
        if residual_norm < tolerance
            println("Iterative Poisson solver converged in $iter iterations")
            break
        end
        
        if iter == max_iter
            @warn "Iterative Poisson solver did not converge after $max_iter iterations"
        end
    end
end

function solve_poisson_3d_iterative!(phi::Array{T,3}, rhs::Array{T,3}, grid::StaggeredGrid, 
                                    bc::BoundaryConditions, max_iter::Int, tolerance::Float64) where T
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    
    for iter = 1:max_iter
        phi_old = copy(phi)
        
        # Gauss-Seidel iterations
        gauss_seidel_3d!(phi, rhs, dx, dy, dz, nx, ny, nz)
        
        # Apply boundary conditions
        apply_poisson_bc_3d!(phi, bc, grid)
        
        # Check convergence
        residual_norm = maximum(abs.(phi - phi_old))
        if residual_norm < tolerance
            println("Iterative Poisson solver converged in $iter iterations")
            break
        end
        
        if iter == max_iter
            @warn "Iterative Poisson solver did not converge after $max_iter iterations"
        end
    end
end

function apply_poisson_bc_2d!(phi::Matrix, bc::BoundaryConditions, grid::StaggeredGrid)
    nx, ny = grid.nx, grid.ny
    
    # Default: homogeneous Neumann boundary conditions for pressure
    phi[1, :] .= phi[2, :]      # ∂φ/∂x = 0 at left
    phi[nx, :] .= phi[nx-1, :]  # ∂φ/∂x = 0 at right
    phi[:, 1] .= phi[:, 2]      # ∂φ/∂y = 0 at bottom
    phi[:, ny] .= phi[:, ny-1]  # ∂φ/∂y = 0 at top
end

function apply_poisson_bc_3d!(phi::Array{T,3}, bc::BoundaryConditions, grid::StaggeredGrid) where T
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # Homogeneous Neumann boundary conditions
    phi[1, :, :] .= phi[2, :, :]
    phi[nx, :, :] .= phi[nx-1, :, :]
    phi[:, 1, :] .= phi[:, 2, :]
    phi[:, ny, :] .= phi[:, ny-1, :]
    phi[:, :, 1] .= phi[:, :, 2]
    phi[:, :, nz] .= phi[:, :, nz-1]
end