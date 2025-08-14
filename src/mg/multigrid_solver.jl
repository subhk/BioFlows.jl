"""
High-Performance Multigrid Solver for Pressure Poisson Equation

This module provides both WaterLily.jl-style multigrid solver and fallback
to GeometricMultigrid.jl for maximum performance and compatibility.
"""

using GeometricMultigrid
include("waterlily_multigrid.jl")
include("staggered_multigrid.jl")
include("mpi_waterlily_multigrid.jl")

struct MultigridPoissonSolver
    mg_solver::Union{GeometricMultigrid.Multigrid, MultiLevelPoisson, StaggeredMultiLevelPoisson, MPIMultiLevelPoisson}
    solver_type::Symbol  # :staggered, :waterlily, :mpi_waterlily, or :geometric
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
                               cycle_type::Symbol=:V,
                               solver_type::Symbol=:auto,
                               pencil::Union{Nothing,Pencil}=nothing,
                               use_mpi::Bool=false,
                               mpi_comm::MPI.Comm=MPI.COMM_WORLD)
    
    # Auto-detect solver type based on MPI availability and grid type
    if solver_type == :auto
        # Check if MPI is initialized and has multiple processes
        mpi_available = false
        try
            if MPI.Initialized() && MPI.Comm_size(mpi_comm) > 1
                mpi_available = true
            end
        catch
            # MPI not available or not initialized
            mpi_available = false
        end
        
        # Override with explicit use_mpi flag
        if use_mpi
            mpi_available = true
        end
        
        if mpi_available || pencil !== nothing
            solver_type = :mpi_waterlily  # PREFER MPI version when available
        else
            solver_type = :staggered  # Use staggered grid-aware version for single process
        end
    end
    
    # Choose solver implementation
    if solver_type == :staggered
        # Use staggered grid-aware multigrid (RECOMMENDED for CFD)
        if grid.grid_type == TwoDimensional
            mg_solver = StaggeredMultiLevelPoisson(grid, levels; n_smooth=3, tol=tolerance)
        elseif grid.grid_type == ThreeDimensional
            # 3D staggered not yet implemented, fall back
            mg_solver = setup_multigrid_3d(grid, levels, smoother, cycle_type)
            solver_type = :geometric
        else
            error("Multigrid not implemented for grid type: $(grid.grid_type)")
        end
    elseif solver_type == :mpi_waterlily
        # Auto-create pencil if not provided
        if pencil === nothing
            if grid.grid_type == TwoDimensional
                # Create 2D pencil for XZ plane
                pencil = Pencil(mpi_comm, (grid.nx, grid.nz))
            elseif grid.grid_type == ThreeDimensional
                # Create 3D pencil
                pencil = Pencil(mpi_comm, (grid.nx, grid.ny, grid.nz))
            else
                error("Cannot create pencil for grid type: $(grid.grid_type)")
            end
        end
        
        if grid.grid_type == TwoDimensional
            mg_solver = MPIMultiLevelPoisson{Float64,2,typeof(pencil)}(
                pencil, grid.nx, grid.nz, grid.dx, grid.dz, levels;  # Use XZ plane for 2D
                n_smooth=3, tol=tolerance)
        elseif grid.grid_type == ThreeDimensional
            # 3D MPI WaterLily multigrid
            mg_solver = MPIMultiLevelPoisson{Float64,3,typeof(pencil)}(
                pencil, grid.nx, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz, levels;
                n_smooth=3, tol=tolerance)
        else
            error("Multigrid not implemented for grid type: $(grid.grid_type)")
        end
    elseif solver_type == :waterlily
        if grid.grid_type == TwoDimensional
            mg_solver = MultiLevelPoisson(grid.nx, grid.nz, grid.dx, grid.dz, levels;  # Use XZ plane for 2D
                                        n_smooth=3, tol=tolerance)
        elseif grid.grid_type == ThreeDimensional
            # For 3D, fall back to GeometricMultigrid.jl for now
            mg_solver = setup_multigrid_3d(grid, levels, smoother, cycle_type)
            solver_type = :geometric
        else
            error("Multigrid not implemented for grid type: $(grid.grid_type)")
        end
    else
        # Fallback to GeometricMultigrid.jl
        if grid.grid_type == TwoDimensional
            mg_solver = setup_multigrid_2d(grid, levels, smoother, cycle_type)
        elseif grid.grid_type == ThreeDimensional
            mg_solver = setup_multigrid_3d(grid, levels, smoother, cycle_type)
        else
            error("Multigrid not implemented for grid type: $(grid.grid_type)")
        end
        solver_type = :geometric
    end
    
    MultigridPoissonSolver(mg_solver, solver_type, levels, max_iterations, tolerance, smoother, cycle_type)
end

"""
    create_mpi_multigrid_solver(grid::StaggeredGrid; kwargs...)

Create an MPI-enabled multigrid solver, prioritizing MPIMultiLevelPoisson.

This is a convenience function that automatically:
1. Detects MPI availability
2. Creates appropriate pencil decomposition
3. Uses MPIMultiLevelPoisson as the default solver

# Arguments
- `grid::StaggeredGrid`: The computational grid
- `levels::Int=4`: Number of multigrid levels
- `use_mpi::Bool=true`: Force MPI usage (default: true)
- Other keyword arguments passed to MultigridPoissonSolver

# Returns
- `MultigridPoissonSolver`: Configured with MPIMultiLevelPoisson when MPI is available
"""
function create_mpi_multigrid_solver(grid::StaggeredGrid; 
                                   levels::Int=4,
                                   use_mpi::Bool=true,
                                   mpi_comm::MPI.Comm=MPI.COMM_WORLD,
                                   kwargs...)
    return MultigridPoissonSolver(grid; 
                                levels=levels,
                                solver_type=:mpi_waterlily,  # Explicitly request MPI solver
                                use_mpi=use_mpi,
                                mpi_comm=mpi_comm,
                                kwargs...)
end

"""
    show_solver_info(solver::MultigridPoissonSolver)

Display information about the current multigrid solver configuration.
"""
function show_solver_info(solver::MultigridPoissonSolver)
    println("Multigrid Poisson Solver Configuration:")
    println("  Solver Type: $(solver.solver_type)")
    println("  Levels: $(solver.levels)")
    println("  Max Iterations: $(solver.max_iterations)")
    println("  Tolerance: $(solver.tolerance)")
    println("  Smoother: $(solver.smoother)")
    println("  Cycle Type: $(solver.cycle_type)")
    
    if solver.solver_type == :mpi_waterlily
        println("  ✅ Using MPIMultiLevelPoisson (MPI-enabled, high performance)")
    elseif solver.solver_type == :staggered
        println("  ⚡ Using StaggeredMultiLevelPoisson (CFD-optimized, single process)")
    elseif solver.solver_type == :geometric
        println("  📐 Using GeometricMultigrid.jl (single process fallback)")
    end
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
        result[1,  :, :]  = result[2, :, :]    - result[1, :, :]
        result[nx, :, :]  = result[nx-1, :, :] - result[nx, :, :]
        result[:,  1, :]  = result[:, 2, :]    - result[:, 1, :]
        result[:, ny, :]  = result[:, ny-1, :] - result[:, ny, :]
        result[:,  :, 1]  = result[:, :, 2]    - result[:, :, 1]
        result[:,  :, nz] = result[:, :, nz-1] - result[:, :, nz]
        
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
    
    if solver.solver_type == :staggered
        # Use staggered grid-aware solver
        if grid.grid_type == TwoDimensional
            solve_poisson_2d_staggered!(solver, phi, rhs, grid, bc)
        else
            error("Staggered 3D solver not yet implemented")
        end
    elseif solver.solver_type == :mpi_waterlily
        # Use MPI WaterLily.jl-style solver  
        if grid.grid_type == TwoDimensional
            solve_poisson_2d_mpi_waterlily!(solver, phi, rhs, grid, bc)
        elseif grid.grid_type == ThreeDimensional
            solve_poisson_3d_mpi_waterlily!(solver, phi, rhs, grid, bc)
        else
            error("MPI WaterLily.jl-style solver not implemented for grid type: $(grid.grid_type)")
        end
    elseif solver.solver_type == :waterlily
        # Use single-node WaterLily.jl-style solver
        if grid.grid_type == TwoDimensional
            solve_poisson_2d_waterlily!(solver, phi, rhs, grid, bc)
        else
            error("WaterLily.jl-style 3D solver not yet implemented")
        end
    else
        # Use GeometricMultigrid.jl solver
        if grid.grid_type == TwoDimensional
            solve_poisson_2d_mg!(solver, phi, rhs, grid, bc)
        elseif grid.grid_type == ThreeDimensional
            solve_poisson_3d_mg!(solver, phi, rhs, grid, bc)
        else
            error("Unsupported grid type for multigrid: $(grid.grid_type)")
        end
    end
end

function solve_poisson_2d_staggered!(solver::MultigridPoissonSolver, phi::Matrix, rhs::Matrix,
                                    grid::StaggeredGrid, bc::BoundaryConditions)
    
    # Apply boundary conditions and ensure compatibility
    rhs_bc = copy(rhs)
    apply_poisson_rhs_bc_2d!(rhs_bc, bc, grid)
    
    # Solve using staggered grid-aware multigrid
    mg = solver.mg_solver::StaggeredMultiLevelPoisson
    residual, iterations = solve_staggered_poisson!(phi, rhs_bc, mg; max_iter=solver.max_iterations)
    
    # Apply boundary conditions to solution
    apply_poisson_bc_2d!(phi, bc, grid)
    
    println("Staggered multigrid converged in $iterations iterations, residual = $residual")
end

function solve_poisson_2d_mpi_waterlily!(solver::MultigridPoissonSolver, phi::PencilArray, rhs::PencilArray,
                                        grid::StaggeredGrid, bc::BoundaryConditions)
    
    # Apply boundary conditions and ensure compatibility
    rhs_bc = similar(rhs)
    copyto!(rhs_bc, rhs)
    apply_poisson_rhs_bc_2d_mpi!(rhs_bc, bc, grid)
    
    # Solve using MPI WaterLily.jl-style multigrid
    mg = solver.mg_solver::MPIMultiLevelPoisson
    residual, iterations = solve_poisson_mpi!(phi, rhs_bc, mg; max_iter=solver.max_iterations)
    
    # Apply boundary conditions to solution
    apply_poisson_bc_2d_mpi!(phi, bc, grid)
    
    if mg.rank == 0
        println("MPI WaterLily.jl multigrid converged in $iterations iterations, residual = $residual")
    end
end

function solve_poisson_3d_mpi_waterlily!(solver::MultigridPoissonSolver, phi::PencilArray, rhs::PencilArray,
                                        grid::StaggeredGrid, bc::BoundaryConditions)
    
    # Apply boundary conditions and ensure compatibility
    rhs_bc = similar(rhs)
    copyto!(rhs_bc, rhs)
    apply_poisson_rhs_bc_3d_mpi!(rhs_bc, bc, grid)
    
    # Solve using 3D MPI WaterLily.jl-style multigrid
    mg = solver.mg_solver::MPIMultiLevelPoisson{Float64,3}
    residual, iterations = solve_poisson_mpi!(phi, rhs_bc, mg; max_iter=solver.max_iterations)
    
    # Apply boundary conditions to solution
    apply_poisson_bc_3d_mpi!(phi, bc, grid)
    
    if mg.rank == 0
        println("3D MPI WaterLily.jl multigrid converged in $iterations iterations, residual = $residual")
    end
end

function solve_poisson_2d_waterlily!(solver::MultigridPoissonSolver, phi::Matrix, rhs::Matrix,
                                    grid::StaggeredGrid, bc::BoundaryConditions)
    
    # Apply boundary conditions and ensure compatibility
    rhs_bc = copy(rhs)
    apply_poisson_rhs_bc_2d!(rhs_bc, bc, grid)
    
    # Solve using WaterLily.jl-style multigrid
    mg = solver.mg_solver::MultiLevelPoisson
    residual, iterations = solve_poisson!(phi, rhs_bc, mg; max_iter=solver.max_iterations)
    
    # Apply boundary conditions to solution
    apply_poisson_bc_2d!(phi, bc, grid)
    
    println("WaterLily.jl multigrid converged in $iterations iterations, residual = $residual")
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

# MPI-specific boundary condition functions
function apply_poisson_rhs_bc_2d_mpi!(rhs::PencilArray{T,2}, bc::BoundaryConditions, grid::StaggeredGrid) where T
    # Modify RHS to incorporate boundary conditions for MPI case
    # For homogeneous Neumann BC (∂φ/∂n = 0), ensure compatibility condition: ∫rhs dV = 0
    
    # Compute global sum using MPI reduction
    local_sum = sum(rhs.data)
    global_sum = MPI.Allreduce(local_sum, MPI.SUM, rhs.decomp.comm)
    global_cells = grid.nx * grid.ny
    rhs_mean = global_sum / global_cells
    
    # Subtract mean from local data
    rhs.data .-= rhs_mean
end

function apply_poisson_bc_2d_mpi!(phi::PencilArray{T,2}, bc::BoundaryConditions, grid::StaggeredGrid) where T
    # Apply boundary conditions only on domain boundaries using the helper function
    apply_boundary_conditions_mpi!(phi, phi.decomp, grid.nx, grid.ny)
end

# 3D MPI boundary condition functions
function apply_poisson_rhs_bc_3d_mpi!(rhs::PencilArray{T,3}, bc::BoundaryConditions, grid::StaggeredGrid) where T
    @warn "Using placeholder 3D MPI RHS boundary conditions"
    # Placeholder - should implement proper 3D boundary condition application
end

function apply_poisson_bc_3d_mpi!(phi::PencilArray{T,3}, bc::BoundaryConditions, grid::StaggeredGrid) where T
    @warn "Using placeholder 3D MPI boundary conditions"
    # Apply boundary conditions only on domain boundaries using the helper function
    apply_boundary_conditions_mpi_3d!(phi, phi.decomp, grid.nx, grid.ny, grid.nz)
end