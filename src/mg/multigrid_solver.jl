"""
High-Performance Multigrid Solver for Pressure Poisson Equation

This module provides a pure Julia multigrid implementation without external dependencies.
"""

using LinearAlgebra
using SparseArrays

# Optional IterativeSolvers for Krylov methods
const HAS_ITERATIVE_SOLVERS = let ok = false
    try
        @eval using IterativeSolvers
        ok = true
    catch
        ok = false
    end
    ok
end

# Build sparse Laplacian (Neumann) for 2D XZ plane
function _build_laplacian_matrix_2d(grid::StaggeredGrid)
    nx, nz = grid.nx, grid.nz
    N = nx * nz
    dx2 = grid.dx^2
    dz2 = grid.dz^2
    rows = Int[]; cols = Int[]; vals = Float64[]
    lin = (i,j)-> i + (j-1)*nx
    for j=1:nz, i=1:nx
        k = lin(i,j)
        diag = 0.0
        # Left/Right
        if i>1
            push!(rows,k); push!(cols,lin(i-1,j)); push!(vals, 1.0/dx2)
            diag -= 1.0/dx2
        else
            # Neumann: φ(0)=φ(1) → contribution: -1/dx2
            diag -= 1.0/dx2
        end
        if i<nx
            push!(rows,k); push!(cols,lin(i+1,j)); push!(vals, 1.0/dx2)
            diag -= 1.0/dx2
        else
            diag -= 1.0/dx2
        end
        # Bottom/Top
        if j>1
            push!(rows,k); push!(cols,lin(i,j-1)); push!(vals, 1.0/dz2)
            diag -= 1.0/dz2
        else
            diag -= 1.0/dz2
        end
        if j<nz
            push!(rows,k); push!(cols,lin(i,j+1)); push!(vals, 1.0/dz2)
            diag -= 1.0/dz2
        else
            diag -= 1.0/dz2
        end
        push!(rows,k); push!(cols,k); push!(vals, -diag)
    end
    return sparse(rows, cols, vals, N, N)
end

# Build sparse Laplacian (Neumann) for 3D
function _build_laplacian_matrix_3d(grid::StaggeredGrid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    N = nx*ny*nz
    dx2 = grid.dx^2; dy2 = grid.dy^2; dz2 = grid.dz^2
    rows = Int[]; cols = Int[]; vals = Float64[]
    lin = (i,j,k)-> i + (j-1)*nx + (k-1)*nx*ny
    for k=1:nz, j=1:ny, i=1:nx
        idx = lin(i,j,k)
        diag = 0.0
        # X neighbors
        if i>1
            push!(rows,idx); push!(cols,lin(i-1,j,k)); push!(vals, 1.0/dx2); diag -= 1.0/dx2
        else
            diag -= 1.0/dx2
        end
        if i<nx
            push!(rows,idx); push!(cols,lin(i+1,j,k)); push!(vals, 1.0/dx2); diag -= 1.0/dx2
        else
            diag -= 1.0/dx2
        end
        # Y neighbors
        if j>1
            push!(rows,idx); push!(cols,lin(i,j-1,k)); push!(vals, 1.0/dy2); diag -= 1.0/dy2
        else
            diag -= 1.0/dy2
        end
        if j<ny
            push!(rows,idx); push!(cols,lin(i,j+1,k)); push!(vals, 1.0/dy2); diag -= 1.0/dy2
        else
            diag -= 1.0/dy2
        end
        # Z neighbors
        if k>1
            push!(rows,idx); push!(cols,lin(i,j,k-1)); push!(vals, 1.0/dz2); diag -= 1.0/dz2
        else
            diag -= 1.0/dz2
        end
        if k<nz
            push!(rows,idx); push!(cols,lin(i,j,k+1)); push!(vals, 1.0/dz2); diag -= 1.0/dz2
        else
            diag -= 1.0/dz2
        end
        push!(rows,idx); push!(cols,idx); push!(vals, -diag)
    end
    return sparse(rows, cols, vals, N, N)
end
# Pure Julia iterative solvers
struct GaussSeidelSolver
    max_iterations::Int
    tolerance::Float64
    omega::Float64  # Relaxation parameter
end

GaussSeidelSolver(max_iter::Int=1000, tol::Float64=1e-6) = GaussSeidelSolver(max_iter, tol, 1.0)

struct SimpleIterativeSolver
    max_iterations::Int
    tolerance::Float64
end

function solve!(solver::SimpleIterativeSolver, x::AbstractVector, b::AbstractVector, A)
    tol = solver.tolerance
    maxit = solver.max_iterations
    # Build a multiplication closure for A
    mulA = nothing
    if A === nothing
        # No operator: best fallback is to return RHS (caller should avoid this path)
        @warn "SimpleIterativeSolver: no operator provided; returning RHS directly"
        x .= b
        return x
    elseif A isa AbstractMatrix
        mulA = v -> A * v
    elseif A isa Function
        mulA = A
    else
        try
            mulA = v -> A * v
        catch
            @warn "SimpleIterativeSolver: unrecognized operator type; returning RHS"
            x .= b
            return x
        end
    end

    # Conjugate Gradient for SPD systems (e.g., Poisson)
    r = similar(x); r .= b .- mulA(x)
    p = copy(r)
    rsold = dot(r, r)
    if sqrt(rsold) < tol
        return x
    end
    Ap = similar(x)
    for k = 1:maxit
        Ap .= mulA(p)
        α = rsold / dot(p, Ap)
        @. x = x + α * p
        @. r = r - α * Ap
        rsnew = dot(r, r)
        if sqrt(rsnew) < tol
            break
        end
        β = rsnew / rsold
        @. p = r + β * p
        rsold = rsnew
    end
    return x
end

struct MultigridPoissonSolver
    mg_solver::Union{GaussSeidelSolver, SimpleIterativeSolver}  # Pure Julia solvers
    solver_type::Symbol  # :gauss_seidel or :simple
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
    # Allow environment overrides for quick tuning
    levels_env = try parse(Int, get(ENV, "BIOFLOWS_MG_LEVELS", string(levels))) catch; levels end
    iters_env = try parse(Int, get(ENV, "BIOFLOWS_MG_MAXITER", string(max_iterations))) catch; max_iterations end
    tol_env    = try parse(Float64, get(ENV, "BIOFLOWS_MG_TOL", string(tolerance))) catch; tolerance end
    smoother_env = try
        Symbol(lowercase(get(ENV, "BIOFLOWS_MG_SMOOTHER", string(smoother))))
    catch
        smoother
    end
    cycle_env = try
        Symbol(uppercase(get(ENV, "BIOFLOWS_MG_CYCLE", string(cycle_type))))
    catch
        cycle_type
    end
    levels = levels_env
    max_iterations = iters_env
    tolerance = tol_env
    smoother = smoother_env
    cycle_type = cycle_env
    
    # Use pure Julia solvers only
    if smoother == :gauss_seidel
        mg_solver = GaussSeidelSolver(max_iterations, tolerance)
        solver_type = :gauss_seidel
    elseif smoother == :staggered
        # Use staggered multigrid path (2D specialized); still carry a tag
        mg_solver = GaussSeidelSolver(max_iterations, tolerance)
        solver_type = :staggered
    else
        mg_solver = SimpleIterativeSolver(max_iterations, tolerance)
        solver_type = :simple
    end
    
    MultigridPoissonSolver(mg_solver, solver_type, levels, max_iterations, tolerance, smoother, cycle_type)
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
    
    if solver.solver_type == :gauss_seidel
        println("  Using custom Gauss-Seidel solver")
    elseif solver.solver_type == :simple
        println("  Using SimpleIterativeSolver (fallback)")
    end
end

# Custom 2D Gauss-Seidel solver for Poisson equation  
function gauss_seidel_2d!(phi::Matrix{T}, rhs::Matrix{T}, grid::StaggeredGrid{T}, 
                         max_iter::Int, tol::Float64, omega::Float64=1.0) where T
    nx, nz = size(phi)
    dx, dz = grid.dx, grid.dz
    dx2, dz2 = dx^2, dz^2
    factor = 1.0 / (2.0 * (1.0/dx2 + 1.0/dz2))
    
    residual = 0.0
    iter = 0
    for iter = 1:max_iter
        residual = 0.0
        
        for j = 2:nz-1, i = 2:nx-1
            old_phi = phi[i, j]
            new_phi = factor * ((phi[i+1,j] + phi[i-1,j]) / dx2 + 
                               (phi[i,j+1] + phi[i,j-1]) / dz2 - rhs[i,j])
            phi[i, j] = old_phi + omega * (new_phi - old_phi)
            residual += (phi[i,j] - old_phi)^2
        end
        
        # Apply boundary conditions
        phi[1, :] .= phi[2, :]      # ∂φ/∂x = 0 at left
        phi[nx, :] .= phi[nx-1, :]  # ∂φ/∂x = 0 at right
        phi[:, 1] .= phi[:, 2]      # ∂φ/∂z = 0 at bottom
        phi[:, nz] .= phi[:, nz-1]  # ∂φ/∂z = 0 at top
        
        if sqrt(residual) < tol
            break
        end
    end
    
    return phi, iter
end

# Custom 3D Gauss-Seidel solver for Poisson equation  
function gauss_seidel_3d!(phi::Array{T,3}, rhs::Array{T,3}, grid::StaggeredGrid{T}, 
                         max_iter::Int, tol::Float64, omega::Float64=1.0) where T
    nx, ny, nz = size(phi)
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    dx2, dy2, dz2 = dx^2, dy^2, dz^2
    factor = 1.0 / (2.0 * (1.0/dx2 + 1.0/dy2 + 1.0/dz2))
    
    residual = 0.0
    iter = 0
    for iter = 1:max_iter
        residual = 0.0
        
        for k = 2:nz-1, j = 2:ny-1, i = 2:nx-1
            old_phi = phi[i, j, k]
            new_phi = factor * ((phi[i+1,j,k] + phi[i-1,j,k]) / dx2 + 
                               (phi[i,j+1,k] + phi[i,j-1,k]) / dy2 +
                               (phi[i,j,k+1] + phi[i,j,k-1]) / dz2 - rhs[i,j,k])
            phi[i, j, k] = old_phi + omega * (new_phi - old_phi)
            residual += (phi[i,j,k] - old_phi)^2
        end
        
        # Apply boundary conditions
        phi[1, :, :]  .= phi[2, :, :]      # ∂φ/∂x = 0 at left
        phi[nx, :, :] .= phi[nx-1, :, :]   # ∂φ/∂x = 0 at right
        phi[:, 1, :]  .= phi[:, 2, :]      # ∂φ/∂y = 0 at bottom
        phi[:, ny, :] .= phi[:, ny-1, :]   # ∂φ/∂y = 0 at top
        phi[:, :, 1]  .= phi[:, :, 2]      # ∂φ/∂z = 0 at front
        phi[:, :, nz] .= phi[:, :, nz-1]   # ∂φ/∂z = 0 at back
        
        if sqrt(residual) < tol
            break
        end
    end
    
    return phi, iter
end

function solve_poisson!(solver::MultigridPoissonSolver, phi::Array, rhs::Array, 
                       grid::StaggeredGrid, bc::BoundaryConditions)
    
    # Use custom pure Julia solver
    if grid.grid_type == TwoDimensional
        solve_poisson_2d_custom!(solver, phi, rhs, grid, bc)
    elseif grid.grid_type == ThreeDimensional
        solve_poisson_3d_custom!(solver, phi, rhs, grid, bc)
    else
        error("Unsupported grid type for multigrid: $(grid.grid_type)")
    end
end

function solve_poisson_2d_custom!(solver::MultigridPoissonSolver, phi::Matrix, rhs::Matrix,
                                 grid::StaggeredGrid, bc::BoundaryConditions)
    
    # Apply boundary conditions to right-hand side
    rhs_bc = copy(rhs)
    apply_poisson_rhs_bc_2d!(rhs_bc, bc, grid)
    
    # Verbose flag
    verbose = lowercase(get(ENV, "BIOFLOWS_MG_VERBOSE", "")) in ("1","true","yes")
    # Residual norm helper
    function _residual_norm_2d(φ, b)
        nx, nz = size(φ)
        dx2 = grid.dx^2; dz2 = grid.dz^2
        s = 0.0
        for j=2:nz-1, i=2:nx-1
            lap = (φ[i+1,j]-2φ[i,j]+φ[i-1,j])/dx2 + (φ[i,j+1]-2φ[i,j]+φ[i,j-1])/dz2
            s += (lap - b[i,j])^2
        end
        return sqrt(s)
    end
    r0 = _residual_norm_2d(phi, rhs_bc)
    # Try IterativeSolvers if requested via env (and available)
    use_iter = lowercase(get(ENV, "BIOFLOWS_USE_ITERATIVE", "")) in ("1","true","yes") && HAS_ITERATIVE_SOLVERS
    if use_iter
        # Build sparse Laplacian with Neumann BC
        A = _build_laplacian_matrix_2d(grid)
        xvec = vec(copy(phi))
        bvec = vec(rhs_bc)
        try
            IterativeSolvers.cg!(xvec, A, bvec; maxiter=solver.max_iterations, atol=solver.tolerance, verbose=verbose)
            phi .= reshape(xvec, size(phi))
        catch e
            @warn "IterativeSolvers.cg! failed ($e); falling back to MG"
            use_iter = false
        end
    end
    # Use staggered multigrid if requested, otherwise Gauss-Seidel
    if !use_iter
    if solver.solver_type == :staggered
        mg = StaggeredMultiLevelPoisson(grid, solver.levels; tol=solver.tolerance)
        solve_staggered_poisson!(phi, rhs_bc, mg; max_iter=solver.max_iterations)
        if verbose
            r1 = _residual_norm_2d(phi, rhs_bc)
            @info "MG(2D) staggered res0=$(r0) res1=$(r1)"
        end
    elseif solver.mg_solver isa GaussSeidelSolver
        phi, iters = gauss_seidel_2d!(phi, rhs_bc, grid,
                                      solver.mg_solver.max_iterations,
                                      solver.mg_solver.tolerance,
                                      solver.mg_solver.omega)
        if verbose
            r1 = _residual_norm_2d(phi, rhs_bc)
            @info "MG(2D) GS iterations=$(iters) res0=$(r0) res1=$(r1)"
        end
    else
        # Use fallback solver
        phi_vec = vec(phi)
        rhs_vec = vec(rhs_bc)
        solution = solve!(solver.mg_solver, phi_vec, rhs_vec, nothing)
        phi .= reshape(solution, size(phi))
    end
    end
    
    # Apply boundary conditions to solution
    apply_poisson_bc_2d!(phi, bc, grid)
end

function solve_poisson_3d_custom!(solver::MultigridPoissonSolver, phi::Array{T,3}, rhs::Array{T,3},
                                 grid::StaggeredGrid, bc::BoundaryConditions) where T
    
    # Apply boundary conditions to right-hand side
    rhs_bc = copy(rhs)
    apply_poisson_rhs_bc_3d!(rhs_bc, bc, grid)
    
    # Verbose flag
    verbose = lowercase(get(ENV, "BIOFLOWS_MG_VERBOSE", "")) in ("1","true","yes")
    # Residual norm helper
    function _residual_norm_3d(φ, b)
        nx, ny, nz = size(φ)
        dx2 = grid.dx^2; dy2 = grid.dy^2; dz2 = grid.dz^2
        s = 0.0
        for k=2:nz-1, j=2:ny-1, i=2:nx-1
            lap = (φ[i+1,j,k]-2φ[i,j,k]+φ[i-1,j,k])/dx2 + (φ[i,j+1,k]-2φ[i,j,k]+φ[i,j-1,k])/dy2 + (φ[i,j,k+1]-2φ[i,j,k]+φ[i,j,k-1])/dz2
            s += (lap - b[i,j,k])^2
        end
        return sqrt(s)
    end
    r0 = _residual_norm_3d(phi, rhs_bc)
    # Try IterativeSolvers if requested via env (and available)
    use_iter = lowercase(get(ENV, "BIOFLOWS_USE_ITERATIVE", "")) in ("1","true","yes") && HAS_ITERATIVE_SOLVERS
    if use_iter
        A = _build_laplacian_matrix_3d(grid)
        xvec = vec(copy(phi))
        bvec = vec(rhs_bc)
        try
            IterativeSolvers.cg!(xvec, A, bvec; maxiter=solver.max_iterations, atol=solver.tolerance, verbose=verbose)
            phi .= reshape(xvec, size(phi))
        catch e
            @warn "IterativeSolvers.cg! (3D) failed ($e); falling back to MG"
            use_iter = false
        end
    end
    # Use staggered multigrid if requested, otherwise Gauss-Seidel
    if !use_iter
    if solver.smoother == :staggered
        mg = StaggeredMultiLevelPoisson3D(grid, solver.levels; tol=solver.tolerance)
        solve_staggered_poisson_3d!(phi, rhs_bc, mg; max_iter=solver.max_iterations)
        if verbose
            r1 = _residual_norm_3d(phi, rhs_bc)
            @info "MG(3D) staggered res0=$(r0) res1=$(r1)"
        end
    elseif solver.mg_solver isa GaussSeidelSolver
        phi, iters = gauss_seidel_3d!(phi, rhs_bc, grid,
                                      solver.mg_solver.max_iterations,
                                      solver.mg_solver.tolerance,
                                      solver.mg_solver.omega)
        if verbose
            r1 = _residual_norm_3d(phi, rhs_bc)
            @info "MG(3D) GS iterations=$(iters) res0=$(r0) res1=$(r1)"
        end
    else
        # Use fallback solver
        phi_vec = vec(phi)
        rhs_vec = vec(rhs_bc)
        solution = solve!(solver.mg_solver, phi_vec, rhs_vec, nothing)
        phi .= reshape(solution, size(phi))
    end
    end
    
    # Apply boundary conditions to solution
    apply_poisson_bc_3d!(phi, bc, grid)
end

# Helper functions for boundary condition application
function apply_poisson_rhs_bc_2d!(rhs::Matrix, bc::BoundaryConditions, grid::StaggeredGrid)
    # Modify RHS to incorporate boundary conditions (2D XZ plane)
    nx, nz = grid.nx, grid.nz
    rhs_mean = sum(rhs) / (nx * nz)
    rhs .-= rhs_mean
end

function apply_poisson_rhs_bc_3d!(rhs::Array{T,3}, bc::BoundaryConditions, grid::StaggeredGrid) where T
    # Similar to 2D case
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # Ensure compatibility for Neumann BC
    rhs_mean = sum(rhs) / (nx * ny * nz)
    rhs .-= rhs_mean
end

function apply_poisson_bc_2d!(phi::Matrix, bc::BoundaryConditions, grid::StaggeredGrid)
    # Homogeneous Neumann BC for XZ plane
    nx, nz = grid.nx, grid.nz
    phi[1, :]      .= phi[2, :]
    phi[nx, :]     .= phi[nx-1, :]
    phi[:, 1]      .= phi[:, 2]
    phi[:, nz]     .= phi[:, nz-1]
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
