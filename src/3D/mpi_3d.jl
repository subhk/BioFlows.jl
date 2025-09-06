using PencilArrays
using MPI

struct MPI3DDecomposition <: AbstractMPIDecomposition
    pencil::Pencil
    comm::MPI.Comm
    rank::Int
    size::Int
    
    # Grid decomposition
    nx_global::Int
    ny_global::Int
    nz_global::Int
    nx_local::Int
    ny_local::Int
    nz_local::Int
    
    # Local indices
    i_start::Int
    i_end::Int
    j_start::Int
    j_end::Int
    k_start::Int
    k_end::Int
    
    # 3D Cartesian topology dimensions
    dims::Vector{Int}
    coords::Vector{Int}
    
    # Neighbor ranks
    left_rank::Int
    right_rank::Int
    bottom_rank::Int
    top_rank::Int
    front_rank::Int
    back_rank::Int
    
    # Communication buffers
    send_buffers::Dict{Symbol, Array{Float64}}
    recv_buffers::Dict{Symbol, Array{Float64}}
end

function mpi_solve_step_3d!(solver, local_state_new, local_state_old, dt::Float64)
    # Minimal distributed projection step with halos and physical BCs
    decomp = solver.decomp
    grid = solver.local_grid

    # Apply BC and exchange halos on old state
    apply_physical_boundary_conditions_3d!(decomp, grid, local_state_old, solver.bc, local_state_old.t)
    exchange_ghost_cells_3d!(decomp, local_state_old.u)
    exchange_ghost_cells_3d!(decomp, local_state_old.v)
    exchange_ghost_cells_3d!(decomp, local_state_old.w)

    # Predictor: u* = u + dt(-adv + diff)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    adv_u = similar(local_state_old.u)
    adv_v = similar(local_state_old.v)
    adv_w = similar(local_state_old.w)
    diff_u = similar(local_state_old.u)
    diff_v = similar(local_state_old.v)
    diff_w = similar(local_state_old.w)

    advection_3d!(adv_u, adv_v, adv_w, local_state_old.u, local_state_old.v, local_state_old.w, grid)
    compute_diffusion_3d!(diff_u, diff_v, diff_w, local_state_old.u, local_state_old.v, local_state_old.w, solver.fluid, grid)

    @inbounds for k = 1:nz, j = 1:ny, i = 1:nx+1
        local_state_new.u[i, j, k] = local_state_old.u[i, j, k] + dt * (-adv_u[i, j, k] + diff_u[i, j, k])
    end
    @inbounds for k = 1:nz, j = 1:ny+1, i = 1:nx
        local_state_new.v[i, j, k] = local_state_old.v[i, j, k] + dt * (-adv_v[i, j, k] + diff_v[i, j, k])
    end
    @inbounds for k = 1:nz+1, j = 1:ny, i = 1:nx
        local_state_new.w[i, j, k] = local_state_old.w[i, j, k] + dt * (-adv_w[i, j, k] + diff_w[i, j, k])
    end

    # Apply BC to predictor and exchange halos
    apply_physical_boundary_conditions_3d!(decomp, grid, local_state_new, solver.bc, local_state_old.t + dt)
    exchange_ghost_cells_3d!(decomp, local_state_new.u)
    exchange_ghost_cells_3d!(decomp, local_state_new.v)
    exchange_ghost_cells_3d!(decomp, local_state_new.w)

    # Build Poisson RHS and solve
    solver.local_rhs_p = zeros(Float64, nx, ny, nz)
    divergence_3d!(solver.local_rhs_p, local_state_new.u, local_state_new.v, local_state_new.w, grid)
    solver.local_rhs_p .*= 1.0 / dt
    solve_poisson!(solver.multigrid_solver, solver.local_phi, solver.local_rhs_p, grid, solver.bc)

    # Correct velocities via ∇p
    dpdx = zeros(size(local_state_new.u))
    dpdy = zeros(size(local_state_new.v))
    dpdz = zeros(size(local_state_new.w))
    gradient_pressure_3d!(dpdx, dpdy, dpdz, solver.local_phi, grid)
    local_state_new.u .-= dt .* dpdx
    local_state_new.v .-= dt .* dpdy
    local_state_new.w .-= dt .* dpdz

    # Apply BC and exchange halos after correction
    apply_physical_boundary_conditions_3d!(decomp, grid, local_state_new, solver.bc, local_state_old.t + dt)
    exchange_ghost_cells_3d!(decomp, local_state_new.u)
    exchange_ghost_cells_3d!(decomp, local_state_new.v)
    exchange_ghost_cells_3d!(decomp, local_state_new.w)
    exchange_ghost_cells_3d!(decomp, local_state_new.p)

    # Update pressure and time
    ρ = solver.fluid.ρ isa ConstantDensity ? solver.fluid.ρ.ρ : error("Variable density not implemented")
    local_state_new.p .= local_state_old.p .+ solver.local_phi .* ρ
    local_state_new.t = local_state_old.t + dt
    local_state_new.step = local_state_old.step + 1
end

function MPI3DDecomposition(nx_global::Int, ny_global::Int, nz_global::Int, comm::MPI.Comm=MPI.COMM_WORLD)
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    # Create 3D Cartesian topology
    dims = [0, 0, 0]
    MPI.Dims_create!(size, dims)
    
    # Create Cartesian communicator (modern MPI.jl API)
    cart_comm = MPI.Cart_create(comm, dims; periodic=[false, false, false], reorder=true)
    coords = MPI.Cart_coords(cart_comm, rank)
    
    # Determine local domain sizes
    nx_local = nx_global ÷ dims[1]
    ny_local = ny_global ÷ dims[2]
    nz_local = nz_global ÷ dims[3]
    
    # Handle remainder cells
    if coords[1] < nx_global % dims[1]
        nx_local += 1
    end
    if coords[2] < ny_global % dims[2]
        ny_local += 1
    end
    if coords[3] < nz_global % dims[3]
        nz_local += 1
    end
    
    # Determine local index ranges
    i_start = coords[1] * (nx_global ÷ dims[1]) + 1 + min(coords[1], nx_global % dims[1])
    i_end = i_start + nx_local - 1
    j_start = coords[2] * (ny_global ÷ dims[2]) + 1 + min(coords[2], ny_global % dims[2])
    j_end = j_start + ny_local - 1
    k_start = coords[3] * (nz_global ÷ dims[3]) + 1 + min(coords[3], nz_global % dims[3])
    k_end = k_start + nz_local - 1
    
    # Find neighbor ranks
    left_rank, right_rank = MPI.Cart_shift(cart_comm, 0, 1)
    bottom_rank, top_rank = MPI.Cart_shift(cart_comm, 1, 1)
    front_rank, back_rank = MPI.Cart_shift(cart_comm, 2, 1)
    
    # Create PencilArrays topology and pencil
    topo = PencilArrays.Pencils.MPITopology(cart_comm, (dims[1], dims[2], dims[3]))
    pencil = PencilArrays.Pencils.Pencil(topo, (nx_global, ny_global, nz_global))
    
    # Initialize communication buffers for 6 faces
    send_buffers = Dict{Symbol, Array{Float64}}()
    recv_buffers = Dict{Symbol, Array{Float64}}()
    
    # X-direction faces (left/right)
    send_buffers[:left] = zeros(ny_local, nz_local)
    send_buffers[:right] = zeros(ny_local, nz_local)
    recv_buffers[:left] = zeros(ny_local, nz_local)
    recv_buffers[:right] = zeros(ny_local, nz_local)
    
    # Y-direction faces (bottom/top)
    send_buffers[:bottom] = zeros(nx_local, nz_local)
    send_buffers[:top] = zeros(nx_local, nz_local)
    recv_buffers[:bottom] = zeros(nx_local, nz_local)
    recv_buffers[:top] = zeros(nx_local, nz_local)
    
    # Z-direction faces (front/back)
    send_buffers[:front] = zeros(nx_local, ny_local)
    send_buffers[:back] = zeros(nx_local, ny_local)
    recv_buffers[:front] = zeros(nx_local, ny_local)
    recv_buffers[:back] = zeros(nx_local, ny_local)
    
    MPI3DDecomposition(pencil, cart_comm, rank, size,
                      nx_global, ny_global, nz_global, nx_local, ny_local, nz_local,
                      i_start, i_end, j_start, j_end, k_start, k_end,
                      dims, coords,
                      left_rank, right_rank, bottom_rank, top_rank, front_rank, back_rank,
                      send_buffers, recv_buffers)
end

function create_local_grid_3d(decomp::MPI3DDecomposition, Lx::Float64, Ly::Float64, Lz::Float64)
    dx = Lx / decomp.nx_global
    dy = Ly / decomp.ny_global
    dz = Lz / decomp.nz_global
    
    # Local domain bounds
    x_min = (decomp.i_start - 1) * dx
    y_min = (decomp.j_start - 1) * dy
    z_min = (decomp.k_start - 1) * dz
    
    local_Lx = decomp.nx_local * dx
    local_Ly = decomp.ny_local * dy
    local_Lz = decomp.nz_local * dz
    
    local_grid = StaggeredGrid3D(decomp.nx_local, decomp.ny_local, decomp.nz_local, 
                                local_Lx, local_Ly, local_Lz;
                                origin_x=x_min, origin_y=y_min, origin_z=z_min)
    
    return local_grid
end

"""
    apply_physical_boundary_conditions_3d!(decomp, grid, state, bc, t)

Apply boundary conditions only at physical domain boundaries for 3D MPI domains.
"""
function apply_physical_boundary_conditions_3d!(decomp::MPI3DDecomposition, grid::StaggeredGrid, 
                                               state::MPISolutionState3D, bc::BoundaryConditions, t::Float64)
    # For now, implement a simplified version that identifies physical boundaries
    # and applies boundary conditions accordingly.
    # This ensures MPI consistency with serial implementation.
    
    # Check if this process is at a physical boundary and apply BC accordingly
    if haskey(bc.conditions, (:x, :left)) && decomp.i_global_start == 1
        condition = bc.conditions[(:x, :left)]
        apply_boundary_conditions_x_left_3d!(state, condition, t)
    end
    
    if haskey(bc.conditions, (:x, :right)) && decomp.i_global_end == decomp.nx_global
        condition = bc.conditions[(:x, :right)]
        apply_boundary_conditions_x_right_3d!(state, condition, t)
    end
    
    if haskey(bc.conditions, (:y, :bottom)) && decomp.j_global_start == 1
        condition = bc.conditions[(:y, :bottom)]
        apply_boundary_conditions_y_bottom_3d!(state, condition, t)
    end
    
    if haskey(bc.conditions, (:y, :top)) && decomp.j_global_end == decomp.ny_global
        condition = bc.conditions[(:y, :top)]
        apply_boundary_conditions_y_top_3d!(state, condition, t)
    end
    
    if haskey(bc.conditions, (:z, :front)) && decomp.k_global_start == 1
        condition = bc.conditions[(:z, :front)]
        apply_boundary_conditions_z_front_3d!(state, condition, t)
    end
    
    if haskey(bc.conditions, (:z, :back)) && decomp.k_global_end == decomp.nz_global
        condition = bc.conditions[(:z, :back)]
        apply_boundary_conditions_z_back_3d!(state, condition, t)
    end
end

# Simplified boundary condition application functions for 3D MPI
function apply_boundary_conditions_x_left_3d!(state::MPISolutionState3D, condition::BoundaryCondition, t::Float64)
    if condition.type == NoSlip
        state.u[1, :, :] .= 0.0
        state.v[1, :, :] .= 0.0
        state.w[1, :, :] .= 0.0
    elseif condition.type == Inlet
        val = condition.value isa Function ? condition.value(t) : condition.value
        state.u[1, :, :] .= val
    end
end

function apply_boundary_conditions_x_right_3d!(state::MPISolutionState3D, condition::BoundaryCondition, t::Float64)
    nx = size(state.u, 1)
    if condition.type == NoSlip
        state.u[nx, :, :] .= 0.0
        state.v[nx-1, :, :] .= 0.0
        state.w[nx-1, :, :] .= 0.0
    elseif condition.type == Outlet
        state.u[nx, :, :] .= state.u[nx-1, :, :]
    end
end

function apply_boundary_conditions_y_bottom_3d!(state::MPISolutionState3D, condition::BoundaryCondition, t::Float64)
    if condition.type == NoSlip
        state.u[:, 1, :] .= 0.0
        state.v[:, 1, :] .= 0.0
        state.w[:, 1, :] .= 0.0
    end
end

function apply_boundary_conditions_y_top_3d!(state::MPISolutionState3D, condition::BoundaryCondition, t::Float64)
    ny = size(state.v, 2)
    if condition.type == NoSlip
        state.u[:, ny-1, :] .= 0.0
        state.v[:, ny, :] .= 0.0
        state.w[:, ny-1, :] .= 0.0
    end
end

function apply_boundary_conditions_z_front_3d!(state::MPISolutionState3D, condition::BoundaryCondition, t::Float64)
    if condition.type == NoSlip
        state.u[:, :, 1] .= 0.0
        state.v[:, :, 1] .= 0.0
        state.w[:, :, 1] .= 0.0
    end
end

function apply_boundary_conditions_z_back_3d!(state::MPISolutionState3D, condition::BoundaryCondition, t::Float64)
    nz = size(state.w, 3)
    if condition.type == NoSlip
        state.u[:, :, nz-1] .= 0.0
        state.v[:, :, nz-1] .= 0.0
        state.w[:, :, nz] .= 0.0
    end
end

function exchange_ghost_cells_3d!(decomp::MPI3DDecomposition, field::Union{Array{Float64,3}, PencilArray})
    # For PencilArrays compatibility, check if field is a PencilArray
    if isa(field, PencilArray)
        # Use PencilArrays built-in halo exchange for better performance
        PencilArrays.exchange_halo!(field)
        return
    end
    
    # Fallback to manual MPI communication for regular arrays
    requests = MPI.Request[]
    
    # X-direction exchange (left-right)
    if decomp.left_rank != MPI.MPI_PROC_NULL
        decomp.send_buffers[:left] .= field[1, :, :]
        req = MPI.Isend(decomp.send_buffers[:left], decomp.left_rank, 0, decomp.comm)
        push!(requests, req)
        
        req = MPI.Irecv!(decomp.recv_buffers[:left], decomp.left_rank, 1, decomp.comm)
        push!(requests, req)
    end
    
    if decomp.right_rank != MPI.MPI_PROC_NULL
        decomp.send_buffers[:right] .= field[end, :, :]
        req = MPI.Isend(decomp.send_buffers[:right], decomp.right_rank, 1, decomp.comm)
        push!(requests, req)
        
        req = MPI.Irecv!(decomp.recv_buffers[:right], decomp.right_rank, 0, decomp.comm)
        push!(requests, req)
    end
    
    # Y-direction exchange (bottom-top)
    if decomp.bottom_rank != MPI.MPI_PROC_NULL
        decomp.send_buffers[:bottom] .= field[:, 1, :]
        req = MPI.Isend(decomp.send_buffers[:bottom], decomp.bottom_rank, 2, decomp.comm)
        push!(requests, req)
        
        req = MPI.Irecv!(decomp.recv_buffers[:bottom], decomp.bottom_rank, 3, decomp.comm)
        push!(requests, req)
    end
    
    if decomp.top_rank != MPI.MPI_PROC_NULL
        decomp.send_buffers[:top] .= field[:, end, :]
        req = MPI.Isend(decomp.send_buffers[:top], decomp.top_rank, 3, decomp.comm)
        push!(requests, req)
        
        req = MPI.Irecv!(decomp.recv_buffers[:top], decomp.top_rank, 2, decomp.comm)
        push!(requests, req)
    end
    
    # Z-direction exchange (front-back)
    if decomp.front_rank != MPI.MPI_PROC_NULL
        decomp.send_buffers[:front] .= field[:, :, 1]
        req = MPI.Isend(decomp.send_buffers[:front], decomp.front_rank, 4, decomp.comm)
        push!(requests, req)
        
        req = MPI.Irecv!(decomp.recv_buffers[:front], decomp.front_rank, 5, decomp.comm)
        push!(requests, req)
    end
    
    if decomp.back_rank != MPI.MPI_PROC_NULL
        decomp.send_buffers[:back] .= field[:, :, end]
        req = MPI.Isend(decomp.send_buffers[:back], decomp.back_rank, 5, decomp.comm)
        push!(requests, req)
        
        req = MPI.Irecv!(decomp.recv_buffers[:back], decomp.back_rank, 4, decomp.comm)
        push!(requests, req)
    end
    
    # Wait for all communications
    MPI.Waitall(requests)
end

function gather_global_field_3d(decomp::MPI3DDecomposition, local_field::Array{Float64,3})
    if decomp.rank == 0
        global_field = zeros(decomp.nx_global, decomp.ny_global, decomp.nz_global)
        
        # Copy local data from root
        global_field[decomp.i_start:decomp.i_end, 
                    decomp.j_start:decomp.j_end, 
                    decomp.k_start:decomp.k_end] .= local_field
        
        # Receive from other processes
        for src_rank = 1:decomp.size-1
            src_coords = MPI.Cart_coords(decomp.comm, src_rank, 3)
            
            # Calculate source domain bounds (similar to 2D case but for 3D)
            src_nx_local = decomp.nx_global ÷ decomp.dims[1]
            src_ny_local = decomp.ny_global ÷ decomp.dims[2]
            src_nz_local = decomp.nz_global ÷ decomp.dims[3]
            
            if src_coords[1] < decomp.nx_global % decomp.dims[1]
                src_nx_local += 1
            end
            if src_coords[2] < decomp.ny_global % decomp.dims[2]
                src_ny_local += 1
            end
            if src_coords[3] < decomp.nz_global % decomp.dims[3]
                src_nz_local += 1
            end
            
            src_i_start = src_coords[1] * (decomp.nx_global ÷ decomp.dims[1]) + 1 + 
                         min(src_coords[1], decomp.nx_global % decomp.dims[1])
            src_i_end = src_i_start + src_nx_local - 1
            src_j_start = src_coords[2] * (decomp.ny_global ÷ decomp.dims[2]) + 1 + 
                         min(src_coords[2], decomp.ny_global % decomp.dims[2])
            src_j_end = src_j_start + src_ny_local - 1
            src_k_start = src_coords[3] * (decomp.nz_global ÷ decomp.dims[3]) + 1 + 
                         min(src_coords[3], decomp.nz_global % decomp.dims[3])
            src_k_end = src_k_start + src_nz_local - 1
            
            recv_buffer = zeros(src_nx_local, src_ny_local, src_nz_local)
            MPI.Recv!(recv_buffer, src_rank, 300 + src_rank, decomp.comm)
            
            global_field[src_i_start:src_i_end, src_j_start:src_j_end, src_k_start:src_k_end] .= recv_buffer
        end
        
        return global_field
    else
        MPI.Send(local_field, 0, 300 + decomp.rank, decomp.comm)
        return nothing
    end
end


struct MPINavierStokesSolver3D <: AbstractSolver
    decomp::MPI3DDecomposition
    local_grid::StaggeredGrid
    fluid::FluidProperties
    bc::BoundaryConditions
    time_scheme::TimeSteppingScheme
    multigrid_solver::Union{MultigridPoissonSolver, Nothing}  # Use MPI multigrid solver
    
    # Local work arrays
    local_u_star::Array{Float64,3}
    local_v_star::Array{Float64,3}
    local_w_star::Array{Float64,3}
    local_phi::Array{Float64,3}
    local_rhs_p::Array{Float64,3}
end

function MPINavierStokesSolver3D(nx_global::Int, ny_global::Int, nz_global::Int, 
                                Lx::Float64, Ly::Float64, Lz::Float64,
                                fluid::FluidProperties, bc::BoundaryConditions,
                                time_scheme::TimeSteppingScheme;
                                comm::MPI.Comm=MPI.COMM_WORLD)
    decomp = MPI3DDecomposition(nx_global, ny_global, nz_global, comm)
    local_grid = create_local_grid_3d(decomp, Lx, Ly, Lz)
    
    nx_local, ny_local, nz_local = decomp.nx_local, decomp.ny_local, decomp.nz_local
    local_u_star = zeros(nx_local+1, ny_local, nz_local)
    local_v_star = zeros(nx_local, ny_local+1, nz_local)
    local_w_star = zeros(nx_local, ny_local, nz_local+1)
    local_phi = zeros(nx_local, ny_local, nz_local)
    local_rhs_p = zeros(nx_local, ny_local, nz_local)
    
    MPINavierStokesSolver3D(decomp, local_grid, fluid, bc, time_scheme, nothing,
                           local_u_star, local_v_star, local_w_star, local_phi, local_rhs_p)
end

function mpi_solve_step_3d!(solver::MPINavierStokesSolver3D, local_state_new::SolutionState,
                           local_state_old::SolutionState, dt::Float64)
    decomp = solver.decomp
    
    # Step 1: Predictor step
    predictor_rhs = compute_predictor_rhs_3d(local_state_old, solver.local_grid, solver.fluid)
    
    local_state_predictor = deepcopy(local_state_old)
    time_step!(local_state_predictor, local_state_old,
              (state, args...) -> predictor_rhs,
              solver.time_scheme, dt, solver.local_grid, solver.fluid, solver.bc)
    
    # Step 2: Apply boundary conditions and exchange ghost cells
    apply_physical_boundary_conditions_3d!(decomp, solver.local_grid, local_state_predictor, solver.bc, local_state_old.t + dt)
    exchange_ghost_cells_3d!(decomp, local_state_predictor.u)
    exchange_ghost_cells_3d!(decomp, local_state_predictor.v)
    exchange_ghost_cells_3d!(decomp, local_state_predictor.w)
    
    # Step 3: Solve pressure Poisson equation
    divergence_3d!(solver.local_rhs_p, local_state_predictor.u, local_state_predictor.v, 
                  local_state_predictor.w, solver.local_grid)
    solver.local_rhs_p .*= 1.0 / dt
    
    # Use MPI multigrid solver (required for optimal performance)
    if solver.multigrid_solver !== nothing
        solve!(solver.multigrid_solver, solver.local_phi, solver.local_rhs_p)
    else
        error("Multigrid solver required for MPI pressure solution. Create solver with MultigridPoissonSolver.")
    end
    
    # Step 4: Velocity correction
    correct_velocity_3d!(local_state_new, local_state_predictor, solver.local_phi, dt, solver.local_grid)
    
    # Step 5: Pressure update
    if solver.fluid.ρ isa ConstantDensity
        ρ = solver.fluid.ρ.ρ
    else
        error("Variable density not implemented")
    end
    
    local_state_new.p .= local_state_old.p .+ solver.local_phi .* ρ
    local_state_new.t = local_state_old.t + dt
    local_state_new.step = local_state_old.step + 1
    
    # Apply final boundary conditions and ghost cell exchange
    apply_physical_boundary_conditions_3d!(decomp, solver.local_grid, local_state_new, solver.bc, local_state_new.t)
    exchange_ghost_cells_3d!(decomp, local_state_new.u)
    exchange_ghost_cells_3d!(decomp, local_state_new.v)
    exchange_ghost_cells_3d!(decomp, local_state_new.w)
    exchange_ghost_cells_3d!(decomp, local_state_new.p)
end

function compute_global_cfl_3d(decomp::MPI3DDecomposition, local_u::Array{Float64,3}, 
                              local_v::Array{Float64,3}, local_w::Array{Float64,3},
                              grid, dt::Float64)
    local_cfl = compute_cfl_3d(local_u, local_v, local_w, grid, dt)
    global_cfl = MPI.Allreduce(local_cfl, MPI.MAX, decomp.comm)
    return global_cfl
end
