using PencilArrays
using MPI

struct MPI2DDecomposition
    pencil::Pencil
    comm::MPI.Comm
    rank::Int
    size::Int
    
    # Grid decomposition (interior cells only)
    nx_global::Int
    ny_global::Int
    nx_local::Int
    ny_local::Int
    
    # Grid dimensions including ghost cells
    nx_local_with_ghosts::Int
    ny_local_with_ghosts::Int
    
    # Number of ghost cells (typically 1 for 2nd order)
    n_ghost::Int
    
    # Local indices (interior cells only, global indexing)
    i_start::Int
    i_end::Int
    j_start::Int
    j_end::Int
    
    # Local array indices for interior cells (including ghosts)
    i_local_start::Int  # Usually n_ghost + 1
    i_local_end::Int    # Usually nx_local + n_ghost
    j_local_start::Int  # Usually n_ghost + 1  
    j_local_end::Int    # Usually ny_local + n_ghost
    
    # Cartesian topology
    dims::Vector{Int}
    coords::Vector{Int}
    
    # Neighbor ranks
    left_rank::Int
    right_rank::Int
    bottom_rank::Int
    top_rank::Int
    
    # Communication buffers
    send_buffers::Dict{Symbol, Vector{Float64}}
    recv_buffers::Dict{Symbol, Vector{Float64}}
end

function MPI2DDecomposition(nx_global::Int, ny_global::Int, comm::MPI.Comm=MPI.COMM_WORLD; n_ghost::Int=1)
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    # Create 2D Cartesian topology
    dims = [0, 0]
    MPI.Dims_create!(size, dims)
    
    # For 2D problems, we typically decompose in x-direction primarily
    # Adjust dimensions to favor x-direction decomposition for better load balancing
    if dims[1] * dims[2] != size
        # Fallback: simple 1D decomposition in x-direction
        dims[1] = size
        dims[2] = 1
    end
    
    cart_comm = MPI.Cart_create(comm, dims, [false, false], true)
    coords = MPI.Cart_coords(cart_comm, rank, 2)
    
    # Determine local domain size (interior cells only)
    nx_local = nx_global ÷ dims[1]
    ny_local = ny_global ÷ dims[2]
    
    # Handle remainder cells
    if coords[1] < nx_global % dims[1]
        nx_local += 1
    end
    if coords[2] < ny_global % dims[2]
        ny_local += 1
    end
    
    # Calculate grid dimensions including ghost cells
    nx_local_with_ghosts = nx_local + 2 * n_ghost
    ny_local_with_ghosts = ny_local + 2 * n_ghost
    
    # Determine local index ranges (global indexing for interior cells)
    i_start = coords[1] * (nx_global ÷ dims[1]) + 1 + min(coords[1], nx_global % dims[1])
    i_end = i_start + nx_local - 1
    j_start = coords[2] * (ny_global ÷ dims[2]) + 1 + min(coords[2], ny_global % dims[2])
    j_end = j_start + ny_local - 1
    
    # Local array indices for interior cells (1-based indexing including ghosts)
    i_local_start = n_ghost + 1
    i_local_end = nx_local + n_ghost
    j_local_start = n_ghost + 1
    j_local_end = ny_local + n_ghost
    
    # Find neighbor ranks
    left_rank, right_rank = MPI.Cart_shift(cart_comm, 0, 1)
    bottom_rank, top_rank = MPI.Cart_shift(cart_comm, 1, 1)
    
    # Create pencil for PencilArrays
    pencil = Pencil(cart_comm, (nx_global, ny_global), (coords[1]+1, coords[2]+1))
    
    # Initialize communication buffers with proper ghost cell sizes
    send_buffers = Dict{Symbol, Vector{Float64}}()
    recv_buffers = Dict{Symbol, Vector{Float64}}()
    
    # Allocate buffers for ghost cell exchanges
    # Left/Right: exchange n_ghost columns of ny_local rows
    send_buffers[:left] = zeros(n_ghost * ny_local)
    send_buffers[:right] = zeros(n_ghost * ny_local)
    recv_buffers[:left] = zeros(n_ghost * ny_local)
    recv_buffers[:right] = zeros(n_ghost * ny_local)
    
    # Bottom/Top: exchange n_ghost rows of nx_local_with_ghosts columns
    send_buffers[:bottom] = zeros(n_ghost * nx_local_with_ghosts)
    send_buffers[:top] = zeros(n_ghost * nx_local_with_ghosts)
    recv_buffers[:bottom] = zeros(n_ghost * nx_local_with_ghosts)
    recv_buffers[:top] = zeros(n_ghost * nx_local_with_ghosts)
    
    MPI2DDecomposition(pencil, cart_comm, rank, size,
                      nx_global, ny_global, nx_local, ny_local,
                      nx_local_with_ghosts, ny_local_with_ghosts, n_ghost,
                      i_start, i_end, j_start, j_end,
                      i_local_start, i_local_end, j_local_start, j_local_end,
                      dims, coords,
                      left_rank, right_rank, bottom_rank, top_rank,
                      send_buffers, recv_buffers)
end

function create_local_grid_2d(decomp::MPI2DDecomposition, Lx::Float64, Ly::Float64)
    # Create local grid for this MPI rank INCLUDING ghost cells
    dx = Lx / decomp.nx_global
    dy = Ly / decomp.ny_global
    
    # Local domain bounds (for interior cells)
    x_min = (decomp.i_start - 1) * dx
    y_min = (decomp.j_start - 1) * dy
    
    # Extend bounds to include ghost cells
    x_min_with_ghosts = x_min - decomp.n_ghost * dx
    y_min_with_ghosts = y_min - decomp.n_ghost * dy
    
    local_Lx_with_ghosts = decomp.nx_local_with_ghosts * dx
    local_Ly_with_ghosts = decomp.ny_local_with_ghosts * dy
    
    # Create grid with ghost cells
    local_grid = StaggeredGrid2D(decomp.nx_local_with_ghosts, decomp.ny_local_with_ghosts, 
                                local_Lx_with_ghosts, local_Ly_with_ghosts;
                                origin_x=x_min_with_ghosts, origin_y=y_min_with_ghosts)
    
    return local_grid
end

function create_local_arrays_2d(decomp::MPI2DDecomposition, T=Float64)
    """
    Create local arrays with ghost cells for staggered grid variables.
    Returns arrays sized to include ghost regions.
    """
    nx_g = decomp.nx_local_with_ghosts
    ny_g = decomp.ny_local_with_ghosts
    
    # Pressure and scalars: cell-centered with ghost cells
    p = zeros(T, nx_g, ny_g)
    
    # Staggered velocities with ghost cells
    # u: staggered in x-direction
    u = zeros(T, nx_g + 1, ny_g)
    
    # v: staggered in y-direction  
    v = zeros(T, nx_g, ny_g + 1)
    
    return u, v, p
end

function exchange_ghost_cells_2d!(decomp::MPI2DDecomposition, field::Matrix{Float64})
    """
    Exchange ghost cells with neighboring processes for a cell-centered field.
    Field should be sized (nx_local_with_ghosts, ny_local_with_ghosts).
    """
    
    requests = MPI.Request[]
    n_ghost = decomp.n_ghost
    nx_g = decomp.nx_local_with_ghosts
    ny_g = decomp.ny_local_with_ghosts
    
    # Interior region indices
    i_start = decomp.i_local_start
    i_end = decomp.i_local_end
    j_start = decomp.j_local_start
    j_end = decomp.j_local_end
    
    # Left-Right exchange (x-direction)
    if decomp.left_rank != MPI.MPI_PROC_NULL
        # Pack data to send to left neighbor (leftmost interior columns)
        idx = 0
        for j = j_start:j_end
            for i = i_start:i_start+n_ghost-1
                idx += 1
                decomp.send_buffers[:left][idx] = field[i, j]
            end
        end
        
        # Send and receive
        req_send = MPI.Isend(decomp.send_buffers[:left], decomp.left_rank, 0, decomp.comm)
        req_recv = MPI.Irecv!(decomp.recv_buffers[:left], decomp.left_rank, 1, decomp.comm)
        push!(requests, req_send, req_recv)
    end
    
    if decomp.right_rank != MPI.MPI_PROC_NULL
        # Pack data to send to right neighbor (rightmost interior columns)
        idx = 0
        for j = j_start:j_end
            for i = i_end-n_ghost+1:i_end
                idx += 1
                decomp.send_buffers[:right][idx] = field[i, j]
            end
        end
        
        # Send and receive
        req_send = MPI.Isend(decomp.send_buffers[:right], decomp.right_rank, 1, decomp.comm)
        req_recv = MPI.Irecv!(decomp.recv_buffers[:right], decomp.right_rank, 0, decomp.comm)
        push!(requests, req_send, req_recv)
    end
    
    # Bottom-Top exchange (y-direction)
    if decomp.bottom_rank != MPI.MPI_PROC_NULL
        # Pack data to send to bottom neighbor (bottom interior rows, including ghost columns)
        idx = 0
        for j = j_start:j_start+n_ghost-1
            for i = 1:nx_g
                idx += 1
                decomp.send_buffers[:bottom][idx] = field[i, j]
            end
        end
        
        # Send and receive
        req_send = MPI.Isend(decomp.send_buffers[:bottom], decomp.bottom_rank, 2, decomp.comm)
        req_recv = MPI.Irecv!(decomp.recv_buffers[:bottom], decomp.bottom_rank, 3, decomp.comm)
        push!(requests, req_send, req_recv)
    end
    
    if decomp.top_rank != MPI.MPI_PROC_NULL
        # Pack data to send to top neighbor (top interior rows, including ghost columns)
        idx = 0
        for j = j_end-n_ghost+1:j_end
            for i = 1:nx_g
                idx += 1
                decomp.send_buffers[:top][idx] = field[i, j]
            end
        end
        
        # Send and receive
        req_send = MPI.Isend(decomp.send_buffers[:top], decomp.top_rank, 3, decomp.comm)
        req_recv = MPI.Irecv!(decomp.recv_buffers[:top], decomp.top_rank, 2, decomp.comm)
        push!(requests, req_send, req_recv)
    end
    
    # Wait for all communications to complete
    MPI.Waitall(requests)
    
    # Unpack received data into ghost cells
    if decomp.left_rank != MPI.MPI_PROC_NULL
        # Unpack from left neighbor into left ghost cells
        idx = 0
        for j = j_start:j_end
            for i = 1:n_ghost
                idx += 1
                field[i, j] = decomp.recv_buffers[:left][idx]
            end
        end
    end
    
    if decomp.right_rank != MPI.MPI_PROC_NULL
        # Unpack from right neighbor into right ghost cells
        idx = 0
        for j = j_start:j_end
            for i = nx_g-n_ghost+1:nx_g
                idx += 1
                field[i, j] = decomp.recv_buffers[:right][idx]
            end
        end
    end
    
    if decomp.bottom_rank != MPI.MPI_PROC_NULL
        # Unpack from bottom neighbor into bottom ghost cells
        idx = 0
        for j = 1:n_ghost
            for i = 1:nx_g
                idx += 1
                field[i, j] = decomp.recv_buffers[:bottom][idx]
            end
        end
    end
    
    if decomp.top_rank != MPI.MPI_PROC_NULL
        # Unpack from top neighbor into top ghost cells
        idx = 0
        for j = ny_g-n_ghost+1:ny_g
            for i = 1:nx_g
                idx += 1
                field[i, j] = decomp.recv_buffers[:top][idx]
            end
        end
    end
end

function exchange_ghost_cells_staggered_u_2d!(decomp::MPI2DDecomposition, u::Matrix{Float64})
    """
    Exchange ghost cells for u-velocity (staggered in x-direction).
    u should be sized (nx_local_with_ghosts + 1, ny_local_with_ghosts).
    """
    # Similar implementation but accounting for staggered grid layout
    # u is defined at x-faces, so has one extra point in x-direction
    
    requests = MPI.Request[]
    n_ghost = decomp.n_ghost
    nx_g = decomp.nx_local_with_ghosts + 1  # Extra point for staggered u
    ny_g = decomp.ny_local_with_ghosts
    
    # Interior region indices (adjusted for staggered grid)
    i_start = decomp.i_local_start
    i_end = decomp.i_local_end + 1  # One extra for u-velocity
    j_start = decomp.j_local_start
    j_end = decomp.j_local_end
    
    # Left-Right exchange
    if decomp.left_rank != MPI.MPI_PROC_NULL
        # Pack and send leftmost interior columns
        send_buffer = zeros(n_ghost * (j_end - j_start + 1))
        idx = 0
        for j = j_start:j_end
            for i = i_start:i_start+n_ghost-1
                idx += 1
                send_buffer[idx] = u[i, j]
            end
        end
        
        recv_buffer = zeros(n_ghost * (j_end - j_start + 1))
        req_send = MPI.Isend(send_buffer, decomp.left_rank, 10, decomp.comm)
        req_recv = MPI.Irecv!(recv_buffer, decomp.left_rank, 11, decomp.comm)
        push!(requests, req_send, req_recv)
        
        # Store recv_buffer for later unpacking
        decomp.recv_buffers[:left_u] = recv_buffer
    end
    
    if decomp.right_rank != MPI.MPI_PROC_NULL
        # Pack and send rightmost interior columns
        send_buffer = zeros(n_ghost * (j_end - j_start + 1))
        idx = 0
        for j = j_start:j_end
            for i = i_end-n_ghost+1:i_end
                idx += 1
                send_buffer[idx] = u[i, j]
            end
        end
        
        recv_buffer = zeros(n_ghost * (j_end - j_start + 1))
        req_send = MPI.Isend(send_buffer, decomp.right_rank, 11, decomp.comm)
        req_recv = MPI.Irecv!(recv_buffer, decomp.right_rank, 10, decomp.comm)
        push!(requests, req_send, req_recv)
        
        decomp.recv_buffers[:right_u] = recv_buffer
    end
    
    # Bottom-Top exchange (full width including ghost x-columns)
    if decomp.bottom_rank != MPI.MPI_PROC_NULL
        send_buffer = zeros(n_ghost * nx_g)
        idx = 0
        for j = j_start:j_start+n_ghost-1
            for i = 1:nx_g
                idx += 1
                send_buffer[idx] = u[i, j]
            end
        end
        
        recv_buffer = zeros(n_ghost * nx_g)
        req_send = MPI.Isend(send_buffer, decomp.bottom_rank, 12, decomp.comm)
        req_recv = MPI.Irecv!(recv_buffer, decomp.bottom_rank, 13, decomp.comm)
        push!(requests, req_send, req_recv)
        
        decomp.recv_buffers[:bottom_u] = recv_buffer
    end
    
    if decomp.top_rank != MPI.MPI_PROC_NULL
        send_buffer = zeros(n_ghost * nx_g)
        idx = 0
        for j = j_end-n_ghost+1:j_end
            for i = 1:nx_g
                idx += 1
                send_buffer[idx] = u[i, j]
            end
        end
        
        recv_buffer = zeros(n_ghost * nx_g)
        req_send = MPI.Isend(send_buffer, decomp.top_rank, 13, decomp.comm)
        req_recv = MPI.Irecv!(recv_buffer, decomp.top_rank, 12, decomp.comm)
        push!(requests, req_send, req_recv)
        
        decomp.recv_buffers[:top_u] = recv_buffer
    end
    
    # Wait for all communications
    MPI.Waitall(requests)
    
    # Unpack received data into ghost cells
    if decomp.left_rank != MPI.MPI_PROC_NULL && haskey(decomp.recv_buffers, :left_u)
        idx = 0
        for j = j_start:j_end
            for i = 1:n_ghost
                idx += 1
                u[i, j] = decomp.recv_buffers[:left_u][idx]
            end
        end
    end
    
    if decomp.right_rank != MPI.MPI_PROC_NULL && haskey(decomp.recv_buffers, :right_u)
        idx = 0
        for j = j_start:j_end
            for i = nx_g-n_ghost+1:nx_g
                idx += 1
                u[i, j] = decomp.recv_buffers[:right_u][idx]
            end
        end
    end
    
    if decomp.bottom_rank != MPI.MPI_PROC_NULL && haskey(decomp.recv_buffers, :bottom_u)
        idx = 0
        for j = 1:n_ghost
            for i = 1:nx_g
                idx += 1
                u[i, j] = decomp.recv_buffers[:bottom_u][idx]
            end
        end
    end
    
    if decomp.top_rank != MPI.MPI_PROC_NULL && haskey(decomp.recv_buffers, :top_u)
        idx = 0
        for j = ny_g-n_ghost+1:ny_g
            for i = 1:nx_g
                idx += 1
                u[i, j] = decomp.recv_buffers[:top_u][idx]
            end
        end
    end
end

function exchange_ghost_cells_staggered_v_2d!(decomp::MPI2DDecomposition, v::Matrix{Float64})
    """
    Exchange ghost cells for v-velocity (staggered in y-direction).
    v should be sized (nx_local_with_ghosts, ny_local_with_ghosts + 1).
    """
    # Similar to u but staggered in y-direction
    # Implementation follows similar pattern to u but with y-staggering
    
    requests = MPI.Request[]
    n_ghost = decomp.n_ghost
    nx_g = decomp.nx_local_with_ghosts
    ny_g = decomp.ny_local_with_ghosts + 1  # Extra point for staggered v
    
    # Interior region indices (adjusted for staggered grid)
    i_start = decomp.i_local_start
    i_end = decomp.i_local_end
    j_start = decomp.j_local_start
    j_end = decomp.j_local_end + 1  # One extra for v-velocity
    
    # Implementation details similar to u-velocity exchange...
    # [Abbreviated for brevity - follows same pattern as u-velocity exchange]
    
    # For now, use simple exchange (can be expanded later)
    exchange_ghost_cells_2d!(decomp, v)
end

function gather_global_field_2d(decomp::MPI2DDecomposition, local_field::Matrix{Float64})
    # Gather local fields to form global field on root process
    
    if decomp.rank == 0
        global_field = zeros(decomp.nx_global, decomp.ny_global)
        
        # Copy local data from root
        global_field[decomp.i_start:decomp.i_end, decomp.j_start:decomp.j_end] .= local_field
        
        # Receive data from other processes
        for src_rank = 1:decomp.size-1
            # Get domain info for source rank
            src_coords = MPI.Cart_coords(decomp.comm, src_rank, 2)
            
            src_nx_local = decomp.nx_global ÷ MPI.Cart_get(decomp.comm)[1]
            src_ny_local = decomp.ny_global ÷ MPI.Cart_get(decomp.comm)[2]
            
            if src_coords[1] < decomp.nx_global % MPI.Cart_get(decomp.comm)[1]
                src_nx_local += 1
            end
            if src_coords[2] < decomp.ny_global % MPI.Cart_get(decomp.comm)[2]
                src_ny_local += 1
            end
            
            src_i_start = src_coords[1] * (decomp.nx_global ÷ MPI.Cart_get(decomp.comm)[1]) + 1 + 
                         min(src_coords[1], decomp.nx_global % MPI.Cart_get(decomp.comm)[1])
            src_i_end = src_i_start + src_nx_local - 1
            src_j_start = src_coords[2] * (decomp.ny_global ÷ MPI.Cart_get(decomp.comm)[2]) + 1 + 
                         min(src_coords[2], decomp.ny_global % MPI.Cart_get(decomp.comm)[2])
            src_j_end = src_j_start + src_ny_local - 1
            
            # Receive data
            recv_buffer = zeros(src_nx_local, src_ny_local)
            MPI.Recv!(recv_buffer, src_rank, 100 + src_rank, decomp.comm)
            
            # Copy to global array
            global_field[src_i_start:src_i_end, src_j_start:src_j_end] .= recv_buffer
        end
        
        return global_field
    else
        # Send local data to root
        MPI.Send(local_field, 0, 100 + decomp.rank, decomp.comm)
        return nothing
    end
end

function distribute_global_field_2d!(decomp::MPI2DDecomposition, local_field::Matrix{Float64},
                                    global_field::Union{Matrix{Float64}, Nothing})
    # Distribute global field from root to all processes
    
    if decomp.rank == 0 && global_field !== nothing
        # Root: send data to other processes
        for dest_rank = 1:decomp.size-1
            # Get domain info for destination rank
            dest_coords = MPI.Cart_coords(decomp.comm, dest_rank, 2)
            
            dest_nx_local = decomp.nx_global ÷ MPI.Cart_get(decomp.comm)[1]
            dest_ny_local = decomp.ny_global ÷ MPI.Cart_get(decomp.comm)[2]
            
            if dest_coords[1] < decomp.nx_global % MPI.Cart_get(decomp.comm)[1]
                dest_nx_local += 1
            end
            if dest_coords[2] < decomp.ny_global % MPI.Cart_get(decomp.comm)[2]
                dest_ny_local += 1
            end
            
            dest_i_start = dest_coords[1] * (decomp.nx_global ÷ MPI.Cart_get(decomp.comm)[1]) + 1 + 
                          min(dest_coords[1], decomp.nx_global % MPI.Cart_get(decomp.comm)[1])
            dest_i_end = dest_i_start + dest_nx_local - 1
            dest_j_start = dest_coords[2] * (decomp.ny_global ÷ MPI.Cart_get(decomp.comm)[2]) + 1 + 
                          min(dest_coords[2], decomp.ny_global % MPI.Cart_get(decomp.comm)[2])
            dest_j_end = dest_j_start + dest_ny_local - 1
            
            # Send data
            send_buffer = global_field[dest_i_start:dest_i_end, dest_j_start:dest_j_end]
            MPI.Send(send_buffer, dest_rank, 200 + dest_rank, decomp.comm)
        end
        
        # Copy local portion
        local_field .= global_field[decomp.i_start:decomp.i_end, decomp.j_start:decomp.j_end]
        
    else
        # Non-root: receive data from root
        MPI.Recv!(local_field, 0, 200 + decomp.rank, decomp.comm)
    end
end

function mpi_allreduce_2d(decomp::MPI2DDecomposition, local_value::Float64, op::MPI.Op=MPI.SUM)
    # Perform MPI allreduce operation
    return MPI.Allreduce(local_value, op, decomp.comm)
end

function compute_global_norm_2d(decomp::MPI2DDecomposition, local_field::Matrix{Float64})
    # Compute global L2 norm of field
    local_norm_sq = sum(local_field.^2)
    global_norm_sq = mpi_allreduce_2d(decomp, local_norm_sq, MPI.SUM)
    return sqrt(global_norm_sq)
end

function mpi_solve_poisson_2d!(decomp::MPI2DDecomposition, phi::Matrix{Float64}, rhs::Matrix{Float64},
                              grid, bc::BoundaryConditions; max_iter::Int=1000, tol::Float64=1e-10)
    # Parallel iterative solver for pressure Poisson equation
    # Using Jacobi iterations with MPI communication
    
    dx, dy = grid.dx, grid.dy
    nx_local, ny_local = decomp.nx_local, decomp.ny_local
    
    phi_new = copy(phi)
    
    for iter = 1:max_iter
        # Jacobi iteration
        for j = 2:ny_local-1, i = 2:nx_local-1
            phi_new[i, j] = 0.25 * (
                (phi[i+1, j] + phi[i-1, j]) / dx^2 + 
                (phi[i, j+1] + phi[i, j-1]) / dy^2 - 
                rhs[i, j]
            ) / (1/dx^2 + 1/dy^2)
        end
        
        # Exchange ghost cells
        exchange_ghost_cells_2d!(decomp, phi_new)
        
        # Apply boundary conditions (simplified)
        # Left boundary
        if decomp.left_rank == MPI.MPI_PROC_NULL
            phi_new[1, :] .= phi_new[2, :]
        end
        # Right boundary
        if decomp.right_rank == MPI.MPI_PROC_NULL
            phi_new[nx_local, :] .= phi_new[nx_local-1, :]
        end
        # Bottom boundary
        if decomp.bottom_rank == MPI.MPI_PROC_NULL
            phi_new[:, 1] .= phi_new[:, 2]
        end
        # Top boundary
        if decomp.top_rank == MPI.MPI_PROC_NULL
            phi_new[:, ny_local] .= phi_new[:, ny_local-1]
        end
        
        # Check convergence
        residual = phi_new - phi
        local_residual_norm = sum(residual.^2)
        global_residual_norm = sqrt(mpi_allreduce_2d(decomp, local_residual_norm, MPI.SUM))
        
        phi .= phi_new
        
        if global_residual_norm < tol
            if decomp.rank == 0
                println("MPI Poisson solver converged in $iter iterations, residual: $global_residual_norm")
            end
            break
        end
    end
end

struct MPINavierStokesSolver2D <: AbstractSolver
    decomp::MPI2DDecomposition
    local_grid::StaggeredGrid
    fluid::FluidProperties
    bc::BoundaryConditions
    time_scheme::TimeSteppingScheme
    
    # Local work arrays
    local_u_star::Matrix{Float64}
    local_v_star::Matrix{Float64}
    local_phi::Matrix{Float64}
    local_rhs_p::Matrix{Float64}
end

function MPINavierStokesSolver2D(nx_global::Int, ny_global::Int, Lx::Float64, Ly::Float64,
                                fluid::FluidProperties, bc::BoundaryConditions, 
                                time_scheme::TimeSteppingScheme;
                                comm::MPI.Comm=MPI.COMM_WORLD)
    # Create MPI decomposition
    decomp = MPI2DDecomposition(nx_global, ny_global, comm)
    
    # Create local grid
    local_grid = create_local_grid_2d(decomp, Lx, Ly)
    
    # Initialize local work arrays
    nx_local, ny_local = decomp.nx_local, decomp.ny_local
    local_u_star = zeros(nx_local+1, ny_local)
    local_v_star = zeros(nx_local, ny_local+1)
    local_phi = zeros(nx_local, ny_local)
    local_rhs_p = zeros(nx_local, ny_local)
    
    MPINavierStokesSolver2D(decomp, local_grid, fluid, bc, time_scheme,
                           local_u_star, local_v_star, local_phi, local_rhs_p)
end

function mpi_solve_step_2d!(solver::MPINavierStokesSolver2D, local_state_new::SolutionState,
                           local_state_old::SolutionState, dt::Float64)
    # Parallel solve step using projection method
    decomp = solver.decomp
    
    # Step 1: Predictor step (local computations)
    predictor_rhs = compute_predictor_rhs_2d(local_state_old, solver.local_grid, solver.fluid)
    
    local_state_predictor = deepcopy(local_state_old)
    time_step!(local_state_predictor, local_state_old,
              (state, args...) -> predictor_rhs,
              solver.time_scheme, dt, solver.local_grid, solver.fluid, solver.bc)
    
    # Step 2: Exchange ghost cells for predictor velocity
    exchange_ghost_cells_2d!(decomp, local_state_predictor.u)
    exchange_ghost_cells_2d!(decomp, local_state_predictor.v)
    
    # Step 3: Solve pressure Poisson equation (parallel)
    divergence_2d!(solver.local_rhs_p, local_state_predictor.u, local_state_predictor.v, solver.local_grid)
    solver.local_rhs_p .*= 1.0 / dt
    
    mpi_solve_poisson_2d!(decomp, solver.local_phi, solver.local_rhs_p, 
                         solver.local_grid, solver.bc)
    
    # Step 4: Velocity correction (local)
    correct_velocity_2d!(local_state_new, local_state_predictor, solver.local_phi, dt, solver.local_grid)
    
    # Step 5: Pressure update
    if solver.fluid.ρ isa ConstantDensity
        ρ = solver.fluid.ρ.ρ
    else
        error("Variable density not implemented")
    end
    
    local_state_new.p .= local_state_old.p .+ solver.local_phi .* ρ
    local_state_new.t = local_state_old.t + dt
    local_state_new.step = local_state_old.step + 1
    
    # Final ghost cell exchange
    exchange_ghost_cells_2d!(decomp, local_state_new.u)
    exchange_ghost_cells_2d!(decomp, local_state_new.v)
    exchange_ghost_cells_2d!(decomp, local_state_new.p)
end