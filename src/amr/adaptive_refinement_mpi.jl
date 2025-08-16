"""
MPI-Aware Adaptive Mesh Refinement

Extends the adaptive refinement system to work efficiently in parallel environments
with proper load balancing and communication patterns.
"""

using MPI

"""
    MPIAMRHierarchy
    
MPI-aware adaptive mesh refinement hierarchy with distributed refinement decisions
and load balancing capabilities.
"""
mutable struct MPIAMRHierarchy
    local_hierarchy::AMRHierarchy      # Local AMR hierarchy on this rank
    mpi_comm::MPI.Comm                 # MPI communicator
    rank::Int                          # MPI rank
    nprocs::Int                        # Number of MPI processes
    
    # Load balancing information
    local_cell_count::Vector{Int}      # Cells per process per level
    global_cell_count::Vector{Int}     # Total cells per level across all processes
    
    # Communication patterns
    neighbor_ranks::Vector{Int}        # Neighboring MPI ranks
    ghost_exchange_needed::Bool        # Whether ghost cell exchange is needed
    
    # Refinement coordination
    global_refinement_step::Int        # Global step counter for refinement decisions
    refinement_sync_interval::Int      # Steps between global refinement synchronization
    
    # Performance monitoring
    refinement_times::Vector{Float64}  # Time spent on refinement operations
    communication_times::Vector{Float64} # Time spent on MPI communication
end

function MPIAMRHierarchy(base_grid::StaggeredGrid, decomp::Union{MPI2DDecomposition, MPI3DDecomposition};
                        max_level::Int=3,
                        refinement_ratio::Int=2,
                        kwargs...)
    
    comm = decomp.comm
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    # Create local AMR hierarchy
    local_hierarchy = AMRHierarchy(base_grid; max_level=max_level, 
                                  refinement_ratio=refinement_ratio, kwargs...)
    
    # Initialize load balancing info
    local_cell_count = zeros(Int, max_level + 1)
    global_cell_count = zeros(Int, max_level + 1)
    local_cell_count[1] = base_grid.nx * base_grid.ny
    
    # Determine neighbor ranks from decomposition
    neighbor_ranks = get_neighbor_ranks(decomp)
    
    mpi_hierarchy = MPIAMRHierarchy(local_hierarchy, comm, rank, nprocs,
                                   local_cell_count, global_cell_count,
                                   neighbor_ranks, true, 0, 5,
                                   Float64[], Float64[])
    
    # Initial load balancing computation
    update_global_load_info!(mpi_hierarchy)
    
    return mpi_hierarchy
end

"""
    get_neighbor_ranks(decomp)
    
Extracts neighbor ranks from MPI decomposition structure.
"""
function get_neighbor_ranks(decomp::MPI2DDecomposition)
    neighbors = Int[]
    
    # Add valid neighbors (not MPI_PROC_NULL)
    for rank in [decomp.left_rank, decomp.right_rank, decomp.bottom_rank, decomp.top_rank]
        if rank != MPI.MPI_PROC_NULL && rank ∉ neighbors
            push!(neighbors, rank)
        end
    end
    
    return neighbors
end

function get_neighbor_ranks(decomp::MPI3DDecomposition)
    neighbors = Int[]
    
    for rank in [decomp.left_rank, decomp.right_rank, decomp.bottom_rank, 
                decomp.top_rank, decomp.front_rank, decomp.back_rank]
        if rank != MPI.MPI_PROC_NULL && rank ∉ neighbors
            push!(neighbors, rank)
        end
    end
    
    return neighbors
end

"""
    update_global_load_info!(mpi_hierarchy)
    
Updates global load balancing information across all MPI ranks.
"""
function update_global_load_info!(mpi_hierarchy::MPIAMRHierarchy)
    # Count local cells at each level
    hierarchy = mpi_hierarchy.local_hierarchy
    
    # FIXED: Ensure arrays are properly sized before counting
    max_levels = length(mpi_hierarchy.local_cell_count)
    fill!(mpi_hierarchy.local_cell_count, 0)
    
    mpi_hierarchy.local_cell_count[1] = count_cells_at_level(hierarchy.base_level, 0)
    
    for level = 1:min(hierarchy.max_level, max_levels-1)
        mpi_hierarchy.local_cell_count[level + 1] = count_cells_at_level(hierarchy.base_level, level)
    end
    
    # Global reduction to get total cell counts
    MPI.Allreduce!(mpi_hierarchy.local_cell_count, mpi_hierarchy.global_cell_count, 
                   MPI.SUM, mpi_hierarchy.mpi_comm)
end

"""
    count_cells_at_level(amr_level, target_level)
    
Recursively counts cells at a specific refinement level.
"""
function count_cells_at_level(amr_level::AMRLevel, target_level::Int)
    if amr_level.level == target_level
        # FIXED: Proper cell counting for 2D and 3D
        if amr_level.grid_type == TwoDimensional
            return amr_level.nx * amr_level.nz  # XZ plane
        else
            return amr_level.nx * amr_level.ny * amr_level.nz  # 3D
        end
    elseif amr_level.level < target_level
        # Count cells in children
        total = 0
        if amr_level.grid_type == TwoDimensional
            # Iterate over 2D children array
            for child in amr_level.children
                if child !== nothing
                    total += count_cells_at_level(child, target_level)
                end
            end
        else
            # Iterate over 3D children array
            for child in amr_level.children
                if child !== nothing
                    total += count_cells_at_level(child, target_level)
                end
            end
        end
        return total
    else
        return 0  # This level is finer than target
    end
end

"""
    coordinate_global_refinement!(mpi_hierarchy, state, bodies)
    
Coordinates adaptive refinement across all MPI processes with proper synchronization.
"""
function coordinate_global_refinement!(mpi_hierarchy::MPIAMRHierarchy,
                                     state::SolutionState,
                                     bodies::Union{RigidBodyCollection, FlexibleBodyCollection, Nothing})
    start_time = time()
    
    # Step 1: Compute local refinement indicators
    local_indicators = compute_local_refinement_needs(mpi_hierarchy, state, bodies)
    
    # Step 2: Exchange refinement decisions with neighbors
    global_refinement_map = exchange_refinement_decisions(mpi_hierarchy, local_indicators)
    
    # Step 3: Coordinate refinement to ensure consistent interfaces
    coordinated_refinement = coordinate_interface_consistency(mpi_hierarchy, global_refinement_map)
    
    # Step 4: Perform actual refinement
    refined_count = execute_coordinated_refinement!(mpi_hierarchy, state, coordinated_refinement)
    
    # Step 5: Update global load information
    update_global_load_info!(mpi_hierarchy)
    
    # Step 6: Rebalance if necessary
    if should_rebalance(mpi_hierarchy)
        rebalance_amr_hierarchy!(mpi_hierarchy)
    end
    
    # Record timing
    elapsed_time = time() - start_time
    push!(mpi_hierarchy.refinement_times, elapsed_time)
    
    # Synchronize step counter
    mpi_hierarchy.global_refinement_step += 1
    
    return refined_count
end

"""
    compute_local_refinement_needs(mpi_hierarchy, state, bodies)
    
Computes refinement indicators for local domain including ghost regions.
"""
function compute_local_refinement_needs(mpi_hierarchy::MPIAMRHierarchy,
                                      state::SolutionState,
                                      bodies::Union{RigidBodyCollection, FlexibleBodyCollection, Nothing})
    
    hierarchy = mpi_hierarchy.local_hierarchy
    
    # Exchange ghost cell data first if needed
    if mpi_hierarchy.ghost_exchange_needed
        exchange_ghost_amr_data!(mpi_hierarchy, state)
    end
    
    # Compute refinement indicators using advanced AMR criteria
    indicators = compute_refinement_indicators_amr(hierarchy.base_level, state, bodies, hierarchy)
    
    # Add MPI-specific considerations (interface consistency, load balancing)
    mpi_indicators = add_mpi_refinement_considerations(mpi_hierarchy, indicators)
    
    return mpi_indicators
end

"""
    add_mpi_refinement_considerations(mpi_hierarchy, base_indicators)
    
Adds MPI-specific refinement criteria such as interface consistency and load balancing.
"""
function add_mpi_refinement_considerations(mpi_hierarchy::MPIAMRHierarchy, 
                                         base_indicators::Matrix{Float64})
    enhanced_indicators = copy(base_indicators)
    nx, ny = size(base_indicators)
    
    # 1. Interface consistency: prefer refinement near processor boundaries
    interface_boost = 0.1
    boundary_width = 2
    
    # Check if this is a boundary process and boost indicators near boundaries
    if mpi_hierarchy.rank in mpi_hierarchy.neighbor_ranks
        # Left boundary
        enhanced_indicators[1:boundary_width, :] .+= interface_boost
        # Right boundary  
        enhanced_indicators[nx-boundary_width+1:nx, :] .+= interface_boost
        # Bottom boundary
        enhanced_indicators[:, 1:boundary_width] .+= interface_boost
        # Top boundary
        enhanced_indicators[:, ny-boundary_width+1:ny] .+= interface_boost
    end
    
    # 2. Load balancing consideration
    current_load = mpi_hierarchy.local_cell_count[1]
    avg_load = sum(mpi_hierarchy.global_cell_count) / mpi_hierarchy.nprocs
    
    if current_load > 1.2 * avg_load  # This process is overloaded
        # Reduce refinement likelihood
        enhanced_indicators .*= 0.8
    elseif current_load < 0.8 * avg_load  # This process is underloaded
        # Increase refinement likelihood
        enhanced_indicators .*= 1.2
    end
    
    return enhanced_indicators
end

"""
    exchange_refinement_decisions(mpi_hierarchy, local_indicators)
    
Exchanges refinement decisions with neighboring processes for coordination.
"""
function exchange_refinement_decisions(mpi_hierarchy::MPIAMRHierarchy,
                                     local_indicators::Matrix{Float64})
    comm_start_time = time()
    
    # Create refinement decision map
    local_decisions = local_indicators .> 0.5
    
    # Exchange boundary information with neighbors
    neighbor_decisions = Dict{Int, Matrix{Bool}}()
    
    requests = MPI.Request[]
    
    # Send local boundary decisions to neighbors
    for neighbor_rank in mpi_hierarchy.neighbor_ranks
        boundary_data = extract_boundary_decisions(local_decisions, neighbor_rank, mpi_hierarchy)
        req = MPI.Isend(boundary_data, neighbor_rank, 100 + mpi_hierarchy.rank, mpi_hierarchy.mpi_comm)
        push!(requests, req)
    end
    
    # Receive neighbor boundary decisions
    for neighbor_rank in mpi_hierarchy.neighbor_ranks
        expected_size = estimate_boundary_data_size(neighbor_rank, mpi_hierarchy)
        neighbor_data = Vector{Bool}(undef, expected_size)
        req = MPI.Irecv!(neighbor_data, neighbor_rank, 100 + neighbor_rank, mpi_hierarchy.mpi_comm)
        push!(requests, req)
        neighbor_decisions[neighbor_rank] = reshape(neighbor_data, :, :)  # Adjust reshape as needed
    end
    
    # Wait for all communications
    MPI.Waitall(requests)
    
    elapsed_comm_time = time() - comm_start_time
    push!(mpi_hierarchy.communication_times, elapsed_comm_time)
    
    # Combine local and neighbor decisions
    global_map = merge_refinement_decisions(local_decisions, neighbor_decisions)
    
    return global_map
end

"""
    coordinate_interface_consistency(mpi_hierarchy, global_refinement_map)
    
Ensures refinement decisions are consistent across processor interfaces.
"""
function coordinate_interface_consistency(mpi_hierarchy::MPIAMRHierarchy,
                                        global_refinement_map::Dict)
    # Apply consistency rules:
    # 1. Adjacent cells across processor boundaries should have similar refinement levels
    # 2. Refinement gradients should be smooth across interfaces
    # 3. No more than 1 level difference between adjacent cells
    
    coordinated_map = copy(global_refinement_map)
    
    # Apply 2:1 refinement ratio constraint across interfaces
    apply_refinement_constraints!(coordinated_map, mpi_hierarchy)
    
    return coordinated_map
end

"""
    apply_refinement_constraints!(refinement_map, mpi_hierarchy)
    
Applies refinement constraints to maintain solution quality and numerical stability.
"""
function apply_refinement_constraints!(refinement_map::Dict, mpi_hierarchy::MPIAMRHierarchy)
    # Implement 2:1 refinement ratio constraint
    # If a cell is refined, its neighbors can be at most 1 level different
    
    local_decisions = refinement_map["local"]
    nx, ny = size(local_decisions)
    
    # Apply constraint sweeps until convergence
    max_sweeps = 5
    for sweep = 1:max_sweeps
        changed = false
        
        for j = 2:ny-1, i = 2:nx-1
            if local_decisions[i, j]  # If this cell is refined
                # Check 4-connected neighbors
                neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
                
                for (ni, nj) in neighbors
                    if ni >= 1 && ni <= nx && nj >= 1 && nj <= ny
                        if !local_decisions[ni, nj]  # Neighbor not refined
                            # Check if refinement level difference would be > 1
                            # For now, mark neighbor for refinement to maintain 2:1 ratio
                            local_decisions[ni, nj] = true
                            changed = true
                        end
                    end
                end
            end
        end
        
        if !changed
            break
        end
    end
end

"""
    should_rebalance(mpi_hierarchy)
    
Determines whether load rebalancing is needed based on current distribution.
"""
function should_rebalance(mpi_hierarchy::MPIAMRHierarchy)
    total_cells = sum(mpi_hierarchy.global_cell_count)
    avg_cells = total_cells / mpi_hierarchy.nprocs
    
    # Calculate load imbalance
    local_cells = sum(mpi_hierarchy.local_cell_count)
    imbalance = abs(local_cells - avg_cells) / avg_cells
    
    # Rebalance if imbalance > 20% and significant cell count
    return imbalance > 0.2 && total_cells > 1000
end

"""
    rebalance_amr_hierarchy!(mpi_hierarchy)
    
Performs load rebalancing by redistributing refined regions across processes.
"""
function rebalance_amr_hierarchy!(mpi_hierarchy::MPIAMRHierarchy)
    # Advanced load balancing algorithm
    # 1. Compute optimal distribution
    # 2. Identify cells to migrate
    # 3. Execute migration with minimal communication
    # 4. Update data structures
    
    if mpi_hierarchy.rank == 0
        println("Performing AMR load rebalancing across $(mpi_hierarchy.nprocs) processes")
    end
    
    # For now, implement a simple space-filling curve based redistribution
    redistribute_using_space_filling_curve!(mpi_hierarchy)
end

"""
    redistribute_using_space_filling_curve!(mpi_hierarchy)
    
Redistributes AMR cells using space-filling curve for load balancing.
"""
function redistribute_using_space_filling_curve!(mpi_hierarchy::MPIAMRHierarchy)
    # Implement Hilbert curve or Morton ordering for optimal locality preservation
    # This is a placeholder for a complex algorithm
    
    # 1. Compute space-filling curve indices for all cells
    # 2. Sort by curve index
    # 3. Distribute equally among processes
    # 4. Migrate cells as needed
    
    println("Space-filling curve redistribution not yet implemented")
end

# Helper functions for MPI communication
function extract_boundary_decisions(decisions::Matrix{Bool}, neighbor_rank::Int, 
                                  mpi_hierarchy::MPIAMRHierarchy)
    # Extract relevant boundary data for this neighbor
    # Implementation depends on neighbor direction and domain decomposition
    
    nx, ny = size(decisions)
    
    # Simplified: return edge data (this needs proper implementation based on decomposition)
    if neighbor_rank == mpi_hierarchy.rank - 1  # Left neighbor
        return vec(decisions[1, :])
    elseif neighbor_rank == mpi_hierarchy.rank + 1  # Right neighbor
        return vec(decisions[nx, :])
    else
        return Bool[]  # Placeholder
    end
end

function estimate_boundary_data_size(neighbor_rank::Int, mpi_hierarchy::MPIAMRHierarchy)
    # Estimate size of boundary data from this neighbor
    # Implementation depends on domain decomposition details
    
    return 100  # Placeholder
end

function merge_refinement_decisions(local_decisions::Matrix{Bool}, 
                                  neighbor_decisions::Dict{Int, Matrix{Bool}})
    # Merge local and neighbor refinement decisions into global map
    
    global_map = Dict{String, Any}()
    global_map["local"] = local_decisions
    global_map["neighbors"] = neighbor_decisions
    
    return global_map
end

function execute_coordinated_refinement!(mpi_hierarchy::MPIAMRHierarchy,
                                       state::SolutionState,
                                       coordinated_refinement::Dict)
    # Execute the actual refinement based on coordinated decisions
    local_decisions = coordinated_refinement["local"]
    
    # Convert boolean decisions to refinement indicators
    indicators = Float64.(local_decisions)
    
    # Use existing refinement routine
    return refine_amr_level!(mpi_hierarchy.local_hierarchy, 
                           mpi_hierarchy.local_hierarchy.base_level,
                           state, indicators)
end

function exchange_ghost_amr_data!(mpi_hierarchy::MPIAMRHierarchy, state::SolutionState)
    # Exchange ghost cell data for AMR-aware operations
    # This needs integration with existing ghost cell exchange routines
    
    println("AMR ghost cell exchange not yet implemented")
end

# Export MPI AMR functionality
export MPIAMRHierarchy, coordinate_global_refinement!
export should_rebalance, rebalance_amr_hierarchy!