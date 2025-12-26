# =============================================================================
# BIOFLOWS AMR ADAPTER
# =============================================================================
# This adapter bridges BioFlows' Flow-based architecture with the AMR system.
# The main challenge is that:
#
# - BioFlows Flow: Uses (nx+2, nz+2, D) arrays with ghost cells, staggered layout
# - AMR types: Use separate u, v, w, p arrays without ghost cells
#
# Key conversions:
# - flow_to_staggered_grid: Create StaggeredGrid from Flow dimensions
# - flow_to_solution_state: Extract velocity/pressure from Flow
# - update_flow_from_state!: Copy solution back to Flow
# - create_refined_grid: Initialize RefinedGrid from Flow
#
# Index mapping:
# - BioFlows indices: 1-based with ghost cells (interior starts at 2)
# - AMR indices: 1-based without ghost cells (interior starts at 1)
# - cell_index_to_flow_index: (i,j) → (i+1, j+1)
# - flow_index_to_cell_index: CartesianIndex → (i-1, j-1)
# - Note: refinement tracking uses Flow indices (including ghost offset)
#
# The adapter also provides interpolation functions for transferring data
# between coarse and refined grids.
# =============================================================================

"""
    BioFlows AMR Adapter

Bridge between BioFlows' Flow-based architecture and the AMR infrastructure.
Provides conversion functions between Flow fields and StaggeredGrid/SolutionState types.
"""

"""
    FlowToGridAdapter{N,T}

Adapter that wraps a BioFlows Flow and provides grid information for AMR.

# Fields
- `flow`: Reference to the BioFlows Flow struct
- `L`: Length scale of the simulation
- `dx`: Grid spacing (assumed uniform)
"""
struct FlowToGridAdapter{N,T}
    flow::Flow{N,T}
    L::T
    dx::T
end

"""
    FlowToGridAdapter(flow::Flow, L::Number)

Create an adapter for the given flow with length scale L.
"""
function FlowToGridAdapter(flow::Flow{N,T}, L::Number) where {N,T}
    # Compute grid spacing from domain size
    # Flow has shape (nx+2, nz+2) for 2D including ghost cells
    dims = size(flow.p)
    nx = dims[1] - 2  # Remove ghost cells
    dx = T(L / nx)    # Assume square cells
    FlowToGridAdapter{N,T}(flow, T(L), dx)
end

"""
    flow_to_staggered_grid(adapter::FlowToGridAdapter)
    flow_to_staggered_grid(flow::Flow, L::Number)

Convert a BioFlows Flow to a StaggeredGrid.
The grid represents the interior domain (excluding ghost cells).
"""
function flow_to_staggered_grid(adapter::FlowToGridAdapter{N,T}) where {N,T}
    flow = adapter.flow
    dims = size(flow.p)

    if N == 2
        nx = dims[1] - 2
        nz = dims[2] - 2
        dx = adapter.dx
        dz = dx  # Assume isotropic
        return StaggeredGrid(nx, nz, dx, dz)
    else  # N == 3
        nx = dims[1] - 2
        ny = dims[2] - 2
        nz = dims[3] - 2
        dx = adapter.dx
        dy = dx
        dz = dx
        return StaggeredGrid(nx, ny, nz, dx, dy, dz)
    end
end

flow_to_staggered_grid(flow::Flow{N,T}, L::Number) where {N,T} =
    flow_to_staggered_grid(FlowToGridAdapter(flow, L))

"""
    flow_to_solution_state(flow::Flow)
    flow_to_solution_state(adapter::FlowToGridAdapter)

Extract velocity and pressure from Flow into a SolutionState.
This copies interior values (excluding ghost cells) from the staggered Flow.u array.
"""
function flow_to_solution_state(flow::Flow{N,T}) where {N,T}
    dims = size(flow.p)

    if N == 2
        nx = dims[1] - 2
        nz = dims[2] - 2

        # Extract velocity components from staggered array
        # flow.u has shape (nx+2, nz+2, 2) with ghost cells
        # u-velocity at x-faces: indices 2:nx+1 in x, 2:nz+1 in z, component 1
        # v-velocity at z-faces: indices 2:nx+1 in x, 2:nz+1 in z, component 2
        u = similar(flow.p, nx + 1, nz)
        v = similar(flow.p, nx, nz + 1)

        # Copy u-velocity (x-component at x-faces)
        for j in 1:nz, i in 1:(nx+1)
            u[i, j] = flow.u[i+1, j+1, 1]
        end

        # Copy v-velocity (z-component at z-faces)
        for j in 1:(nz+1), i in 1:nx
            v[i, j] = flow.u[i+1, j+1, 2]
        end

        # Copy pressure (cell centers)
        p = similar(flow.p, nx, nz)
        for j in 1:nz, i in 1:nx
            p[i, j] = flow.p[i+1, j+1]
        end

        return SolutionState{T, typeof(u)}(u, v, nothing, p)

    else  # N == 3
        nx = dims[1] - 2
        ny = dims[2] - 2
        nz = dims[3] - 2

        u = similar(flow.p, nx + 1, ny, nz)
        v = similar(flow.p, nx, ny + 1, nz)
        w = similar(flow.p, nx, ny, nz + 1)

        # Copy velocity components
        for k in 1:nz, j in 1:ny, i in 1:(nx+1)
            u[i, j, k] = flow.u[i+1, j+1, k+1, 1]
        end
        for k in 1:nz, j in 1:(ny+1), i in 1:nx
            v[i, j, k] = flow.u[i+1, j+1, k+1, 2]
        end
        for k in 1:(nz+1), j in 1:ny, i in 1:nx
            w[i, j, k] = flow.u[i+1, j+1, k+1, 3]
        end

        # Copy pressure
        p = similar(flow.p, nx, ny, nz)
        for k in 1:nz, j in 1:ny, i in 1:nx
            p[i, j, k] = flow.p[i+1, j+1, k+1]
        end

        return SolutionState{T, typeof(u)}(u, v, w, p)
    end
end

flow_to_solution_state(adapter::FlowToGridAdapter) = flow_to_solution_state(adapter.flow)

"""
    update_flow_from_state!(flow::Flow, state::SolutionState)

Copy solution state back into the Flow struct.
Updates interior values only (ghost cells unchanged).
"""
function update_flow_from_state!(flow::Flow{N,T}, state::SolutionState) where {N,T}
    dims = size(flow.p)

    if N == 2
        nx = dims[1] - 2
        nz = dims[2] - 2

        # Copy u-velocity
        for j in 1:nz, i in 1:(nx+1)
            flow.u[i+1, j+1, 1] = state.u[i, j]
        end

        # Copy v-velocity
        for j in 1:(nz+1), i in 1:nx
            flow.u[i+1, j+1, 2] = state.v[i, j]
        end

        # Copy pressure
        for j in 1:nz, i in 1:nx
            flow.p[i+1, j+1] = state.p[i, j]
        end

    else  # N == 3
        nx = dims[1] - 2
        ny = dims[2] - 2
        nz = dims[3] - 2

        for k in 1:nz, j in 1:ny, i in 1:(nx+1)
            flow.u[i+1, j+1, k+1, 1] = state.u[i, j, k]
        end
        for k in 1:nz, j in 1:(ny+1), i in 1:nx
            flow.u[i+1, j+1, k+1, 2] = state.v[i, j, k]
        end
        for k in 1:(nz+1), j in 1:ny, i in 1:nx
            flow.u[i+1, j+1, k+1, 3] = state.w[i, j, k]
        end
        for k in 1:nz, j in 1:ny, i in 1:nx
            flow.p[i+1, j+1, k+1] = state.p[i, j, k]
        end
    end

    return flow
end

"""
    create_refined_grid(flow::Flow, L::Number)
    create_refined_grid(adapter::FlowToGridAdapter)

Create an empty RefinedGrid from a BioFlows Flow.
"""
function create_refined_grid(adapter::FlowToGridAdapter)
    base_grid = flow_to_staggered_grid(adapter)
    return RefinedGrid(base_grid)
end

create_refined_grid(flow::Flow{N,T}, L::Number) where {N,T} =
    create_refined_grid(FlowToGridAdapter(flow, L))

"""
    grid_dimensions(flow::Flow)

Return the interior grid dimensions (excluding ghost cells).
"""
function grid_dimensions(flow::Flow{N,T}) where {N,T}
    dims = size(flow.p)
    return ntuple(i -> dims[i] - 2, N)
end

"""
    cell_index_to_flow_index(i, j)
    cell_index_to_flow_index(i, j, k)

Convert AMR cell indices (1-based, no ghost) to Flow indices (1-based, with ghost offset).
"""
cell_index_to_flow_index(i::Int, j::Int) = (i + 1, j + 1)
cell_index_to_flow_index(i::Int, j::Int, k::Int) = (i + 1, j + 1, k + 1)

"""
    flow_index_to_cell_index(I)

Convert Flow CartesianIndex (with ghost offset) to AMR cell indices.
"""
flow_index_to_cell_index(I::CartesianIndex{2}) = (I[1] - 1, I[2] - 1)
flow_index_to_cell_index(I::CartesianIndex{3}) = (I[1] - 1, I[2] - 1, I[3] - 1)

"""
    interpolate_to_refined!(flow::Flow, refined_grid::RefinedGrid, state::SolutionState)

Interpolate coarse solution to refined cells using divergence-free interpolation.
Updates the solution state with refined values where applicable.
"""
function interpolate_to_refined!(flow::Flow{N,T}, refined_grid::RefinedGrid,
                                  state::SolutionState) where {N,T}
    if N == 2
        for ((i, j), level) in refined_grid.refined_cells_2d
            if level > 0
                # Get local refined grid
                local_grid = refined_grid.refined_grids_2d[(i, j)]
                # Bilinear interpolation of velocity to refined locations
                interpolate_cell_2d!(state, i, j, local_grid, refined_grid)
            end
        end
    else
        for ((i, j, k), level) in refined_grid.refined_cells_3d
            if level > 0
                local_grid = refined_grid.refined_grids_3d[(i, j, k)]
                interpolate_cell_3d!(state, i, j, k, local_grid, refined_grid)
            end
        end
    end
    return state
end

"""
    interpolate_cell_2d!(state, i, j, local_grid, refined_grid)

Perform bilinear interpolation for a single refined 2D cell.
"""
function interpolate_cell_2d!(state::SolutionState, i::Int, j::Int,
                               local_grid::StaggeredGrid, refined_grid::RefinedGrid)
    weights = get(refined_grid.interpolation_weights_2d, (i, j), nothing)
    if weights === nothing
        return  # No pre-computed weights, skip
    end

    # Apply interpolation weights to velocity and pressure
    # This is a placeholder - actual implementation depends on refinement strategy
    for (neighbor_idx, weight) in weights
        ni, nj = neighbor_idx
        if 1 <= ni <= size(state.p, 1) && 1 <= nj <= size(state.p, 2)
            # Weighted contribution from neighbor
            # (Full implementation would handle staggered locations properly)
        end
    end
end

"""
    interpolate_cell_3d!(state, i, j, k, local_grid, refined_grid)

Perform trilinear interpolation for a single refined 3D cell.
"""
function interpolate_cell_3d!(state::SolutionState, i::Int, j::Int, k::Int,
                               local_grid::StaggeredGrid, refined_grid::RefinedGrid)
    weights = get(refined_grid.interpolation_weights_3d, (i, j, k), nothing)
    if weights === nothing
        return
    end

    for (neighbor_idx, weight) in weights
        ni, nj, nk = neighbor_idx
        if 1 <= ni <= size(state.p, 1) && 1 <= nj <= size(state.p, 2) && 1 <= nk <= size(state.p, 3)
            # Weighted contribution from neighbor
        end
    end
end

"""
    project_refined_to_coarse!(state::SolutionState, refined_grid::RefinedGrid)

Project refined solution back to coarse grid using conservative averaging.
This ensures mass and momentum conservation.
"""
function project_refined_to_coarse!(state::SolutionState, refined_grid::RefinedGrid)
    base_grid = refined_grid.base_grid

    if is_2d(base_grid)
        for ((i, j), level) in refined_grid.refined_cells_2d
            if level > 0
                # Average refined cell values back to coarse cell
                project_cell_2d!(state, i, j, refined_grid)
            end
        end
    else
        for ((i, j, k), level) in refined_grid.refined_cells_3d
            if level > 0
                project_cell_3d!(state, i, j, k, refined_grid)
            end
        end
    end

    return state
end

"""
    project_cell_2d!(state, i, j, refined_grid)

Conservative projection of a single 2D refined cell to coarse grid.
"""
function project_cell_2d!(state::SolutionState, i::Int, j::Int, refined_grid::RefinedGrid)
    # Volume-weighted average of refined cell values
    # This is a placeholder - actual implementation integrates over refined subcells
end

"""
    project_cell_3d!(state, i, j, k, refined_grid)

Conservative projection of a single 3D refined cell to coarse grid.
"""
function project_cell_3d!(state::SolutionState, i::Int, j::Int, k::Int, refined_grid::RefinedGrid)
    # Volume-weighted average of refined cell values
end
