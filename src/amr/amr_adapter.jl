# =============================================================================
# AMR ADAPTER
# =============================================================================
# This adapter bridges BioFlows' Flow-based architecture with the AMR system.
# The main challenge is that:
#
# - BioFlows Flow: Uses (nx+2, nz+2, D) arrays with ghost cells, staggered layout
# - AMR types: Use separate u, v, w, p arrays without ghost cells
#
# Key conversions:
# - flow_to_staggered_grid: Create StaggeredGrid from Flow dimensions/spacing
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
    AMR Adapter

Bridge between BioFlows' Flow-based architecture and the AMR infrastructure.
Provides conversion functions between Flow fields and StaggeredGrid/SolutionState types.
"""

"""
    FlowToGridAdapter{N,T}

Adapter that wraps a BioFlows Flow and provides grid information for AMR.

# Fields
- `flow`: Reference to the BioFlows Flow struct
- `L`: Length scale of the simulation
- `dx`: Grid spacing in the x-direction (uses `flow.Δx[1]`)
"""
struct FlowToGridAdapter{N,T}
    flow::Flow{N,T}
    L::T
    dx::T
end

"""
    FlowToGridAdapter(flow::Flow, L::Number)

Create an adapter for the given flow with length scale L.

The grid spacing `dx` is computed as `L / nx` where `nx` is the number of
interior cells in the x-direction. This gives the non-dimensional grid spacing
based on the characteristic length L.
"""
function FlowToGridAdapter(flow::Flow{N,T}, L::Number) where {N,T}
    # Compute non-dimensional grid spacing: L / nx
    # Interior grid size is (size(p) - 2) due to ghost cells
    nx = size(flow.p, 1) - 2
    dx = T(L) / T(nx)
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

        dx = T(flow.Δx[1])
        dz = T(flow.Δx[2])
        return StaggeredGrid(nx, nz, dx, dz)
    else  # N == 3
        nx = dims[1] - 2
        ny = dims[2] - 2
        nz = dims[3] - 2

        dx = T(flow.Δx[1])
        dy = T(flow.Δx[2])
        dz = T(flow.Δx[3])
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

Interpolate coarse solution to refined cells using staggered-aware bilinear/trilinear
interpolation. Creates refined SolutionStates stored in refined_grid.refined_states_2d/3d.

For each refined cell, this:
1. Creates a local refined SolutionState if not already present
2. Interpolates velocity components accounting for staggered grid locations
3. Interpolates pressure at cell centers
"""
function interpolate_to_refined!(flow::Flow{N,T}, refined_grid::RefinedGrid{T},
                                  state::SolutionState{T}) where {N,T}
    if N == 2
        for ((i, j), level) in refined_grid.refined_cells_2d
            if level > 0
                # Get local refined grid structure
                local_grid = get(refined_grid.refined_grids_2d, (i, j), nothing)
                if local_grid !== nothing
                    # Bilinear interpolation of velocity and pressure to refined locations
                    interpolate_cell_2d!(state, i, j, local_grid, refined_grid)
                end
            end
        end
    else
        for ((i, j, k), level) in refined_grid.refined_cells_3d
            if level > 0
                local_grid = get(refined_grid.refined_grids_3d, (i, j, k), nothing)
                if local_grid !== nothing
                    interpolate_cell_3d!(state, i, j, k, local_grid, refined_grid)
                end
            end
        end
    end
    return state
end

"""
    interpolate_cell_2d!(state, i, j, local_grid, refined_grid)

Perform bilinear interpolation for a single refined 2D cell.
Creates refined SolutionState and stores in refined_grid.refined_states_2d.

Uses staggered-aware interpolation:
- u-velocity: offset by 0.5 in x-direction (at x-faces)
- v-velocity: offset by 0.5 in z-direction (at z-faces)
- pressure: no offset (at cell centers)
"""
function interpolate_cell_2d!(state::SolutionState{T}, i::Int, j::Int,
                               local_grid::StaggeredGrid{T}, refined_grid::RefinedGrid{T}) where {T}
    level = get(refined_grid.refined_cells_2d, (i, j), 0)
    if level == 0
        return
    end

    ratio = 2^level  # refinement ratio

    # Create or get refined solution state for this cell
    if !haskey(refined_grid.refined_states_2d, (i, j))
        refined_grid.refined_states_2d[(i, j)] = SolutionState(local_grid)
    end
    refined_state = refined_grid.refined_states_2d[(i, j)]

    # Coarse grid dimensions
    nx_c, nz_c = size(state.p)

    # Fine grid dimensions
    nx_f = local_grid.nx
    nz_f = local_grid.nz

    # Interpolate u-velocity (at x-faces, staggered in x)
    for fj in 1:nz_f, fi in 1:(nx_f+1)
        # Fine grid position relative to coarse cell (i,j)
        # Account for x-face staggering (faces at 0, dx, 2dx, ...)
        xf = (T(fi) - T(1)) / ratio  # 0 to 1 across the cell
        zf = (T(fj) - T(0.5)) / ratio  # 0.5/ratio to (nz_f-0.5)/ratio

        # Bilinear interpolation from coarse u-velocity
        # Coarse u at x-faces: indices 1:nx_c+1 in x, 1:nz_c in z
        ic = clamp(i + floor(Int, xf), 1, nx_c)
        jc = clamp(j + floor(Int, zf), 1, nz_c)
        wx = xf - floor(xf)
        wz = zf - floor(zf)

        # Clamp indices for boundary safety
        ic1 = clamp(ic + 1, 1, size(state.u, 1))
        jc1 = clamp(jc + 1, 1, size(state.u, 2))

        refined_state.u[fi, fj] = (one(T) - wx) * (one(T) - wz) * state.u[ic, jc] +
                                   wx * (one(T) - wz) * state.u[ic1, jc] +
                                   (one(T) - wx) * wz * state.u[ic, jc1] +
                                   wx * wz * state.u[ic1, jc1]
    end

    # Interpolate v-velocity (at z-faces, staggered in z)
    for fj in 1:(nz_f+1), fi in 1:nx_f
        xf = (T(fi) - T(0.5)) / ratio
        zf = (T(fj) - T(1)) / ratio  # z-face staggering

        ic = clamp(i + floor(Int, xf), 1, nx_c)
        jc = clamp(j + floor(Int, zf), 1, nz_c)
        wx = xf - floor(xf)
        wz = zf - floor(zf)

        ic1 = clamp(ic + 1, 1, size(state.v, 1))
        jc1 = clamp(jc + 1, 1, size(state.v, 2))

        refined_state.v[fi, fj] = (one(T) - wx) * (one(T) - wz) * state.v[ic, jc] +
                                   wx * (one(T) - wz) * state.v[ic1, jc] +
                                   (one(T) - wx) * wz * state.v[ic, jc1] +
                                   wx * wz * state.v[ic1, jc1]
    end

    # Interpolate pressure (at cell centers)
    for fj in 1:nz_f, fi in 1:nx_f
        xf = (T(fi) - T(0.5)) / ratio
        zf = (T(fj) - T(0.5)) / ratio

        ic = clamp(i + floor(Int, xf), 1, nx_c)
        jc = clamp(j + floor(Int, zf), 1, nz_c)
        wx = xf - floor(xf)
        wz = zf - floor(zf)

        ic1 = clamp(ic + 1, 1, nx_c)
        jc1 = clamp(jc + 1, 1, nz_c)

        refined_state.p[fi, fj] = (one(T) - wx) * (one(T) - wz) * state.p[ic, jc] +
                                   wx * (one(T) - wz) * state.p[ic1, jc] +
                                   (one(T) - wx) * wz * state.p[ic, jc1] +
                                   wx * wz * state.p[ic1, jc1]
    end
end

"""
    interpolate_cell_3d!(state, i, j, k, local_grid, refined_grid)

Perform trilinear interpolation for a single refined 3D cell.
Creates refined SolutionState and stores in refined_grid.refined_states_3d.

Uses staggered-aware interpolation:
- u-velocity: offset by 0.5 in x-direction (at x-faces)
- v-velocity: offset by 0.5 in y-direction (at y-faces)
- w-velocity: offset by 0.5 in z-direction (at z-faces)
- pressure: no offset (at cell centers)
"""
function interpolate_cell_3d!(state::SolutionState{T}, i::Int, j::Int, k::Int,
                               local_grid::StaggeredGrid{T}, refined_grid::RefinedGrid{T}) where {T}
    level = get(refined_grid.refined_cells_3d, (i, j, k), 0)
    if level == 0
        return
    end

    ratio = 2^level

    # Create or get refined solution state
    if !haskey(refined_grid.refined_states_3d, (i, j, k))
        refined_grid.refined_states_3d[(i, j, k)] = SolutionState(local_grid)
    end
    refined_state = refined_grid.refined_states_3d[(i, j, k)]

    # Coarse grid dimensions
    nx_c, ny_c, nz_c = size(state.p)

    # Fine grid dimensions
    nx_f = local_grid.nx
    ny_f = local_grid.ny
    nz_f = local_grid.nz

    # Helper for trilinear interpolation
    @inline function trilinear(arr, ic, jc, kc, wx, wy, wz)
        ic1 = clamp(ic + 1, 1, size(arr, 1))
        jc1 = clamp(jc + 1, 1, size(arr, 2))
        kc1 = clamp(kc + 1, 1, size(arr, 3))

        c00 = (one(T) - wx) * arr[ic, jc, kc] + wx * arr[ic1, jc, kc]
        c10 = (one(T) - wx) * arr[ic, jc1, kc] + wx * arr[ic1, jc1, kc]
        c01 = (one(T) - wx) * arr[ic, jc, kc1] + wx * arr[ic1, jc, kc1]
        c11 = (one(T) - wx) * arr[ic, jc1, kc1] + wx * arr[ic1, jc1, kc1]

        c0 = (one(T) - wy) * c00 + wy * c10
        c1 = (one(T) - wy) * c01 + wy * c11

        return (one(T) - wz) * c0 + wz * c1
    end

    # Interpolate u-velocity (at x-faces)
    for fk in 1:nz_f, fj in 1:ny_f, fi in 1:(nx_f+1)
        xf = (T(fi) - T(1)) / ratio
        yf = (T(fj) - T(0.5)) / ratio
        zf = (T(fk) - T(0.5)) / ratio

        ic = clamp(i + floor(Int, xf), 1, size(state.u, 1) - 1)
        jc = clamp(j + floor(Int, yf), 1, ny_c)
        kc = clamp(k + floor(Int, zf), 1, nz_c)
        wx = clamp(xf - floor(xf), zero(T), one(T))
        wy = clamp(yf - floor(yf), zero(T), one(T))
        wz = clamp(zf - floor(zf), zero(T), one(T))

        refined_state.u[fi, fj, fk] = trilinear(state.u, ic, jc, kc, wx, wy, wz)
    end

    # Interpolate v-velocity (at y-faces)
    for fk in 1:nz_f, fj in 1:(ny_f+1), fi in 1:nx_f
        xf = (T(fi) - T(0.5)) / ratio
        yf = (T(fj) - T(1)) / ratio
        zf = (T(fk) - T(0.5)) / ratio

        ic = clamp(i + floor(Int, xf), 1, nx_c)
        jc = clamp(j + floor(Int, yf), 1, size(state.v, 2) - 1)
        kc = clamp(k + floor(Int, zf), 1, nz_c)
        wx = clamp(xf - floor(xf), zero(T), one(T))
        wy = clamp(yf - floor(yf), zero(T), one(T))
        wz = clamp(zf - floor(zf), zero(T), one(T))

        refined_state.v[fi, fj, fk] = trilinear(state.v, ic, jc, kc, wx, wy, wz)
    end

    # Interpolate w-velocity (at z-faces)
    for fk in 1:(nz_f+1), fj in 1:ny_f, fi in 1:nx_f
        xf = (T(fi) - T(0.5)) / ratio
        yf = (T(fj) - T(0.5)) / ratio
        zf = (T(fk) - T(1)) / ratio

        ic = clamp(i + floor(Int, xf), 1, nx_c)
        jc = clamp(j + floor(Int, yf), 1, ny_c)
        kc = clamp(k + floor(Int, zf), 1, size(state.w, 3) - 1)
        wx = clamp(xf - floor(xf), zero(T), one(T))
        wy = clamp(yf - floor(yf), zero(T), one(T))
        wz = clamp(zf - floor(zf), zero(T), one(T))

        refined_state.w[fi, fj, fk] = trilinear(state.w, ic, jc, kc, wx, wy, wz)
    end

    # Interpolate pressure (at cell centers)
    for fk in 1:nz_f, fj in 1:ny_f, fi in 1:nx_f
        xf = (T(fi) - T(0.5)) / ratio
        yf = (T(fj) - T(0.5)) / ratio
        zf = (T(fk) - T(0.5)) / ratio

        ic = clamp(i + floor(Int, xf), 1, nx_c)
        jc = clamp(j + floor(Int, yf), 1, ny_c)
        kc = clamp(k + floor(Int, zf), 1, nz_c)
        wx = clamp(xf - floor(xf), zero(T), one(T))
        wy = clamp(yf - floor(yf), zero(T), one(T))
        wz = clamp(zf - floor(zf), zero(T), one(T))

        refined_state.p[fi, fj, fk] = trilinear(state.p, ic, jc, kc, wx, wy, wz)
    end
end

"""
    project_refined_to_coarse!(state::SolutionState, refined_grid::RefinedGrid)

Project refined solution back to coarse grid using conservative averaging.
This ensures mass and momentum conservation.

For velocities: averages fine face values to corresponding coarse faces
For pressure: volume-weighted average of all fine cells to coarse cell
"""
function project_refined_to_coarse!(state::SolutionState{T},
                                     refined_grid::RefinedGrid{T}) where {T}
    base_grid = refined_grid.base_grid

    if is_2d(base_grid)
        for ((i, j), level) in refined_grid.refined_cells_2d
            if level > 0
                # Conservative averaging of refined cell values back to coarse cell
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
Uses volume-weighted averaging for pressure and face-averaged fluxes for velocities.
"""
function project_cell_2d!(state::SolutionState{T}, i::Int, j::Int,
                           refined_grid::RefinedGrid{T}) where {T}
    # Get refined state for this cell
    refined_state = get(refined_grid.refined_states_2d, (i, j), nothing)
    if refined_state === nothing
        return
    end

    level = get(refined_grid.refined_cells_2d, (i, j), 0)
    if level == 0
        return
    end

    ratio = 2^level
    inv_ratio = one(T) / ratio
    inv_ratio2 = inv_ratio * inv_ratio  # For area averaging

    # Fine grid dimensions
    nx_f = size(refined_state.p, 1)
    nz_f = size(refined_state.p, 2)

    # Project u-velocity: average fine u-values at the coarse x-face
    # For each coarse x-face, average the fine x-faces along z-direction
    # Left face of coarse cell (i,j) -> fine face at fi=1
    if 1 <= i <= size(state.u, 1)
        sum_u = zero(T)
        for fj in 1:nz_f
            sum_u += refined_state.u[1, fj]
        end
        state.u[i, j] = sum_u * inv_ratio
    end

    # Right face of coarse cell (i,j) -> fine face at fi=nx_f+1
    if 1 <= i+1 <= size(state.u, 1)
        sum_u = zero(T)
        for fj in 1:nz_f
            sum_u += refined_state.u[nx_f+1, fj]
        end
        state.u[i+1, j] = sum_u * inv_ratio
    end

    # Project v-velocity: average fine v-values at the coarse z-face
    # Bottom face of coarse cell (i,j) -> fine face at fj=1
    if 1 <= j <= size(state.v, 2)
        sum_v = zero(T)
        for fi in 1:nx_f
            sum_v += refined_state.v[fi, 1]
        end
        state.v[i, j] = sum_v * inv_ratio
    end

    # Top face of coarse cell (i,j) -> fine face at fj=nz_f+1
    if 1 <= j+1 <= size(state.v, 2)
        sum_v = zero(T)
        for fi in 1:nx_f
            sum_v += refined_state.v[fi, nz_f+1]
        end
        state.v[i, j+1] = sum_v * inv_ratio
    end

    # Project pressure: volume-weighted average of all fine cells
    sum_p = zero(T)
    for fj in 1:nz_f, fi in 1:nx_f
        sum_p += refined_state.p[fi, fj]
    end
    state.p[i, j] = sum_p * inv_ratio2
end

"""
    project_cell_3d!(state, i, j, k, refined_grid)

Conservative projection of a single 3D refined cell to coarse grid.
Uses volume-weighted averaging for pressure and face-averaged fluxes for velocities.
"""
function project_cell_3d!(state::SolutionState{T}, i::Int, j::Int, k::Int,
                           refined_grid::RefinedGrid{T}) where {T}
    # Get refined state for this cell
    refined_state = get(refined_grid.refined_states_3d, (i, j, k), nothing)
    if refined_state === nothing
        return
    end

    level = get(refined_grid.refined_cells_3d, (i, j, k), 0)
    if level == 0
        return
    end

    ratio = 2^level
    inv_ratio2 = one(T) / (ratio * ratio)   # For face averaging
    inv_ratio3 = inv_ratio2 / ratio          # For volume averaging

    # Fine grid dimensions
    nx_f = size(refined_state.p, 1)
    ny_f = size(refined_state.p, 2)
    nz_f = size(refined_state.p, 3)

    # Project u-velocity: average fine u-values at coarse x-faces
    # Left face (i,j,k) -> fine face at fi=1
    if 1 <= i <= size(state.u, 1)
        sum_u = zero(T)
        for fk in 1:nz_f, fj in 1:ny_f
            sum_u += refined_state.u[1, fj, fk]
        end
        state.u[i, j, k] = sum_u * inv_ratio2
    end

    # Right face (i+1,j,k) -> fine face at fi=nx_f+1
    if 1 <= i+1 <= size(state.u, 1)
        sum_u = zero(T)
        for fk in 1:nz_f, fj in 1:ny_f
            sum_u += refined_state.u[nx_f+1, fj, fk]
        end
        state.u[i+1, j, k] = sum_u * inv_ratio2
    end

    # Project v-velocity: average fine v-values at coarse y-faces
    # Front face (i,j,k) -> fine face at fj=1
    if 1 <= j <= size(state.v, 2)
        sum_v = zero(T)
        for fk in 1:nz_f, fi in 1:nx_f
            sum_v += refined_state.v[fi, 1, fk]
        end
        state.v[i, j, k] = sum_v * inv_ratio2
    end

    # Back face (i,j+1,k) -> fine face at fj=ny_f+1
    if 1 <= j+1 <= size(state.v, 2)
        sum_v = zero(T)
        for fk in 1:nz_f, fi in 1:nx_f
            sum_v += refined_state.v[fi, ny_f+1, fk]
        end
        state.v[i, j+1, k] = sum_v * inv_ratio2
    end

    # Project w-velocity: average fine w-values at coarse z-faces
    # Bottom face (i,j,k) -> fine face at fk=1
    if 1 <= k <= size(state.w, 3)
        sum_w = zero(T)
        for fj in 1:ny_f, fi in 1:nx_f
            sum_w += refined_state.w[fi, fj, 1]
        end
        state.w[i, j, k] = sum_w * inv_ratio2
    end

    # Top face (i,j,k+1) -> fine face at fk=nz_f+1
    if 1 <= k+1 <= size(state.w, 3)
        sum_w = zero(T)
        for fj in 1:ny_f, fi in 1:nx_f
            sum_w += refined_state.w[fi, fj, nz_f+1]
        end
        state.w[i, j, k+1] = sum_w * inv_ratio2
    end

    # Project pressure: volume-weighted average of all fine cells
    sum_p = zero(T)
    for fk in 1:nz_f, fj in 1:ny_f, fi in 1:nx_f
        sum_p += refined_state.p[fi, fj, fk]
    end
    state.p[i, j, k] = sum_p * inv_ratio3
end
