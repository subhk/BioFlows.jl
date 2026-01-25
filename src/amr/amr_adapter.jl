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

GPU-compatible: Uses @loop macro and array views for parallel execution.
"""

# =============================================================================
# GPU-COMPATIBLE INTERPOLATION HELPERS
# =============================================================================
# These inline functions perform bilinear/trilinear interpolation and are
# designed to be called from @loop kernels.

"""
    _bilinear_interp_u_2d(arr, i, j, ratio, fi, fj, T)

Bilinear interpolation for u-velocity (staggered in x) at fine index (fi, fj).
"""
@inline function _bilinear_interp_u_2d(arr::AbstractArray{T}, i::Int, j::Int,
                                        ratio::Int, fi::Int, fj::Int) where T
    inv_ratio = one(T) / ratio
    # x-face staggered: face positions at 0, dx, 2dx, ...
    xf = (T(fi) - one(T)) * inv_ratio
    # z-direction is cell-centered
    zf = (T(fj) - T(0.5)) * inv_ratio - T(0.5)

    nx_c, nz_c = size(arr, 1), size(arr, 2)
    ic = clamp(i + floor(Int, xf), 1, nx_c)
    jc = clamp(j + floor(Int, zf), 1, nz_c)
    wx = clamp(xf - floor(xf), zero(T), one(T))
    wz = clamp(zf - floor(zf), zero(T), one(T))

    ic1 = clamp(ic + 1, 1, nx_c)
    jc1 = clamp(jc + 1, 1, nz_c)

    return (one(T) - wx) * (one(T) - wz) * arr[ic, jc] +
           wx * (one(T) - wz) * arr[ic1, jc] +
           (one(T) - wx) * wz * arr[ic, jc1] +
           wx * wz * arr[ic1, jc1]
end

"""
    _bilinear_interp_v_2d(arr, i, j, ratio, fi, fj, T)

Bilinear interpolation for v-velocity (staggered in z) at fine index (fi, fj).
"""
@inline function _bilinear_interp_v_2d(arr::AbstractArray{T}, i::Int, j::Int,
                                        ratio::Int, fi::Int, fj::Int) where T
    inv_ratio = one(T) / ratio
    # x-direction is cell-centered
    xf = (T(fi) - T(0.5)) * inv_ratio - T(0.5)
    # z-face staggered: face positions at 0, dz, 2dz, ...
    zf = (T(fj) - one(T)) * inv_ratio

    nx_c, nz_c = size(arr, 1), size(arr, 2)
    ic = clamp(i + floor(Int, xf), 1, nx_c)
    jc = clamp(j + floor(Int, zf), 1, nz_c)
    wx = clamp(xf - floor(xf), zero(T), one(T))
    wz = clamp(zf - floor(zf), zero(T), one(T))

    ic1 = clamp(ic + 1, 1, nx_c)
    jc1 = clamp(jc + 1, 1, nz_c)

    return (one(T) - wx) * (one(T) - wz) * arr[ic, jc] +
           wx * (one(T) - wz) * arr[ic1, jc] +
           (one(T) - wx) * wz * arr[ic, jc1] +
           wx * wz * arr[ic1, jc1]
end

"""
    _bilinear_interp_p_2d(arr, i, j, ratio, fi, fj, T)

Bilinear interpolation for pressure (cell-centered) at fine index (fi, fj).
"""
@inline function _bilinear_interp_p_2d(arr::AbstractArray{T}, i::Int, j::Int,
                                        ratio::Int, fi::Int, fj::Int) where T
    inv_ratio = one(T) / ratio
    # Both directions cell-centered
    xf = (T(fi) - T(0.5)) * inv_ratio - T(0.5)
    zf = (T(fj) - T(0.5)) * inv_ratio - T(0.5)

    nx_c, nz_c = size(arr, 1), size(arr, 2)
    ic = clamp(i + floor(Int, xf), 1, nx_c)
    jc = clamp(j + floor(Int, zf), 1, nz_c)
    wx = clamp(xf - floor(xf), zero(T), one(T))
    wz = clamp(zf - floor(zf), zero(T), one(T))

    ic1 = clamp(ic + 1, 1, nx_c)
    jc1 = clamp(jc + 1, 1, nz_c)

    return (one(T) - wx) * (one(T) - wz) * arr[ic, jc] +
           wx * (one(T) - wz) * arr[ic1, jc] +
           (one(T) - wx) * wz * arr[ic, jc1] +
           wx * wz * arr[ic1, jc1]
end

"""
    _trilinear_interp(arr, i, j, k, ratio, fi, fj, fk, stagger, T)

Trilinear interpolation at fine index (fi, fj, fk) with stagger offset.
`stagger` is a tuple (sx, sy, sz) where 1 means face-staggered, 0 means cell-centered.
"""
@inline function _trilinear_interp(arr::AbstractArray{T}, i::Int, j::Int, k::Int,
                                    ratio::Int, fi::Int, fj::Int, fk::Int,
                                    stagger::NTuple{3,Int}) where T
    inv_ratio = one(T) / ratio
    sx, sy, sz = stagger

    # Compute fine position: face-staggered uses (f-1)/ratio, cell-centered uses (f-0.5)/ratio - 0.5
    xf = sx == 1 ? (T(fi) - one(T)) * inv_ratio : (T(fi) - T(0.5)) * inv_ratio - T(0.5)
    yf = sy == 1 ? (T(fj) - one(T)) * inv_ratio : (T(fj) - T(0.5)) * inv_ratio - T(0.5)
    zf = sz == 1 ? (T(fk) - one(T)) * inv_ratio : (T(fk) - T(0.5)) * inv_ratio - T(0.5)

    nx_c, ny_c, nz_c = size(arr, 1), size(arr, 2), size(arr, 3)
    ic = clamp(i + floor(Int, xf), 1, nx_c - 1)
    jc = clamp(j + floor(Int, yf), 1, ny_c - 1)
    kc = clamp(k + floor(Int, zf), 1, nz_c - 1)

    wx = clamp(xf - floor(xf), zero(T), one(T))
    wy = clamp(yf - floor(yf), zero(T), one(T))
    wz = clamp(zf - floor(zf), zero(T), one(T))

    # Trilinear interpolation
    c00 = (one(T) - wx) * arr[ic, jc, kc] + wx * arr[ic+1, jc, kc]
    c10 = (one(T) - wx) * arr[ic, jc+1, kc] + wx * arr[ic+1, jc+1, kc]
    c01 = (one(T) - wx) * arr[ic, jc, kc+1] + wx * arr[ic+1, jc, kc+1]
    c11 = (one(T) - wx) * arr[ic, jc+1, kc+1] + wx * arr[ic+1, jc+1, kc+1]

    c0 = (one(T) - wy) * c00 + wy * c10
    c1 = (one(T) - wy) * c01 + wy * c11

    return (one(T) - wz) * c0 + wz * c1
end

# =============================================================================
# ADAPTER TYPES
# =============================================================================

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

GPU-compatible: Uses array views and broadcast assignment.
"""
function flow_to_solution_state(flow::Flow{N,T}) where {N,T}
    dims = size(flow.p)

    if N == 2
        nx = dims[1] - 2
        nz = dims[2] - 2

        # Extract velocity components from staggered array using views (GPU-compatible)
        # flow.u has shape (nx+2, nz+2, 2) with ghost cells
        # u-velocity at x-faces: need indices 2:nx+2 in x, 2:nz+1 in z, component 1
        # v-velocity at z-faces: need indices 2:nx+1 in x, 2:nz+2 in z, component 2
        u = similar(flow.p, nx + 1, nz)
        v = similar(flow.p, nx, nz + 1)
        p = similar(flow.p, nx, nz)

        # Copy using broadcast assignment with views (GPU-compatible)
        u .= @view flow.u[2:nx+2, 2:nz+1, 1]
        v .= @view flow.u[2:nx+1, 2:nz+2, 2]
        p .= @view flow.p[2:nx+1, 2:nz+1]

        return SolutionState{T, typeof(u)}(u, v, nothing, p)

    else  # N == 3
        nx = dims[1] - 2
        ny = dims[2] - 2
        nz = dims[3] - 2

        u = similar(flow.p, nx + 1, ny, nz)
        v = similar(flow.p, nx, ny + 1, nz)
        w = similar(flow.p, nx, ny, nz + 1)
        p = similar(flow.p, nx, ny, nz)

        # Copy using broadcast assignment with views (GPU-compatible)
        u .= @view flow.u[2:nx+2, 2:ny+1, 2:nz+1, 1]
        v .= @view flow.u[2:nx+1, 2:ny+2, 2:nz+1, 2]
        w .= @view flow.u[2:nx+1, 2:ny+1, 2:nz+2, 3]
        p .= @view flow.p[2:nx+1, 2:ny+1, 2:nz+1]

        return SolutionState{T, typeof(u)}(u, v, w, p)
    end
end

flow_to_solution_state(adapter::FlowToGridAdapter) = flow_to_solution_state(adapter.flow)

"""
    update_flow_from_state!(flow::Flow, state::SolutionState)

Copy solution state back into the Flow struct.
Updates interior values only (ghost cells unchanged).

GPU-compatible: Uses array views and broadcast assignment.
"""
function update_flow_from_state!(flow::Flow{N,T}, state::SolutionState) where {N,T}
    dims = size(flow.p)

    if N == 2
        nx = dims[1] - 2
        nz = dims[2] - 2

        # Copy using broadcast assignment with views (GPU-compatible)
        @view(flow.u[2:nx+2, 2:nz+1, 1]) .= state.u
        @view(flow.u[2:nx+1, 2:nz+2, 2]) .= state.v
        @view(flow.p[2:nx+1, 2:nz+1]) .= state.p

    else  # N == 3
        nx = dims[1] - 2
        ny = dims[2] - 2
        nz = dims[3] - 2

        # Copy using broadcast assignment with views (GPU-compatible)
        @view(flow.u[2:nx+2, 2:ny+1, 2:nz+1, 1]) .= state.u
        @view(flow.u[2:nx+1, 2:ny+2, 2:nz+1, 2]) .= state.v
        @view(flow.u[2:nx+1, 2:ny+1, 2:nz+2, 3]) .= state.w
        @view(flow.p[2:nx+1, 2:ny+1, 2:nz+1]) .= state.p
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

GPU-compatible: Uses @loop macro for parallel execution on fine grid.
"""
function interpolate_cell_2d!(state::SolutionState{T}, i::Int, j::Int,
                               local_grid::StaggeredGrid{T}, refined_grid::RefinedGrid{T}) where {T}
    level = get(refined_grid.refined_cells_2d, (i, j), 0)
    level == 0 && return

    ratio = 2^level  # refinement ratio

    # Create or get refined solution state for this cell
    if !haskey(refined_grid.refined_states_2d, (i, j))
        refined_grid.refined_states_2d[(i, j)] = SolutionState(local_grid)
    end
    refined_state = refined_grid.refined_states_2d[(i, j)]

    # Fine grid dimensions
    nx_f = local_grid.nx
    nz_f = local_grid.nz

    # Interpolate u-velocity (at x-faces, staggered in x) using @loop
    R_u = CartesianIndices((1:(nx_f+1), 1:nz_f))
    @loop refined_state.u[I] = _bilinear_interp_u_2d(state.u, i, j, ratio, I[1], I[2]) over I ∈ R_u

    # Interpolate v-velocity (at z-faces, staggered in z) using @loop
    R_v = CartesianIndices((1:nx_f, 1:(nz_f+1)))
    @loop refined_state.v[I] = _bilinear_interp_v_2d(state.v, i, j, ratio, I[1], I[2]) over I ∈ R_v

    # Interpolate pressure (at cell centers) using @loop
    R_p = CartesianIndices((1:nx_f, 1:nz_f))
    @loop refined_state.p[I] = _bilinear_interp_p_2d(state.p, i, j, ratio, I[1], I[2]) over I ∈ R_p
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

GPU-compatible: Uses @loop macro for parallel execution on fine grid.
"""
function interpolate_cell_3d!(state::SolutionState{T}, i::Int, j::Int, k::Int,
                            local_grid::StaggeredGrid{T},
                            refined_grid::RefinedGrid{T}) where {T}

    level = get(refined_grid.refined_cells_3d, (i, j, k), 0)
    level == 0 && return

    ratio = 2^level

    # Create or get refined solution state
    if !haskey(refined_grid.refined_states_3d, (i, j, k))
        refined_grid.refined_states_3d[(i, j, k)] = SolutionState(local_grid)
    end
    refined_state = refined_grid.refined_states_3d[(i, j, k)]

    # Fine grid dimensions
    nx_f = local_grid.nx
    ny_f = local_grid.ny
    nz_f = local_grid.nz

    # Interpolate u-velocity (at x-faces): x is face-staggered (1), y and z are cell-centered (0)
    R_u = CartesianIndices((1:(nx_f+1), 1:ny_f, 1:nz_f))
    @loop refined_state.u[I] = _trilinear_interp(state.u, i, j, k, ratio, I[1], I[2], I[3], (1, 0, 0)) over I ∈ R_u

    # Interpolate v-velocity (at y-faces): y is face-staggered (1), x and z are cell-centered (0)
    R_v = CartesianIndices((1:nx_f, 1:(ny_f+1), 1:nz_f))
    @loop refined_state.v[I] = _trilinear_interp(state.v, i, j, k, ratio, I[1], I[2], I[3], (0, 1, 0)) over I ∈ R_v

    # Interpolate w-velocity (at z-faces): z is face-staggered (1), x and y are cell-centered (0)
    R_w = CartesianIndices((1:nx_f, 1:ny_f, 1:(nz_f+1)))
    @loop refined_state.w[I] = _trilinear_interp(state.w, i, j, k, ratio, I[1], I[2], I[3], (0, 0, 1)) over I ∈ R_w

    # Interpolate pressure (at cell centers): all cell-centered (0, 0, 0)
    R_p = CartesianIndices((1:nx_f, 1:ny_f, 1:nz_f))
    @loop refined_state.p[I] = _trilinear_interp(state.p, i, j, k, ratio, I[1], I[2], I[3], (0, 0, 0)) over I ∈ R_p
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

GPU-compatible: Uses sum() with array views for parallel reduction.
"""
function project_cell_2d!(state::SolutionState{T}, i::Int, j::Int,
                           refined_grid::RefinedGrid{T}) where {T}
    # Get refined state for this cell
    refined_state = get(refined_grid.refined_states_2d, (i, j), nothing)
    refined_state === nothing && return

    level = get(refined_grid.refined_cells_2d, (i, j), 0)
    level == 0 && return

    ratio = 2^level
    inv_ratio = one(T) / ratio
    inv_ratio2 = inv_ratio * inv_ratio  # For area averaging

    # Fine grid dimensions
    nx_f = size(refined_state.p, 1)
    nz_f = size(refined_state.p, 2)

    # Project u-velocity: average fine u-values at the coarse x-face using sum (GPU-compatible)
    # Left face of coarse cell (i,j) -> fine face at fi=1
    if 1 <= i <= size(state.u, 1)
        state.u[i, j] = sum(@view refined_state.u[1, 1:nz_f]) * inv_ratio
    end

    # Right face of coarse cell (i,j) -> fine face at fi=nx_f+1
    if 1 <= i+1 <= size(state.u, 1)
        state.u[i+1, j] = sum(@view refined_state.u[nx_f+1, 1:nz_f]) * inv_ratio
    end

    # Project v-velocity: average fine v-values at the coarse z-face using sum (GPU-compatible)
    # Bottom face of coarse cell (i,j) -> fine face at fj=1
    if 1 <= j <= size(state.v, 2)
        state.v[i, j] = sum(@view refined_state.v[1:nx_f, 1]) * inv_ratio
    end

    # Top face of coarse cell (i,j) -> fine face at fj=nz_f+1
    if 1 <= j+1 <= size(state.v, 2)
        state.v[i, j+1] = sum(@view refined_state.v[1:nx_f, nz_f+1]) * inv_ratio
    end

    # Project pressure: volume-weighted average of all fine cells using sum (GPU-compatible)
    state.p[i, j] = sum(@view refined_state.p[1:nx_f, 1:nz_f]) * inv_ratio2
end

"""
    project_cell_3d!(state, i, j, k, refined_grid)

Conservative projection of a single 3D refined cell to coarse grid.
Uses volume-weighted averaging for pressure and face-averaged fluxes for velocities.

GPU-compatible: Uses sum() with array views for parallel reduction.
"""
function project_cell_3d!(state::SolutionState{T}, i::Int, j::Int, k::Int,
                           refined_grid::RefinedGrid{T}) where {T}
    # Get refined state for this cell
    refined_state = get(refined_grid.refined_states_3d, (i, j, k), nothing)
    refined_state === nothing && return

    level = get(refined_grid.refined_cells_3d, (i, j, k), 0)
    level == 0 && return

    ratio = 2^level
    inv_ratio2 = one(T) / (ratio * ratio)   # For face averaging
    inv_ratio3 = inv_ratio2 / ratio          # For volume averaging

    # Fine grid dimensions
    nx_f = size(refined_state.p, 1)
    ny_f = size(refined_state.p, 2)
    nz_f = size(refined_state.p, 3)

    # Project u-velocity: average fine u-values at coarse x-faces using sum (GPU-compatible)
    # Left face (i,j,k) -> fine face at fi=1
    if 1 <= i <= size(state.u, 1)
        state.u[i, j, k] = sum(@view refined_state.u[1, 1:ny_f, 1:nz_f]) * inv_ratio2
    end

    # Right face (i+1,j,k) -> fine face at fi=nx_f+1
    if 1 <= i+1 <= size(state.u, 1)
        state.u[i+1, j, k] = sum(@view refined_state.u[nx_f+1, 1:ny_f, 1:nz_f]) * inv_ratio2
    end

    # Project v-velocity: average fine v-values at coarse y-faces using sum (GPU-compatible)
    # Front face (i,j,k) -> fine face at fj=1
    if 1 <= j <= size(state.v, 2)
        state.v[i, j, k] = sum(@view refined_state.v[1:nx_f, 1, 1:nz_f]) * inv_ratio2
    end

    # Back face (i,j+1,k) -> fine face at fj=ny_f+1
    if 1 <= j+1 <= size(state.v, 2)
        state.v[i, j+1, k] = sum(@view refined_state.v[1:nx_f, ny_f+1, 1:nz_f]) * inv_ratio2
    end

    # Project w-velocity: average fine w-values at coarse z-faces using sum (GPU-compatible)
    # Bottom face (i,j,k) -> fine face at fk=1
    if 1 <= k <= size(state.w, 3)
        state.w[i, j, k] = sum(@view refined_state.w[1:nx_f, 1:ny_f, 1]) * inv_ratio2
    end

    # Top face (i,j,k+1) -> fine face at fk=nz_f+1
    if 1 <= k+1 <= size(state.w, 3)
        state.w[i, j, k+1] = sum(@view refined_state.w[1:nx_f, 1:ny_f, nz_f+1]) * inv_ratio2
    end

    # Project pressure: volume-weighted average of all fine cells using sum (GPU-compatible)
    state.p[i, j, k] = sum(@view refined_state.p[1:nx_f, 1:ny_f, 1:nz_f]) * inv_ratio3
end
