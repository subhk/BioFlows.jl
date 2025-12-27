# =============================================================================
# PATCH CREATION FOR AMR
# =============================================================================
# This file handles the creation and management of refined patches from
# cell-level refinement markers. The key steps are:
#
# 1. Cell marking: Mark cells that need refinement based on indicators
#    (e.g., vorticity magnitude, gradient magnitude, proximity to body)
#
# 2. Clustering: Group adjacent marked cells into connected components
#    using BFS/flood-fill algorithm
#
# 3. Bounding box: Compute rectangular bounding boxes for each cluster
#
# 4. Patch creation: Create PatchPoisson solvers for each bounding box
#
# Proper nesting ensures smooth transitions between refinement levels:
# - A level-n cell must be surrounded by level-(n-1) cells
# - This prevents abrupt jumps in resolution
#
# The algorithm supports:
# - Multiple refinement levels (1, 2, 3 = 2x, 4x, 8x)
# - 2D and 3D grids
# - Dynamic regridding during simulation
# =============================================================================

"""
    Patch Creation for AMR

Create PatchPoisson solvers from RefinedGrid cell markings.
Groups refined cells into rectangular patches for efficient solving.
"""

"""
    create_patches!(cp::CompositePoisson, rg::RefinedGrid, μ₀)

Create patches from a RefinedGrid's marked cells.
Clears existing patches and creates new ones.

# Arguments
- `cp`: CompositePoisson to populate with patches
- `rg`: RefinedGrid with cell refinement markers
- `μ₀`: Base grid coefficient array (flow.μ₀)

# Note
For flexible bodies, the μ₀ reference is stored in cp.μ₀_ref so that
patch coefficients can be re-interpolated when the body moves.
"""
function create_patches!(cp::CompositePoisson{T}, rg::RefinedGrid, μ₀::AbstractArray) where T
    # Clear existing patches
    clear_patches!(cp)

    # Store reference to μ₀ for flexible body coefficient updates
    set_μ₀_reference!(cp, μ₀)

    if is_2d(rg.base_grid)
        create_patches_2d!(cp, rg, μ₀)
    else
        create_patches_3d!(cp, rg, μ₀)
    end
end

"""
    create_patches_2d!(cp, rg, μ₀)

Create 2D patches from refined cell markers.
"""
function create_patches_2d!(cp::CompositePoisson{T}, rg::RefinedGrid, μ₀::AbstractArray) where T
    isempty(rg.refined_cells_2d) && return

    # Group cells by level
    cells_by_level = Dict{Int, Vector{Tuple{Int,Int}}}()
    for (cell, level) in rg.refined_cells_2d
        level = clamp(level, 1, cp.max_level)
        if !haskey(cells_by_level, level)
            cells_by_level[level] = Tuple{Int,Int}[]
        end
        push!(cells_by_level[level], cell)
    end

    # Create patches for each level
    for level in 1:cp.max_level
        haskey(cells_by_level, level) || continue
        cells = cells_by_level[level]

        # Cluster cells into rectangular patches
        clusters = cluster_cells_2d(cells)

        for cluster in clusters
            anchor, extent = bounding_box_2d(cluster)
            # Add padding if patch is too small (minimum 2x2 coarse cells)
            extent = max.(extent, (2, 2))

            # Clamp to grid bounds (flow indices include a ghost offset)
            nx, nz = rg.base_grid.nx + 1, rg.base_grid.nz + 1
            i_min = clamp(anchor[1], 2, nx)
            j_min = clamp(anchor[2], 2, nz)
            i_max = clamp(anchor[1] + extent[1] - 1, 2, nx)
            j_max = clamp(anchor[2] + extent[2] - 1, 2, nz)
            anchor = (i_min, j_min)
            extent = (max(i_max - i_min + 1, 2), max(j_max - j_min + 1, 2))

            add_patch!(cp, anchor, extent, level, μ₀)
        end
    end
end

"""
    create_patches_3d!(cp, rg, μ₀)

Create 3D patches from refined cell markers.
"""
function create_patches_3d!(cp::CompositePoisson{T}, rg::RefinedGrid, μ₀::AbstractArray) where T
    isempty(rg.refined_cells_3d) && return

    # Group cells by level
    cells_by_level = Dict{Int, Vector{Tuple{Int,Int,Int}}}()
    for (cell, level) in rg.refined_cells_3d
        level = clamp(level, 1, cp.max_level)
        if !haskey(cells_by_level, level)
            cells_by_level[level] = Tuple{Int,Int,Int}[]
        end
        push!(cells_by_level[level], cell)
    end

    # Create patches for each level
    for level in 1:cp.max_level
        haskey(cells_by_level, level) || continue
        cells = cells_by_level[level]

        # Cluster cells into rectangular patches
        clusters = cluster_cells_3d(cells)

        for cluster in clusters
            anchor, extent = bounding_box_3d(cluster)
            # Add padding if patch is too small (minimum 2x2x2 coarse cells)
            extent = max.(extent, (2, 2, 2))

            # Clamp to grid bounds (flow indices include a ghost offset)
            nx, ny, nz = rg.base_grid.nx + 1, rg.base_grid.ny + 1, rg.base_grid.nz + 1
            i_min = clamp(anchor[1], 2, nx)
            j_min = clamp(anchor[2], 2, ny)
            k_min = clamp(anchor[3], 2, nz)
            i_max = clamp(anchor[1] + extent[1] - 1, 2, nx)
            j_max = clamp(anchor[2] + extent[2] - 1, 2, ny)
            k_max = clamp(anchor[3] + extent[3] - 1, 2, nz)
            anchor = (i_min, j_min, k_min)
            extent = (max(i_max - i_min + 1, 2), max(j_max - j_min + 1, 2), max(k_max - k_min + 1, 2))

            add_patch_3d!(cp, anchor, extent, level, μ₀)
        end
    end
end

"""
    cluster_cells_2d(cells::Vector{Tuple{Int,Int}})

Cluster adjacent cells into groups for patch creation.
Uses connected component analysis.

# Returns
- Vector of cell clusters (each cluster is a vector of cell indices)
"""
function cluster_cells_2d(cells::Vector{Tuple{Int,Int}})
    isempty(cells) && return Vector{Vector{Tuple{Int,Int}}}()

    # Create a set for O(1) lookup
    cell_set = Set(cells)

    # Track which cells have been visited
    visited = Set{Tuple{Int,Int}}()
    clusters = Vector{Vector{Tuple{Int,Int}}}()

    for cell in cells
        cell in visited && continue

        # BFS to find connected component
        cluster = Tuple{Int,Int}[]
        queue = [cell]

        while !isempty(queue)
            current = popfirst!(queue)
            current in visited && continue
            push!(visited, current)
            push!(cluster, current)

            # Check 4-connected neighbors (can extend to 8-connected)
            i, j = current
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                neighbor = (i + di, j + dj)
                if neighbor in cell_set && !(neighbor in visited)
                    push!(queue, neighbor)
                end
            end
        end

        !isempty(cluster) && push!(clusters, cluster)
    end

    return clusters
end

"""
    bounding_box_2d(cells::Vector{Tuple{Int,Int}})

Compute bounding box for a cluster of cells.

# Returns
- `(anchor, extent)` where anchor is (i_min, j_min) and extent is (nx, nz)
"""
function bounding_box_2d(cells::Vector{Tuple{Int,Int}})
    i_min = minimum(c[1] for c in cells)
    i_max = maximum(c[1] for c in cells)
    j_min = minimum(c[2] for c in cells)
    j_max = maximum(c[2] for c in cells)

    anchor = (i_min, j_min)
    extent = (i_max - i_min + 1, j_max - j_min + 1)

    return anchor, extent
end

"""
    bounding_box_3d(cells::Vector{Tuple{Int,Int,Int}})

Compute bounding box for a cluster of 3D cells.
"""
function bounding_box_3d(cells::Vector{Tuple{Int,Int,Int}})
    i_min = minimum(c[1] for c in cells)
    i_max = maximum(c[1] for c in cells)
    j_min = minimum(c[2] for c in cells)
    j_max = maximum(c[2] for c in cells)
    k_min = minimum(c[3] for c in cells)
    k_max = maximum(c[3] for c in cells)

    anchor = (i_min, j_min, k_min)
    extent = (i_max - i_min + 1, j_max - j_min + 1, k_max - k_min + 1)

    return anchor, extent
end

"""
    cluster_cells_3d(cells::Vector{Tuple{Int,Int,Int}})

Cluster adjacent 3D cells into groups for patch creation.
Uses connected component analysis with 6-connectivity.

# Returns
- Vector of cell clusters (each cluster is a vector of cell indices)
"""
function cluster_cells_3d(cells::Vector{Tuple{Int,Int,Int}})
    isempty(cells) && return Vector{Vector{Tuple{Int,Int,Int}}}()

    # Create a set for O(1) lookup
    cell_set = Set(cells)

    # Track which cells have been visited
    visited = Set{Tuple{Int,Int,Int}}()
    clusters = Vector{Vector{Tuple{Int,Int,Int}}}()

    for cell in cells
        cell in visited && continue

        # BFS to find connected component
        cluster = Tuple{Int,Int,Int}[]
        queue = [cell]

        while !isempty(queue)
            current = popfirst!(queue)
            current in visited && continue
            push!(visited, current)
            push!(cluster, current)

            # Check 6-connected neighbors
            i, j, k = current
            for (di, dj, dk) in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1))
                neighbor = (i + di, j + dj, k + dk)
                if neighbor in cell_set && !(neighbor in visited)
                    push!(queue, neighbor)
                end
            end
        end

        !isempty(cluster) && push!(clusters, cluster)
    end

    return clusters
end

"""
    update_patches!(cp::CompositePoisson, rg::RefinedGrid, μ₀)

Update patches based on changed refinement markers.
More efficient than full recreation - only updates changed regions.

# Arguments
- `cp`: CompositePoisson
- `rg`: RefinedGrid with updated cell markers
- `μ₀`: Base grid coefficients
"""
function update_patches!(cp::CompositePoisson{T}, rg::RefinedGrid, μ₀::AbstractArray) where T
    # For now, just recreate all patches
    # TODO: Implement incremental update
    create_patches!(cp, rg, μ₀)
end

"""
    should_regrid(rg::RefinedGrid, indicators::Dict, threshold::Real)

Check if regridding is needed based on refinement indicators.

# Arguments
- `rg`: Current RefinedGrid
- `indicators`: Cell -> indicator value mapping
- `threshold`: Refinement threshold

# Returns
- `true` if regridding should be performed
"""
function should_regrid(rg::RefinedGrid, indicators::Dict, threshold::Real)
    # Check for cells that should be refined but aren't
    for (cell, value) in indicators
        current_level = is_2d(rg.base_grid) ?
            get(rg.refined_cells_2d, cell, 0) :
            get(rg.refined_cells_3d, cell, 0)

        if value > threshold && current_level == 0
            return true
        end
    end

    # Check for cells that should be coarsened
    refined_cells = is_2d(rg.base_grid) ? rg.refined_cells_2d : rg.refined_cells_3d
    for (cell, level) in refined_cells
        value = get(indicators, cell, 0.0)
        if value < threshold * 0.5 && level > 0
            return true
        end
    end

    return false
end

"""
    mark_cells_for_refinement!(rg::RefinedGrid, indicators::Dict, threshold, max_level)

Mark cells for refinement based on indicator values.

# Arguments
- `rg`: RefinedGrid to update
- `indicators`: Cell -> indicator value mapping
- `threshold`: Refinement threshold
- `max_level`: Maximum refinement level
"""
function mark_cells_for_refinement!(rg::RefinedGrid{T}, indicators::Dict,
                                     threshold::Real, max_level::Int) where T
    if is_2d(rg.base_grid)
        empty!(rg.refined_cells_2d)
        for (cell, value) in indicators
            cell isa Tuple{Int,Int} || continue
            if value > threshold
                # Determine level based on indicator strength
                level = min(ceil(Int, log2(value / threshold + 1)), max_level)
                rg.refined_cells_2d[cell] = level
            end
        end
    else
        empty!(rg.refined_cells_3d)
        for (cell, value) in indicators
            cell isa Tuple{Int,Int,Int} || continue
            if value > threshold
                level = min(ceil(Int, log2(value / threshold + 1)), max_level)
                rg.refined_cells_3d[cell] = level
            end
        end
    end
end

"""
    ensure_proper_nesting!(rg::RefinedGrid, buffer::Int=1)

Ensure proper nesting of refinement levels.
A level-n cell must be surrounded by level-(n-1) cells.

# Arguments
- `rg`: RefinedGrid to fix
- `buffer`: Number of buffer cells between levels
"""
function ensure_proper_nesting!(rg::RefinedGrid, buffer::Int=1)
    if is_2d(rg.base_grid)
        ensure_proper_nesting_2d!(rg, buffer)
    else
        ensure_proper_nesting_3d!(rg, buffer)
    end
end

function ensure_proper_nesting_2d!(rg::RefinedGrid, buffer::Int)
    nx, nz = rg.base_grid.nx + 1, rg.base_grid.nz + 1
    max_level = maximum(values(rg.refined_cells_2d); init=0)
    max_level == 0 && return

    # Process levels from finest to coarsest
    for level in max_level:-1:2
        # Find all cells at this level
        cells_at_level = [(c, l) for (c, l) in rg.refined_cells_2d if l == level]

        for (cell, _) in cells_at_level
            i, j = cell
            # Ensure surrounding cells are at least level-1
            for di in -buffer:buffer, dj in -buffer:buffer
                (di == 0 && dj == 0) && continue
                ni, nj = i + di, j + dj
                (ni < 2 || ni > nx || nj < 2 || nj > nz) && continue

                neighbor = (ni, nj)
                current = get(rg.refined_cells_2d, neighbor, 0)
                if current < level - 1
                    rg.refined_cells_2d[neighbor] = level - 1
                end
            end
        end
    end
end

function ensure_proper_nesting_3d!(rg::RefinedGrid, buffer::Int)
    nx, ny, nz = rg.base_grid.nx + 1, rg.base_grid.ny + 1, rg.base_grid.nz + 1
    max_level = maximum(values(rg.refined_cells_3d); init=0)
    max_level == 0 && return

    for level in max_level:-1:2
        cells_at_level = [(c, l) for (c, l) in rg.refined_cells_3d if l == level]

        for (cell, _) in cells_at_level
            i, j, k = cell
            for di in -buffer:buffer, dj in -buffer:buffer, dk in -buffer:buffer
                (di == 0 && dj == 0 && dk == 0) && continue
                ni, nj, nk = i + di, j + dj, k + dk
                (ni < 2 || ni > nx || nj < 2 || nj > ny || nk < 2 || nk > nz) && continue

                neighbor = (ni, nj, nk)
                current = get(rg.refined_cells_3d, neighbor, 0)
                if current < level - 1
                    rg.refined_cells_3d[neighbor] = level - 1
                end
            end
        end
    end
end

"""
    patches_overlap(p1_anchor, p1_extent, p2_anchor, p2_extent)

Check if two patches overlap.
"""
function patches_overlap(p1_anchor::NTuple{N,Int}, p1_extent::NTuple{N,Int},
                         p2_anchor::NTuple{N,Int}, p2_extent::NTuple{N,Int}) where N
    for d in 1:N
        p1_min, p1_max = p1_anchor[d], p1_anchor[d] + p1_extent[d] - 1
        p2_min, p2_max = p2_anchor[d], p2_anchor[d] + p2_extent[d] - 1

        # No overlap if one is entirely before the other
        if p1_max < p2_min || p2_max < p1_min
            return false
        end
    end
    return true
end

"""
    merge_overlapping_patches(patches)

Merge patches that overlap into larger patches.
Returns a new list of non-overlapping patches.
"""
function merge_overlapping_patches(patches::Vector{Tuple{NTuple{2,Int}, NTuple{2,Int}}})
    isempty(patches) && return patches

    merged = Vector{Tuple{NTuple{2,Int}, NTuple{2,Int}}}()

    for (anchor, extent) in patches
        overlap_idx = 0
        for (idx, (m_anchor, m_extent)) in enumerate(merged)
            if patches_overlap(anchor, extent, m_anchor, m_extent)
                overlap_idx = idx
                break
            end
        end

        if overlap_idx > 0
            # Merge with existing patch
            m_anchor, m_extent = merged[overlap_idx]

            # Compute bounding box of both patches
            new_anchor = min.(anchor, m_anchor)
            end1 = anchor .+ extent .- 1
            end2 = m_anchor .+ m_extent .- 1
            new_end = max.(end1, end2)
            new_extent = new_end .- new_anchor .+ 1

            merged[overlap_idx] = (new_anchor, new_extent)
        else
            push!(merged, (anchor, extent))
        end
    end

    return merged
end
