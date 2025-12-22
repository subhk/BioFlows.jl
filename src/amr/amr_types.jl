"""
    AMR Type Definitions for BioFlows.jl

Core data structures used by the adaptive mesh refinement system.
These types bridge the BioFlows Flow-based architecture with the AMR infrastructure.
"""

# Grid dimensionality enum
@enum GridType begin
    TwoDimensional
    ThreeDimensional
end

"""
    StaggeredGrid

Represents a staggered (MAC) grid for CFD computations.
Velocity components are stored at face centers, pressure at cell centers.

# Fields
- `grid_type`: Dimensionality (TwoDimensional or ThreeDimensional)
- `nx, ny, nz`: Number of cells in each direction (ny=1 for 2D)
- `dx, dy, dz`: Cell spacing in each direction
- `x, y, z`: Cell center coordinates
- `xf, yf, zf`: Face coordinates (staggered locations)
"""
struct StaggeredGrid{T<:AbstractFloat}
    grid_type::GridType
    nx::Int
    ny::Int
    nz::Int
    dx::T
    dy::T
    dz::T
    x::Vector{T}   # Cell center x-coordinates
    y::Vector{T}   # Cell center y-coordinates
    z::Vector{T}   # Cell center z-coordinates
    xf::Vector{T}  # Face x-coordinates (staggered)
    yf::Vector{T}  # Face y-coordinates (staggered)
    zf::Vector{T}  # Face z-coordinates (staggered)
end

"""
    StaggeredGrid(nx, nz, dx, dz; T=Float64)

Construct a 2D staggered grid in the XZ plane.
"""
function StaggeredGrid(nx::Int, nz::Int, dx::T, dz::T) where {T<:AbstractFloat}
    # Cell centers
    x = collect(T, (0.5:nx) .* dx)
    z = collect(T, (0.5:nz) .* dz)
    y = T[zero(T)]

    # Face locations (staggered)
    xf = collect(T, (0:nx) .* dx)
    zf = collect(T, (0:nz) .* dz)
    yf = T[zero(T)]

    StaggeredGrid{T}(TwoDimensional, nx, 1, nz, dx, one(T), dz, x, y, z, xf, yf, zf)
end

"""
    StaggeredGrid(nx, ny, nz, dx, dy, dz; T=Float64)

Construct a 3D staggered grid.
"""
function StaggeredGrid(nx::Int, ny::Int, nz::Int, dx::T, dy::T, dz::T) where {T<:AbstractFloat}
    # Cell centers
    x = collect(T, (0.5:nx) .* dx)
    y = collect(T, (0.5:ny) .* dy)
    z = collect(T, (0.5:nz) .* dz)

    # Face locations (staggered)
    xf = collect(T, (0:nx) .* dx)
    yf = collect(T, (0:ny) .* dy)
    zf = collect(T, (0:nz) .* dz)

    StaggeredGrid{T}(ThreeDimensional, nx, ny, nz, dx, dy, dz, x, y, z, xf, yf, zf)
end

"""
    SolutionState

Container for flow solution variables on a staggered grid.

# Fields
- `u`: x-velocity component (at x-faces)
- `v`: y-velocity component (at y-faces), or z-velocity in 2D XZ plane
- `w`: z-velocity component (at z-faces), only used in 3D
- `p`: Pressure (at cell centers)
"""
struct SolutionState{T<:AbstractFloat, A<:AbstractArray{T}}
    u::A
    v::A
    w::Union{A, Nothing}
    p::A
end

"""
    SolutionState(grid::StaggeredGrid; T=Float64, mem=Array)

Create an empty solution state for the given grid.
"""
function SolutionState(grid::StaggeredGrid{T}; mem=Array) where {T}
    if grid.grid_type == TwoDimensional
        u = mem{T}(undef, grid.nx + 1, grid.nz)      # x-faces
        v = mem{T}(undef, grid.nx, grid.nz + 1)      # z-faces (in XZ plane)
        p = mem{T}(undef, grid.nx, grid.nz)          # cell centers
        fill!(u, zero(T))
        fill!(v, zero(T))
        fill!(p, zero(T))
        SolutionState{T, typeof(u)}(u, v, nothing, p)
    else
        u = mem{T}(undef, grid.nx + 1, grid.ny, grid.nz)  # x-faces
        v = mem{T}(undef, grid.nx, grid.ny + 1, grid.nz)  # y-faces
        w = mem{T}(undef, grid.nx, grid.ny, grid.nz + 1)  # z-faces
        p = mem{T}(undef, grid.nx, grid.ny, grid.nz)      # cell centers
        fill!(u, zero(T))
        fill!(v, zero(T))
        fill!(w, zero(T))
        fill!(p, zero(T))
        SolutionState{T, typeof(u)}(u, v, w, p)
    end
end

"""
    is_2d(grid::StaggeredGrid)

Check if the grid is two-dimensional.
"""
is_2d(grid::StaggeredGrid) = grid.grid_type == TwoDimensional

"""
    is_3d(grid::StaggeredGrid)

Check if the grid is three-dimensional.
"""
is_3d(grid::StaggeredGrid) = grid.grid_type == ThreeDimensional

"""
    domain_size(grid::StaggeredGrid)

Return the physical domain size (Lx, Ly, Lz).
"""
function domain_size(grid::StaggeredGrid{T}) where {T}
    Lx = grid.nx * grid.dx
    Ly = grid.ny * grid.dy
    Lz = grid.nz * grid.dz
    return (Lx, Ly, Lz)
end

"""
    cell_volume(grid::StaggeredGrid)

Return the cell volume (dx * dy * dz).
"""
cell_volume(grid::StaggeredGrid) = grid.dx * grid.dy * grid.dz

"""
    RefinedGrid

Container for adaptive mesh refinement data.
Stores the base grid and tracking information for refined cells.

# Fields
- `base_grid`: The coarse/base level grid
- `refined_cells_2d/3d`: Dict mapping cell indices to refinement level
- `refined_grids_2d/3d`: Dict mapping cell indices to local refined StaggeredGrid
- `interpolation_weights_2d/3d`: Pre-computed interpolation weights
"""
mutable struct RefinedGrid{T<:AbstractFloat}
    base_grid::StaggeredGrid{T}
    refined_cells_2d::Dict{Tuple{Int,Int}, Int}
    refined_cells_3d::Dict{Tuple{Int,Int,Int}, Int}
    refined_grids_2d::Dict{Tuple{Int,Int}, StaggeredGrid{T}}
    refined_grids_3d::Dict{Tuple{Int,Int,Int}, StaggeredGrid{T}}
    interpolation_weights_2d::Dict{Tuple{Int,Int}, Vector{Tuple{Tuple{Int,Int}, T}}}
    interpolation_weights_3d::Dict{Tuple{Int,Int,Int}, Vector{Tuple{Tuple{Int,Int,Int}, T}}}
end

"""
    RefinedGrid(base_grid::StaggeredGrid)

Create an empty RefinedGrid from a base grid.
"""
function RefinedGrid(base_grid::StaggeredGrid{T}) where {T}
    RefinedGrid{T}(
        base_grid,
        Dict{Tuple{Int,Int}, Int}(),
        Dict{Tuple{Int,Int,Int}, Int}(),
        Dict{Tuple{Int,Int}, StaggeredGrid{T}}(),
        Dict{Tuple{Int,Int,Int}, StaggeredGrid{T}}(),
        Dict{Tuple{Int,Int}, Vector{Tuple{Tuple{Int,Int}, T}}}(),
        Dict{Tuple{Int,Int,Int}, Vector{Tuple{Tuple{Int,Int,Int}, T}}}()
    )
end

"""
    num_refined_cells(rg::RefinedGrid)

Return the total number of refined cells.
"""
function num_refined_cells(rg::RefinedGrid)
    if is_2d(rg.base_grid)
        return length(rg.refined_cells_2d)
    else
        return length(rg.refined_cells_3d)
    end
end

"""
    refinement_level(rg::RefinedGrid, i, j)
    refinement_level(rg::RefinedGrid, i, j, k)

Get the refinement level of a cell (0 = base grid, 1+ = refined).
"""
function refinement_level(rg::RefinedGrid, i::Int, j::Int)
    get(rg.refined_cells_2d, (i, j), 0)
end

function refinement_level(rg::RefinedGrid, i::Int, j::Int, k::Int)
    get(rg.refined_cells_3d, (i, j, k), 0)
end
