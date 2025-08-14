struct StaggeredGrid{T<:Real} <: AbstractGrid
    nx::Int
    ny::Int
    nz::Int
    Lx::T
    Ly::T
    Lz::T
    dx::T
    dy::T
    dz::T
    x::Vector{T}      # Cell centers x-coordinates
    y::Vector{T}      # Cell centers y-coordinates
    z::Vector{T}      # Cell centers z-coordinates
    xu::Vector{T}     # u-velocity grid points
    yv::Vector{T}     # v-velocity grid points
    zw::Vector{T}     # w-velocity grid points
    grid_type::GridType
    refined_cells::Vector{Tuple{Int,Int}}  # For adaptive refinement
    refinement_level::Matrix{Int}
end

function StaggeredGrid2D(nx::Int, nz::Int, Lx::T, Lz::T; 
                        origin_x::T=zero(T), origin_z::T=zero(T)) where T<:Real
    dx = Lx / nx
    dz = Lz / nz
    
    # Cell centers (XZ plane: x-horizontal, z-vertical)
    x = origin_x .+ (0.5:nx-0.5) * dx
    z = origin_z .+ (0.5:nz-0.5) * dz
    
    # Staggered grids for velocities (u: x-direction, w: z-direction)
    xu = origin_x .+ (0:nx) * dx      # u-velocity points
    zw = origin_z .+ (0:nz) * dz      # w-velocity points
    
    StaggeredGrid{T}(
        nx, 0, nz,
        Lx, zero(T), Lz,
        dx, zero(T), dz,
        collect(x), T[], collect(z),
        collect(xu), T[], collect(zw),
        TwoDimensional,
        Tuple{Int,Int}[],
        ones(Int, nx, nz)
    )
end

function StaggeredGrid3D(nx::Int, ny::Int, nz::Int, Lx::T, Ly::T, Lz::T;
                        origin_x::T=zero(T), origin_y::T=zero(T), origin_z::T=zero(T)) where T<:Real
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz
    
    # Cell centers
    x = origin_x .+ (0.5:nx-0.5) * dx
    y = origin_y .+ (0.5:ny-0.5) * dy
    z = origin_z .+ (0.5:nz-0.5) * dz
    
    # Staggered grids for velocities
    xu = origin_x .+ (0:nx) * dx
    yv = origin_y .+ (0:ny) * dy
    zw = origin_z .+ (0:nz) * dz
    
    StaggeredGrid{T}(
        nx, ny, nz,
        Lx, Ly, Lz,
        dx, dy, dz,
        collect(x), collect(y), collect(z),
        collect(xu), collect(yv), collect(zw),
        ThreeDimensional,
        Tuple{Int,Int}[],
        ones(Int, nx, ny)
    )
end

function is_2d(grid::StaggeredGrid)
    return grid.grid_type == TwoDimensional
end

function is_3d(grid::StaggeredGrid)
    return grid.grid_type == ThreeDimensional
end

function grid_size(grid::StaggeredGrid)
    if grid.grid_type == TwoDimensional
        return (grid.nx, grid.nz)  # XZ plane
    else
        return (grid.nx, grid.ny, grid.nz)
    end
end