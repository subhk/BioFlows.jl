"""
    Refined Fields for AMR

Storage for velocity and other fields at refined resolution.
Enables full fine-grid velocity representation for accurate AMR simulations.
"""

"""
    RefinedVelocityPatch{T,N}

Stores velocity field for a single refined patch.

# Fields
- `u`: Velocity array at refined resolution (size: fine_dims..., N)
- `level`: Refinement level (1=2x, 2=4x, 3=8x)
- `anchor`: Coarse cell anchor (Flow indices including ghost offset)
- `coarse_extent`: Number of coarse cells covered in each direction
- `fine_dims`: Dimensions of fine grid (excluding ghost cells)
"""
struct RefinedVelocityPatch{T,N,A<:AbstractArray{T}}
    u::A                  # Velocity array: (fine_nx+2, fine_nz+2, N) for 2D
    level::Int            # Refinement level
    anchor::NTuple{N,Int} # Coarse cell anchor
    coarse_extent::NTuple{N,Int}  # Coarse cells covered
    fine_dims::NTuple{N,Int}      # Fine grid interior dimensions
end

"""
    RefinedVelocityPatch(anchor, coarse_extent, level, N, T; mem=Array)

Create a refined velocity patch.

# Arguments
- `anchor`: Coarse cell anchor position (Flow indices including ghost offset)
- `coarse_extent`: Number of coarse cells in each direction
- `level`: Refinement level (1, 2, or 3)
- `N`: Number of spatial dimensions
- `T`: Element type
- `mem`: Memory backend (Array for CPU, CuArray for GPU, etc.)
"""
function RefinedVelocityPatch(anchor::NTuple{D,Int}, coarse_extent::NTuple{D,Int},
                               level::Int, ::Val{D}, ::Type{T}; mem=Array) where {D,T}
    ratio = 2^level
    fine_dims = coarse_extent .* ratio
    # Add 2 ghost cells in each direction
    u_dims = (fine_dims .+ 2)..., D
    u = zeros(T, u_dims...) |> mem
    RefinedVelocityPatch{T,D,typeof(u)}(u, level, anchor, coarse_extent, fine_dims)
end

# Convenience constructors
RefinedVelocityPatch(anchor::NTuple{2,Int}, extent::NTuple{2,Int}, level::Int, T::Type=Float64; mem=Array) =
    RefinedVelocityPatch(anchor, extent, level, Val{2}(), T; mem=mem)
RefinedVelocityPatch(anchor::NTuple{3,Int}, extent::NTuple{3,Int}, level::Int, T::Type=Float64; mem=Array) =
    RefinedVelocityPatch(anchor, extent, level, Val{3}(), T; mem=mem)

"""
    refinement_ratio(patch::RefinedVelocityPatch)

Return the refinement ratio (2^level).
"""
refinement_ratio(patch::RefinedVelocityPatch) = 2^patch.level

"""
    fine_index_range(patch::RefinedVelocityPatch)

Return the CartesianIndices for the interior fine cells (excluding ghost cells).
"""
function fine_index_range(patch::RefinedVelocityPatch{T,N}) where {T,N}
    CartesianIndices(ntuple(i -> 2:patch.fine_dims[i]+1, N))
end

# Staggered-grid interpolation helpers (component-aware offsets).
@inline function _interp_velocity_2d(u_coarse::AbstractArray{T},
                                     ai::Int, aj::Int, ratio::Int,
                                     fi::Int, fj::Int, d::Int) where T
    shift_x = d == 1 ? T(0.5) : zero(T)
    shift_z = d == 2 ? T(0.5) : zero(T)

    xf = (T(fi) - T(1.5) - shift_x) / ratio
    zf = (T(fj) - T(1.5) - shift_z) / ratio

    ic = floor(Int, xf) + ai
    jc = floor(Int, zf) + aj
    wx = xf - floor(xf)
    wz = zf - floor(zf)

    nc_i, nc_j = size(u_coarse, 1), size(u_coarse, 2)
    ic = clamp(ic, 1, nc_i - 1)
    jc = clamp(jc, 1, nc_j - 1)
    wx = clamp(wx, zero(T), one(T))
    wz = clamp(wz, zero(T), one(T))

    v00 = u_coarse[ic, jc, d]
    v10 = u_coarse[ic+1, jc, d]
    v01 = u_coarse[ic, jc+1, d]
    v11 = u_coarse[ic+1, jc+1, d]

    return (one(T) - wx) * (one(T) - wz) * v00 +
           wx * (one(T) - wz) * v10 +
           (one(T) - wx) * wz * v01 +
           wx * wz * v11
end

@inline function _interp_velocity_3d(u_coarse::AbstractArray{T},
                                     ai::Int, aj::Int, ak::Int, ratio::Int,
                                     fi::Int, fj::Int, fk::Int, d::Int) where T
    shift_x = d == 1 ? T(0.5) : zero(T)
    shift_y = d == 2 ? T(0.5) : zero(T)
    shift_z = d == 3 ? T(0.5) : zero(T)

    xf = (T(fi) - T(1.5) - shift_x) / ratio
    yf = (T(fj) - T(1.5) - shift_y) / ratio
    zf = (T(fk) - T(1.5) - shift_z) / ratio

    ic = floor(Int, xf) + ai
    jc = floor(Int, yf) + aj
    kc = floor(Int, zf) + ak
    wx = xf - floor(xf)
    wy = yf - floor(yf)
    wz = zf - floor(zf)

    nc_i, nc_j, nc_k = size(u_coarse, 1), size(u_coarse, 2), size(u_coarse, 3)
    ic = clamp(ic, 1, nc_i - 1)
    jc = clamp(jc, 1, nc_j - 1)
    kc = clamp(kc, 1, nc_k - 1)
    wx = clamp(wx, zero(T), one(T))
    wy = clamp(wy, zero(T), one(T))
    wz = clamp(wz, zero(T), one(T))

    v000 = u_coarse[ic, jc, kc, d]
    v100 = u_coarse[ic+1, jc, kc, d]
    v010 = u_coarse[ic, jc+1, kc, d]
    v110 = u_coarse[ic+1, jc+1, kc, d]
    v001 = u_coarse[ic, jc, kc+1, d]
    v101 = u_coarse[ic+1, jc, kc+1, d]
    v011 = u_coarse[ic, jc+1, kc+1, d]
    v111 = u_coarse[ic+1, jc+1, kc+1, d]

    c00 = (one(T) - wx) * v000 + wx * v100
    c10 = (one(T) - wx) * v010 + wx * v110
    c01 = (one(T) - wx) * v001 + wx * v101
    c11 = (one(T) - wx) * v011 + wx * v111

    c0 = (one(T) - wy) * c00 + wy * c10
    c1 = (one(T) - wy) * c01 + wy * c11

    return (one(T) - wz) * c0 + wz * c1
end

"""
    coarse_to_fine_index(patch, I_coarse, offset)

Convert coarse cell index to fine cell index within patch.

# Arguments
- `patch`: RefinedVelocityPatch
- `I_coarse`: Coarse cell CartesianIndex (global)
- `offset`: Offset within the refined cell (1 to ratio in each direction)

# Returns
- Fine cell CartesianIndex (local to patch, with ghost offset)
"""
function coarse_to_fine_index(patch::RefinedVelocityPatch{T,N},
                               I_coarse::CartesianIndex{N},
                               offset::NTuple{N,Int}) where {T,N}
    ratio = refinement_ratio(patch)
    # Relative position in coarse grid
    rel = I_coarse.I .- patch.anchor .+ 1
    # Fine index = (rel-1)*ratio + offset + 1 (for ghost cell)
    fine = (rel .- 1) .* ratio .+ offset .+ 1
    CartesianIndex(fine)
end

"""
    fine_to_coarse_index(patch, I_fine)

Convert fine cell index to coarse cell index.

# Arguments
- `patch`: RefinedVelocityPatch
- `I_fine`: Fine cell CartesianIndex (local to patch, with ghost offset)

# Returns
- Coarse cell CartesianIndex (global)
"""
function fine_to_coarse_index(patch::RefinedVelocityPatch{T,N},
                               I_fine::CartesianIndex{N}) where {T,N}
    ratio = refinement_ratio(patch)
    # Remove ghost offset, compute coarse relative position
    fine_rel = I_fine.I .- 1  # Remove ghost offset
    coarse_rel = (fine_rel .- 1) .÷ ratio .+ 1
    # Add anchor offset
    coarse = coarse_rel .+ patch.anchor .- 1
    CartesianIndex(coarse)
end

"""
    RefinedVelocityField{T,N}

Collection of refined velocity patches for AMR simulation.

# Fields
- `patches`: Dictionary mapping anchor -> RefinedVelocityPatch
- `mem`: Memory backend (Array for CPU, CuArray for GPU)
"""
mutable struct RefinedVelocityField{T,N}
    patches::Dict{NTuple{N,Int}, RefinedVelocityPatch{T,N}}
    mem::Any  # Memory backend for GPU support
end

"""
    RefinedVelocityField(N, T; mem=Array)

Create an empty refined velocity field.

# Arguments
- `N`: Number of spatial dimensions (Val{2} or Val{3})
- `T`: Element type
- `mem`: Memory backend (Array for CPU, CuArray for GPU)
"""
RefinedVelocityField(::Val{N}, ::Type{T}; mem=Array) where {N,T} =
    RefinedVelocityField{T,N}(Dict{NTuple{N,Int}, RefinedVelocityPatch{T,N}}(), mem)

RefinedVelocityField(N::Int, T::Type=Float64; mem=Array) = RefinedVelocityField(Val{N}(), T; mem=mem)

# Convenience for 2D/3D
RefinedVelocityField2D(T::Type=Float64; mem=Array) = RefinedVelocityField(Val{2}(), T; mem=mem)
RefinedVelocityField3D(T::Type=Float64; mem=Array) = RefinedVelocityField(Val{3}(), T; mem=mem)

"""
    add_patch!(field, anchor, coarse_extent, level)

Add a refined velocity patch to the field.
Uses the memory backend stored in the field for GPU support.
"""
function add_patch!(field::RefinedVelocityField{T,N},
                    anchor::NTuple{N,Int},
                    coarse_extent::NTuple{N,Int},
                    level::Int) where {T,N}
    patch = RefinedVelocityPatch(anchor, coarse_extent, level, Val{N}(), T; mem=field.mem)
    field.patches[anchor] = patch
    return patch
end

"""
    remove_patch!(field, anchor)

Remove a refined velocity patch from the field.
"""
function remove_patch!(field::RefinedVelocityField{T,N}, anchor::NTuple{N,Int}) where {T,N}
    delete!(field.patches, anchor)
end

"""
    clear_patches!(field)

Remove all patches from the field.
"""
function clear_patches!(field::RefinedVelocityField)
    empty!(field.patches)
end

"""
    has_patch(field, anchor)

Check if a patch exists at the given anchor.
"""
has_patch(field::RefinedVelocityField{T,N}, anchor::NTuple{N,Int}) where {T,N} =
    haskey(field.patches, anchor)

"""
    get_patch(field, anchor)

Get the patch at the given anchor, or nothing if not found.
"""
get_patch(field::RefinedVelocityField{T,N}, anchor::NTuple{N,Int}) where {T,N} =
    get(field.patches, anchor, nothing)

"""
    num_patches(field)

Return the number of patches in the field.
"""
num_patches(field::RefinedVelocityField) = length(field.patches)

"""
    total_fine_cells(field)

Return the total number of fine cells across all patches.
"""
function total_fine_cells(field::RefinedVelocityField)
    total = 0
    for (_, patch) in field.patches
        total += prod(patch.fine_dims)
    end
    return total
end

"""
    interpolate_from_coarse!(patch, u_coarse, anchor)

Interpolate coarse velocity to fine patch using bilinear/trilinear interpolation.
Accounts for staggered velocity component offsets.
Fills the entire patch interior from the coarse grid.

GPU-compatible: Uses @loop macro for parallel execution.

# Arguments
- `patch`: RefinedVelocityPatch to fill
- `u_coarse`: Coarse velocity array (with ghost cells)
- `anchor`: Anchor position in coarse grid
"""
function interpolate_from_coarse!(patch::RefinedVelocityPatch{T,2},
                                   u_coarse::AbstractArray{T},
                                   anchor::NTuple{2,Int}) where T
    ratio = refinement_ratio(patch)
    ai, aj = anchor
    R = fine_index_range(patch)

    # Interpolate each velocity component using @loop (GPU-compatible)
    @loop patch.u[I,1] = _interp_velocity_2d(u_coarse, ai, aj, ratio, I[1], I[2], 1) over I ∈ R
    @loop patch.u[I,2] = _interp_velocity_2d(u_coarse, ai, aj, ratio, I[1], I[2], 2) over I ∈ R
end

function interpolate_from_coarse!(patch::RefinedVelocityPatch{T,3},
                                   u_coarse::AbstractArray{T},
                                   anchor::NTuple{3,Int}) where T
    ratio = refinement_ratio(patch)
    ai, aj, ak = anchor
    R = fine_index_range(patch)

    # Interpolate each velocity component using @loop (GPU-compatible)
    @loop patch.u[I,1] = _interp_velocity_3d(u_coarse, ai, aj, ak, ratio, I[1], I[2], I[3], 1) over I ∈ R
    @loop patch.u[I,2] = _interp_velocity_3d(u_coarse, ai, aj, ak, ratio, I[1], I[2], I[3], 2) over I ∈ R
    @loop patch.u[I,3] = _interp_velocity_3d(u_coarse, ai, aj, ak, ratio, I[1], I[2], I[3], 3) over I ∈ R
end

"""
    restrict_to_coarse!(u_coarse, patch, anchor)

Restrict fine velocity to coarse grid using volume-weighted averaging.
Updates the coarse velocity in cells covered by the patch.

GPU-compatible: Uses @loop with sum() for parallel reduction.

# Arguments
- `u_coarse`: Coarse velocity array to update
- `patch`: RefinedVelocityPatch with fine velocity
- `anchor`: Anchor position in coarse grid
"""
function restrict_to_coarse!(u_coarse::AbstractArray{T},
                              patch::RefinedVelocityPatch{T,2},
                              anchor::NTuple{2,Int}) where T
    ratio = refinement_ratio(patch)
    ai, aj = anchor
    inv_ratio = one(T) / ratio
    ext_i, ext_j = patch.coarse_extent

    # Iterate over coarse cells covered by the patch using @loop
    R_coarse = CartesianIndices((1:ext_i, 1:ext_j))

    # x-component: average across transverse fine faces (z-direction)
    @loop u_coarse[ai + I[1] - 1, aj + I[2] - 1, 1] =
        sum(@view patch.u[(I[1]-1)*ratio + 2, (I[2]-1)*ratio + 2 : I[2]*ratio + 1, 1]) * inv_ratio over I ∈ R_coarse

    # z-component: average across transverse fine faces (x-direction)
    @loop u_coarse[ai + I[1] - 1, aj + I[2] - 1, 2] =
        sum(@view patch.u[(I[1]-1)*ratio + 2 : I[1]*ratio + 1, (I[2]-1)*ratio + 2, 2]) * inv_ratio over I ∈ R_coarse
end

function restrict_to_coarse!(u_coarse::AbstractArray{T},
                              patch::RefinedVelocityPatch{T,3},
                              anchor::NTuple{3,Int}) where T
    ratio = refinement_ratio(patch)
    ai, aj, ak = anchor
    inv_ratio2 = one(T) / (ratio * ratio)
    ext_i, ext_j, ext_k = patch.coarse_extent

    # Iterate over coarse cells covered by the patch using @loop
    R_coarse = CartesianIndices((1:ext_i, 1:ext_j, 1:ext_k))

    # x-component: average across transverse faces (y-z plane)
    @loop u_coarse[ai + I[1] - 1, aj + I[2] - 1, ak + I[3] - 1, 1] =
        sum(@view patch.u[(I[1]-1)*ratio + 2,
                          (I[2]-1)*ratio + 2 : I[2]*ratio + 1,
                          (I[3]-1)*ratio + 2 : I[3]*ratio + 1, 1]) * inv_ratio2 over I ∈ R_coarse

    # y-component: average across transverse faces (x-z plane)
    @loop u_coarse[ai + I[1] - 1, aj + I[2] - 1, ak + I[3] - 1, 2] =
        sum(@view patch.u[(I[1]-1)*ratio + 2 : I[1]*ratio + 1,
                          (I[2]-1)*ratio + 2,
                          (I[3]-1)*ratio + 2 : I[3]*ratio + 1, 2]) * inv_ratio2 over I ∈ R_coarse

    # z-component: average across transverse faces (x-y plane)
    @loop u_coarse[ai + I[1] - 1, aj + I[2] - 1, ak + I[3] - 1, 3] =
        sum(@view patch.u[(I[1]-1)*ratio + 2 : I[1]*ratio + 1,
                          (I[2]-1)*ratio + 2 : I[2]*ratio + 1,
                          (I[3]-1)*ratio + 2, 3]) * inv_ratio2 over I ∈ R_coarse
end

"""
    fill_ghost_cells!(patch, u_coarse, anchor)

Fill patch ghost cells by interpolation from coarse grid.
Used for boundary conditions at patch edges.
Includes bounds checking to handle patches near domain boundaries.

GPU-compatible: Uses @loop macro for parallel execution on boundary faces.
"""
function fill_ghost_cells!(patch::RefinedVelocityPatch{T,2},
                            u_coarse::AbstractArray{T},
                            anchor::NTuple{2,Int}) where T
    ratio = refinement_ratio(patch)
    ai, aj = anchor
    nx, nz = patch.fine_dims

    # Left/right ghost faces - use 1D range along z
    R_lr = CartesianIndices((1:nz+2,))
    for d in 1:2
        @loop patch.u[1, I[1], d] = _interp_velocity_2d(u_coarse, ai, aj, ratio, 1, I[1], d) over I ∈ R_lr
        @loop patch.u[nx+2, I[1], d] = _interp_velocity_2d(u_coarse, ai, aj, ratio, nx+2, I[1], d) over I ∈ R_lr
    end

    # Bottom/top ghost faces - use 1D range along x
    R_bt = CartesianIndices((1:nx+2,))
    for d in 1:2
        @loop patch.u[I[1], 1, d] = _interp_velocity_2d(u_coarse, ai, aj, ratio, I[1], 1, d) over I ∈ R_bt
        @loop patch.u[I[1], nz+2, d] = _interp_velocity_2d(u_coarse, ai, aj, ratio, I[1], nz+2, d) over I ∈ R_bt
    end
end

"""
    fill_ghost_cells!(patch, u_coarse, anchor) - 3D version

Fill 3D patch ghost cells by interpolation from coarse grid.
Uses trilinear interpolation on each face.

GPU-compatible: Uses @loop macro for parallel execution on boundary faces.
"""
function fill_ghost_cells!(patch::RefinedVelocityPatch{T,3},
                            u_coarse::AbstractArray{T},
                            anchor::NTuple{3,Int}) where T
    ratio = refinement_ratio(patch)
    ai, aj, ak = anchor
    nx, ny, nz = patch.fine_dims

    # Left/right ghost faces (x = 1 and x = nx+2) - 2D range over y,z
    R_x = CartesianIndices((1:ny+2, 1:nz+2))
    for d in 1:3
        @loop patch.u[1, I[1], I[2], d] = _interp_velocity_3d(u_coarse, ai, aj, ak, ratio, 1, I[1], I[2], d) over I ∈ R_x
        @loop patch.u[nx+2, I[1], I[2], d] = _interp_velocity_3d(u_coarse, ai, aj, ak, ratio, nx+2, I[1], I[2], d) over I ∈ R_x
    end

    # Front/back ghost faces (y = 1 and y = ny+2) - 2D range over x,z
    R_y = CartesianIndices((1:nx+2, 1:nz+2))
    for d in 1:3
        @loop patch.u[I[1], 1, I[2], d] = _interp_velocity_3d(u_coarse, ai, aj, ak, ratio, I[1], 1, I[2], d) over I ∈ R_y
        @loop patch.u[I[1], ny+2, I[2], d] = _interp_velocity_3d(u_coarse, ai, aj, ak, ratio, I[1], ny+2, I[2], d) over I ∈ R_y
    end

    # Bottom/top ghost faces (z = 1 and z = nz+2) - 2D range over x,y
    R_z = CartesianIndices((1:nx+2, 1:ny+2))
    for d in 1:3
        @loop patch.u[I[1], I[2], 1, d] = _interp_velocity_3d(u_coarse, ai, aj, ak, ratio, I[1], I[2], 1, d) over I ∈ R_z
        @loop patch.u[I[1], I[2], nz+2, d] = _interp_velocity_3d(u_coarse, ai, aj, ak, ratio, I[1], I[2], nz+2, d) over I ∈ R_z
    end
end
