# =============================================================================
# COMPOSITE POISSON SOLVER FOR AMR
# =============================================================================
# CompositePoisson manages the pressure Poisson equation across multiple grid
# levels in an AMR simulation. The key challenge is maintaining:
#
# 1. Flux conservation: The coarse-fine interface must conserve mass
# 2. Solution accuracy: The fine grid should improve local accuracy
# 3. Efficiency: Most work should be on the coarse grid
#
# Structure:
# - base: MultiLevelPoisson for the coarse (base) grid
# - patches: Dict of PatchPoisson solvers for refined regions
# - refined_velocity: Velocity field at fine resolution
#
# The solver uses defect correction:
# 1. V-cycle on base grid (captures smooth/global errors)
# 2. Local smoothing on patches (captures fine-scale errors)
# 3. Flux correction at interfaces (ensures conservation)
# 4. Restrict corrections back to base grid
# =============================================================================

"""
    CompositePoisson - Composite Grid Poisson Solver

Main solver for AMR simulations. Manages base grid MultiLevelPoisson
plus refined patches with proper coarse-fine coupling.
"""

"""
    CompositePoisson{T,S,V}

Composite grid Poisson solver managing base grid and refined patches.

# Fields
- `base`: Base grid MultiLevelPoisson solver
- `patches`: 2D patch solvers (anchor -> PatchPoisson)
- `patches_3d`: 3D patch solvers
- `refined_velocity`: Velocity field at refined resolution
- `refinement_ratio`: Base ratio (always 2 for standard AMR)
- `max_level`: Maximum refinement level (1, 2, or 3)
- `n`: Iteration count history
- `perdir`: Periodic directions tuple
"""
mutable struct CompositePoisson{T,S<:AbstractArray{T},V<:AbstractArray{T}} <: AbstractPoisson{T,S,V}
    base::MultiLevelPoisson{T,S,V}
    patches::Dict{Tuple{Int,Int}, PatchPoisson{T,S,V}}
    patches_3d::Dict{Tuple{Int,Int,Int}, PatchPoisson{T,S,V}}
    refined_velocity::RefinedVelocityField{T,2}
    refinement_ratio::Int
    max_level::Int
    n::Vector{Int16}
    perdir::NTuple

    # Forward x, L, z to base for compatibility
    x::S
    L::V
    z::S
end

"""
    CompositePoisson(base::MultiLevelPoisson; max_level=3)

Create a CompositePoisson wrapping an existing MultiLevelPoisson.

# Arguments
- `base`: Existing MultiLevelPoisson solver
- `max_level`: Maximum refinement level (1=2x, 2=4x, 3=8x)
"""
function CompositePoisson(base::MultiLevelPoisson{T,S,V};
                          max_level::Int=3) where {T,S,V}
    CompositePoisson{T,S,V}(
        base,
        Dict{Tuple{Int,Int}, PatchPoisson{T,S,V}}(),
        Dict{Tuple{Int,Int,Int}, PatchPoisson{T,S,V}}(),
        RefinedVelocityField(Val{2}(), T),
        2,  # Always 2:1 ratio
        max_level,
        Int16[],
        base.perdir,
        base.x,
        base.L,
        base.z
    )
end

"""
    CompositePoisson(x, L, z; perdir=(), max_level=3)

Create a CompositePoisson from arrays (creates MultiLevelPoisson internally).
"""
function CompositePoisson(x::AbstractArray{T}, L::AbstractArray{T}, z::AbstractArray{T};
                          perdir=(), max_level::Int=3) where T
    base = MultiLevelPoisson(x, L, z; perdir)
    CompositePoisson(base; max_level)
end

# AbstractPoisson interface compatibility
Base.getproperty(cp::CompositePoisson, s::Symbol) = begin
    if s in (:base, :patches, :patches_3d, :refined_velocity, :refinement_ratio,
             :max_level, :n, :perdir, :x, :L, :z)
        getfield(cp, s)
    else
        getfield(cp, s)
    end
end

"""
    num_patches(cp::CompositePoisson)

Return total number of patches.
"""
num_patches(cp::CompositePoisson) = length(cp.patches) + length(cp.patches_3d)

"""
    has_patches(cp::CompositePoisson)

Check if any patches exist.
"""
has_patches(cp::CompositePoisson) = !isempty(cp.patches) || !isempty(cp.patches_3d)

"""
    clear_patches!(cp::CompositePoisson)

Remove all patches.
"""
function clear_patches!(cp::CompositePoisson)
    empty!(cp.patches)
    empty!(cp.patches_3d)
    clear_patches!(cp.refined_velocity)
end

"""
    add_patch!(cp, anchor, extent, level, μ₀)

Add a 2D patch at the given anchor.

# Arguments
- `cp`: CompositePoisson
- `anchor`: Coarse cell anchor (i, j)
- `extent`: Coarse cells covered (nx, nz)
- `level`: Refinement level
- `μ₀`: Coarse coefficient array (flow.μ₀)

# Returns
- Created PatchPoisson
"""
function add_patch!(cp::CompositePoisson{T},
                    anchor::Tuple{Int,Int},
                    extent::Tuple{Int,Int},
                    level::Int,
                    μ₀::AbstractArray) where T
    level = clamp(level, 1, cp.max_level)

    # Create PatchPoisson
    patch = PatchPoisson(anchor, extent, level, μ₀, T)
    cp.patches[anchor] = patch

    # Create corresponding velocity patch
    add_patch!(cp.refined_velocity, anchor, extent, level)

    return patch
end

"""
    remove_patch!(cp, anchor)

Remove a 2D patch.
"""
function remove_patch!(cp::CompositePoisson, anchor::Tuple{Int,Int})
    delete!(cp.patches, anchor)
    remove_patch!(cp.refined_velocity, anchor)
end

"""
    get_patch(cp, anchor)

Get a patch by anchor, or nothing if not found.
"""
get_patch(cp::CompositePoisson, anchor::Tuple{Int,Int}) =
    get(cp.patches, anchor, nothing)

"""
    patches_at_level(cp, level)

Iterator over patches at a specific refinement level.
"""
function patches_at_level(cp::CompositePoisson, level::Int)
    Iterators.filter(((k, v),) -> v.level == level, cp.patches)
end

"""
    update!(cp::CompositePoisson)

Update all solver components after coefficient changes.
"""
function update!(cp::CompositePoisson)
    update!(cp.base)
    for (_, patch) in cp.patches
        update_coefficients!(patch)
    end
end

"""
    mult!(cp::CompositePoisson, x)

Matrix-vector multiplication (delegates to base).
"""
mult!(cp::CompositePoisson, x) = mult!(cp.base, x)

"""
    L₂(cp::CompositePoisson)

L₂ norm of residual (from base level).
"""
L₂(cp::CompositePoisson) = L₂(cp.base.levels[1])

"""
    L∞(cp::CompositePoisson)

L∞ norm of residual (from base level).
"""
L∞(cp::CompositePoisson) = L∞(cp.base.levels[1])

"""
    total_refined_cells(cp::CompositePoisson)

Total number of refined cells across all patches.
"""
function total_refined_cells(cp::CompositePoisson)
    total = 0
    for (_, patch) in cp.patches
        total += prod(patch.fine_dims)
    end
    return total
end

"""
    print_summary(cp::CompositePoisson)

Print summary of composite solver state.
"""
function print_summary(cp::CompositePoisson)
    println("CompositePoisson Summary:")
    println("  Base grid levels: ", length(cp.base.levels))
    println("  Max refinement level: ", cp.max_level)
    println("  Number of 2D patches: ", length(cp.patches))
    println("  Total refined cells: ", total_refined_cells(cp))
    if !isempty(cp.n)
        println("  Last iteration count: ", cp.n[end])
    end
end
