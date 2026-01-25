# =============================================================================
# PATCH POISSON SOLVER
# =============================================================================
# PatchPoisson is a local Poisson solver for a single refined patch.
# It mirrors the structure of the base Poisson solver but operates on
# a smaller, finer grid with boundary data interpolated from the coarse grid.
#
# Boundary conditions:
# - At interior interfaces (patch edge not at domain boundary):
#   Use Dirichlet BC with pressure interpolated from coarse grid
# - At domain boundaries:
#   Use Neumann BC (zero gradient) matching the base solver
#
# Matrix structure is identical to base Poisson:
#   A = L + D + L'   (symmetric, negative definite)
# where L contains off-diagonal coefficients and D is the diagonal.
#
# The coefficients L are initialized by interpolating the coarse μ₀
# (BDIM volume fraction) to fine grid locations.
# =============================================================================

"""
    PatchPoisson - Local Poisson Solver for Refined Patches

A Poisson solver for individual refined patches in AMR.
Follows the same structure as the base Poisson solver but operates
on local refined grids with boundary data from coarse grid.
"""

"""
    PatchPoisson{T,S,V}

Poisson solver for a single refined patch.
Inherits the matrix structure A = L + D + L' from the base solver.

# Fields
- `L`, `D`, `iD`, `x`, `ϵ`, `r`, `z`: Standard Poisson arrays (see Poisson.jl)
- `n`: Iteration count history
- `level`: Refinement level (1=2x, 2=4x, 3=8x)
- `anchor`: Coarse cell anchor (bottom-left corner in coarse grid)
- `coarse_extent`: Number of coarse cells covered
- `fine_dims`: Fine grid interior dimensions
- `Δx`: Grid spacing (relative to coarse = 1, so Δx = 1/ratio)

Note: Uses the same "unit spacing" convention as the base Poisson solver.
The L coefficients are NOT scaled by ratio² - the unit spacing convention
is consistent throughout and the Δx factors cancel in the solve/correct cycle.
"""
struct PatchPoisson{T,S<:AbstractArray{T},V<:AbstractArray{T}} <: AbstractPoisson{T,S,V}
    # Standard Poisson fields
    L :: V      # Lower diagonal coefficients (unit spacing convention)
    D :: S      # Diagonal coefficients
    iD :: S     # 1/Diagonal
    x :: S      # Solution (pressure)
    ϵ :: S      # Increment/error
    r :: S      # Residual
    z :: S      # Source term
    n :: Vector{Int16}  # Iteration count

    # Patch-specific fields
    level :: Int                    # Refinement level
    anchor :: NTuple{2,Int}         # Coarse anchor (2D)
    coarse_extent :: NTuple{2,Int}  # Coarse cells covered
    fine_dims :: NTuple{2,Int}      # Fine interior dimensions
    Δx :: T                         # Grid spacing (1/ratio relative to coarse)
end

"""
    PatchPoisson(anchor, coarse_extent, level, μ₀_coarse, T)

Create a PatchPoisson solver for a 2D refined patch.

# Arguments
- `anchor`: Coarse cell anchor position (1-indexed, global)
- `coarse_extent`: Number of coarse cells in (x, z)
- `level`: Refinement level (1, 2, or 3)
- `μ₀_coarse`: Coarse grid μ₀ coefficient array
- `T`: Element type (Float32 or Float64)

Uses the same "unit spacing" convention as the base Poisson solver.
The L coefficients are NOT scaled by ratio² - this is consistent with
the unit spacing convention used throughout the solver.
"""
function PatchPoisson(anchor::NTuple{2,Int},
                      coarse_extent::NTuple{2,Int},
                      level::Int,
                      μ₀_coarse::AbstractArray{Tc},
                      ::Type{T}=Float64;
                      mem=Array) where {Tc,T}
    ratio = 2^level
    fine_dims = coarse_extent .* ratio
    # Fine grid with ghost cells
    nx, nz = fine_dims
    Ng = (nx + 2, nz + 2)

    # Grid spacing relative to coarse (coarse = 1)
    Δx = one(T) / ratio

    # Create arrays (use mem for GPU support)
    x = zeros(T, Ng) |> mem
    z = zeros(T, Ng) |> mem
    r = zeros(T, Ng) |> mem
    ϵ = zeros(T, Ng) |> mem
    D = zeros(T, Ng) |> mem
    iD = zeros(T, Ng) |> mem
    L = ones(T, Ng..., 2) |> mem

    # Initialize L from coarse μ₀ (interpolated)
    # Uses unit spacing convention - NO ratio² scaling
    initialize_patch_coefficients!(L, μ₀_coarse, anchor, coarse_extent, ratio)

    # Compute diagonal
    patch_set_diag!(D, iD, L, Ng)

    PatchPoisson{T, typeof(x), typeof(L)}(
        L, D, iD, x, ϵ, r, z, Int16[],
        level, anchor, coarse_extent, fine_dims, Δx
    )
end

# Helper for bilinear interpolation of coefficients (GPU-compatible)
@inline function _interp_coeff_2d(μ₀_coarse::AbstractArray, I::CartesianIndex{2},
                                   ai::Int, aj::Int, ratio::Int, d::Int, ::Type{T},
                                   nc_i::Int, nc_j::Int) where T
    fi, fj = I.I
    inv_ratio = one(T) / ratio
    xf = (T(fi) - T(1.5)) * inv_ratio
    zf = (T(fj) - T(1.5)) * inv_ratio

    ic = clamp(floor(Int, xf) + ai, 1, nc_i - 1)
    jc = clamp(floor(Int, zf) + aj, 1, nc_j - 1)
    wx = clamp(xf - floor(xf), zero(T), one(T))
    wz = clamp(zf - floor(zf), zero(T), one(T))

    if ic >= 1 && ic < nc_i && jc >= 1 && jc < nc_j
        v00 = T(μ₀_coarse[ic, jc, d])
        v10 = T(μ₀_coarse[ic+1, jc, d])
        v01 = T(μ₀_coarse[ic, jc+1, d])
        v11 = T(μ₀_coarse[ic+1, jc+1, d])
        return (1-wx)*(1-wz)*v00 + wx*(1-wz)*v10 + (1-wx)*wz*v01 + wx*wz*v11
    else
        return one(T)
    end
end

"""
    initialize_patch_coefficients!(L, μ₀_coarse, anchor, extent, ratio)

Initialize patch coefficient array L by interpolating from coarse μ₀.
The coefficient at fine faces is interpolated from surrounding coarse values.
GPU-compatible via @loop.
"""
function initialize_patch_coefficients!(L::AbstractArray{T},
                                         μ₀_coarse::AbstractArray,
                                         anchor::NTuple{2,Int},
                                         extent::NTuple{2,Int},
                                         ratio::Int) where T
    ai, aj = anchor
    nx, nz = size(L, 1) - 2, size(L, 2) - 2
    nc_i, nc_j = size(μ₀_coarse, 1), size(μ₀_coarse, 2)

    R = CartesianIndices((1:nx+2, 1:nz+2))
    @loop L[I, 1] = _interp_coeff_2d(μ₀_coarse, I, ai, aj, ratio, 1, T, nc_i, nc_j) over I ∈ R
    @loop L[I, 2] = _interp_coeff_2d(μ₀_coarse, I, ai, aj, ratio, 2, T, nc_i, nc_j) over I ∈ R
end

# Helper for computing diagonal entry (GPU-compatible)
@inline function _patch_diag_entry_2d(L::AbstractArray{T}, I::CartesianIndex{2}) where T
    i, j = I.I
    s = zero(T)
    s -= L[i, j, 1] + L[i+1, j, 1]  # x-direction
    s -= L[i, j, 2] + L[i, j+1, 2]  # z-direction
    return s
end

"""
    patch_set_diag!(D, iD, L, Ng)

Compute diagonal and inverse diagonal for patch Poisson matrix.
Same formula as base Poisson: D[I] = -Σᵢ(L[I,i] + L[I+δ(i),i])
GPU-compatible via @loop.
"""
function patch_set_diag!(D::AbstractArray{T}, iD::AbstractArray{T},
                          L::AbstractArray{T}, Ng::NTuple{2,Int}) where T
    R = CartesianIndices((2:Ng[1]-1, 2:Ng[2]-1))
    @loop (s = _patch_diag_entry_2d(L, I);
           D[I] = s;
           iD[I] = abs2(s) < 2eps(T) ? zero(T) : inv(s)) over I ∈ R
end

"""
    refinement_ratio(patch::PatchPoisson)

Return the refinement ratio for this patch.
"""
refinement_ratio(patch::PatchPoisson) = 2^patch.level

"""
    inside(patch::PatchPoisson)

Return CartesianIndices for interior cells (excluding ghost cells).
"""
inside(patch::PatchPoisson) = CartesianIndices((2:patch.fine_dims[1]+1, 2:patch.fine_dims[2]+1))

"""
    patch_mult!(patch::PatchPoisson, x)

Matrix-vector multiplication for patch: z = A*x
"""
function patch_mult!(patch::PatchPoisson{T}, x::AbstractArray{T}) where T
    L, D, z = patch.L, patch.D, patch.z
    fill!(z, zero(T))
    R = inside(patch)
    @loop z[I] = x[I]*D[I] + x[I-δ(1,I)]*L[I,1] + x[I+δ(1,I)]*L[I+δ(1,I),1] +
                 x[I-δ(2,I)]*L[I,2] + x[I+δ(2,I)]*L[I+δ(2,I),2] over I ∈ R
    return z
end

"""
    patch_residual!(patch::PatchPoisson)

Compute residual r = z - A*x for the patch.
Sets r[I] = 0 where iD[I] = 0 (boundary/solid cells).
Also enforces global solvability by subtracting mean residual.
"""
function patch_residual!(patch::PatchPoisson{T}) where T
    L, D, x, r, z, iD = patch.L, patch.D, patch.x, patch.r, patch.z, patch.iD
    R = inside(patch)

    # Compute residual: r = z - A*x (GPU-compatible via @loop)
    @loop r[I] = ifelse(iD[I] == zero(T), zero(T),
        z[I] - (x[I]*D[I] + x[I-δ(1,I)]*L[I,1] + x[I+δ(1,I)]*L[I+δ(1,I),1] +
                x[I-δ(2,I)]*L[I,2] + x[I+δ(2,I)]*L[I+δ(2,I),2])) over I ∈ R

    # Enforce global solvability using GPU-compatible reductions
    r_inside = @view r[R]
    iD_inside = @view iD[R]
    count = sum(x -> x != zero(T) ? 1 : 0, iD_inside)
    if count > 0
        sum_r = sum(r_inside)
        mean_r = sum_r / count
        if abs(mean_r) > 2eps(T)
            @loop r[I] = r[I] - mean_r over I ∈ R
        end
    end
end

"""
    patch_increment!(patch::PatchPoisson)

Apply increment: x += ϵ, r -= A*ϵ
"""
function patch_increment!(patch::PatchPoisson{T}) where T
    L, D, x, ϵ, r = patch.L, patch.D, patch.x, patch.ϵ, patch.r
    R = inside(patch)
    # Compute A*ϵ and update r, x (GPU-compatible via @loop)
    # Note: inline computation required - @loop macro doesn't support local variable definitions
    @loop (r[I] -= ϵ[I]*D[I] + ϵ[I-δ(1,I)]*L[I,1] + ϵ[I+δ(1,I)]*L[I+δ(1,I),1] +
                   ϵ[I-δ(2,I)]*L[I,2] + ϵ[I+δ(2,I)]*L[I+δ(2,I),2];
           x[I] += ϵ[I]) over I ∈ R
end

"""
    patch_jacobi!(patch::PatchPoisson; it=1)

Jacobi smoother for patch.
"""
function patch_jacobi!(patch::PatchPoisson{T}; it::Int=1) where T
    R = inside(patch)
    for _ in 1:it
        @loop patch.ϵ[I] = patch.r[I] * patch.iD[I] over I ∈ R
        patch_increment!(patch)
    end
end

"""
    patch_L₂(patch::PatchPoisson)

Compute L₂ norm of residual (GPU-compatible via sum).
"""
function patch_L₂(patch::PatchPoisson{T}) where T
    R = inside(patch)
    return sum(abs2, @view patch.r[R])
end

"""
    patch_L∞(patch::PatchPoisson)

Compute L∞ norm of residual (GPU-compatible via maximum).
"""
function patch_L∞(patch::PatchPoisson{T}) where T
    R = inside(patch)
    return maximum(abs, @view patch.r[R])
end

"""
    patch_pcg!(patch::PatchPoisson; it=6)

Preconditioned Conjugate Gradient smoother for patch.
Uses Jacobi preconditioning. GPU-compatible via @loop and broadcast reductions.
"""
function patch_pcg!(patch::PatchPoisson{T}; it::Int=6) where T
    x, r, ϵ, z, iD = patch.x, patch.r, patch.ϵ, patch.z, patch.iD
    L, D = patch.L, patch.D
    R = inside(patch)

    # z = M⁻¹r (Jacobi preconditioner), ϵ = z (search direction)
    @loop (z[I] = r[I] * iD[I]; ϵ[I] = z[I]) over I ∈ R

    # rho = r·z (GPU-compatible dot product)
    r_inside = @view r[R]
    z_inside = @view z[R]
    rho = sum(r_inside .* z_inside)
    abs(rho) < 10eps(T) && return

    for iter in 1:it
        # z = A*ϵ (reusing z array)
        @loop z[I] = ϵ[I]*D[I] + ϵ[I-δ(1,I)]*L[I,1] + ϵ[I+δ(1,I)]*L[I+δ(1,I),1] +
                     ϵ[I-δ(2,I)]*L[I,2] + ϵ[I+δ(2,I)]*L[I+δ(2,I),2] over I ∈ R

        # alpha = rho / (z·ϵ)
        ϵ_inside = @view ϵ[R]
        z_inside = @view z[R]
        zϵ = sum(z_inside .* ϵ_inside)
        abs(zϵ) < 10eps(T) && return
        alpha = rho / zϵ
        (!isfinite(alpha) || abs(alpha) < 1e-2 || abs(alpha) > 1e2) && return

        # x += alpha*ϵ, r -= alpha*z
        @loop (x[I] = x[I] + alpha * ϵ[I]; r[I] = r[I] - alpha * z[I]) over I ∈ R

        iter == it && return

        # z = M⁻¹r
        @loop z[I] = r[I] * iD[I] over I ∈ R

        # rho2 = r·z
        r_inside = @view r[R]
        z_inside = @view z[R]
        rho2 = sum(r_inside .* z_inside)
        abs(rho2) < 10eps(T) && return

        # beta = rho2/rho
        beta = rho2 / rho

        # ϵ = z + beta*ϵ
        @loop ϵ[I] = z[I] + beta * ϵ[I] over I ∈ R

        rho = rho2
    end
end

"""
    patch_smooth!(patch::PatchPoisson)

Default smoother for patch (PCG).
"""
patch_smooth!(patch::PatchPoisson) = patch_pcg!(patch)

"""
    patch_solver!(patch::PatchPoisson; tol=1e-4, itmx=100)

Iterative solver for patch.
"""
function patch_solver!(patch::PatchPoisson{T}; tol::T=T(1e-4), itmx::Int=100) where T
    patch_residual!(patch)
    r2 = patch_L₂(patch)

    np = 0
    while np < itmx
        patch_smooth!(patch)
        r2 = patch_L₂(patch)
        np += 1
        r2 < tol && break
    end

    push!(patch.n, Int16(np))
end

# Helper for bilinear interpolation from coarse pressure (GPU-compatible)
@inline function _bilinear_interp_p(p_coarse::AbstractArray{T}, fi::Int, fj::Int,
                                     ai::Int, aj::Int, ratio::Int,
                                     nc_i::Int, nc_j::Int) where T
    inv_ratio = one(T) / ratio
    x_coarse = (T(fi) - T(1.5)) * inv_ratio + ai
    z_coarse = (T(fj) - T(1.5)) * inv_ratio + aj

    ic = clamp(floor(Int, x_coarse), 1, nc_i - 1)
    jc = clamp(floor(Int, z_coarse), 1, nc_j - 1)

    wx = clamp(x_coarse - T(ic), zero(T), one(T))
    wz = clamp(z_coarse - T(jc), zero(T), one(T))

    v00 = p_coarse[ic, jc]
    v10 = p_coarse[ic+1, jc]
    v01 = p_coarse[ic, jc+1]
    v11 = p_coarse[ic+1, jc+1]

    return (1-wx)*(1-wz)*v00 + wx*(1-wz)*v10 + (1-wx)*wz*v01 + wx*wz*v11
end

"""
    set_patch_boundary!(patch, p_coarse, anchor)

Set patch ghost cell values from coarse pressure solution.
Uses 2D bilinear interpolation for Dirichlet BCs at patch edges.
Falls back to Neumann BC (zero gradient) at domain boundaries.
GPU-compatible via @loop.

The fine grid has cells 1..nx+2 where:
- Cell 1 is the left ghost
- Cells 2..nx+1 are interior
- Cell nx+2 is the right ghost

The mapping to coarse coordinates:
- Fine cell center fi has position (fi - 1.5) in fine cell units from patch origin
- In coarse cell units: (fi - 1.5) / ratio
- Adding anchor gives global coarse position
"""
function set_patch_boundary!(patch::PatchPoisson{T},
                              p_coarse::AbstractArray{T},
                              anchor::NTuple{2,Int}) where T
    ratio = refinement_ratio(patch)
    ai, aj = anchor
    nx, nz = patch.fine_dims
    nc_i, nc_j = size(p_coarse, 1), size(p_coarse, 2)
    x = patch.x

    # Left boundary (i = 1) - ghost cell to the left of interior
    R_left = CartesianIndices((1:nz+2,))
    if ai > 2  # Interior interface - use Dirichlet from coarse
        @loop x[1, I[1]] = _bilinear_interp_p(p_coarse, 1, I[1], ai, aj, ratio, nc_i, nc_j) over I ∈ R_left
    else  # Domain boundary - use Neumann BC (zero gradient)
        @loop x[1, I[1]] = x[2, I[1]] over I ∈ R_left
    end

    # Right boundary (i = nx+2) - ghost cell to the right of interior
    right_coarse = ai + patch.coarse_extent[1]
    if right_coarse < nc_i  # Interior interface - use Dirichlet
        @loop x[nx+2, I[1]] = _bilinear_interp_p(p_coarse, nx+2, I[1], ai, aj, ratio, nc_i, nc_j) over I ∈ R_left
    else  # Domain boundary - use Neumann BC
        @loop x[nx+2, I[1]] = x[nx+1, I[1]] over I ∈ R_left
    end

    # Bottom boundary (j = 1) - ghost cell below interior
    R_bottom = CartesianIndices((1:nx+2,))
    if aj > 2  # Interior interface - use Dirichlet
        @loop x[I[1], 1] = _bilinear_interp_p(p_coarse, I[1], 1, ai, aj, ratio, nc_i, nc_j) over I ∈ R_bottom
    else  # Domain boundary - use Neumann BC
        @loop x[I[1], 1] = x[I[1], 2] over I ∈ R_bottom
    end

    # Top boundary (j = nz+2) - ghost cell above interior
    top_coarse = aj + patch.coarse_extent[2]
    if top_coarse < nc_j  # Interior interface - use Dirichlet
        @loop x[I[1], nz+2] = _bilinear_interp_p(p_coarse, I[1], nz+2, ai, aj, ratio, nc_i, nc_j) over I ∈ R_bottom
    else  # Domain boundary - use Neumann BC
        @loop x[I[1], nz+2] = x[I[1], nz+1] over I ∈ R_bottom
    end
end

"""
    update_coefficients!(patch::PatchPoisson)

Recompute diagonal after coefficient update.
"""
function update_coefficients!(patch::PatchPoisson)
    Ng = (patch.fine_dims[1] + 2, patch.fine_dims[2] + 2)
    patch_set_diag!(patch.D, patch.iD, patch.L, Ng)
end

# =============================================================================
# 3D PATCH POISSON SOLVER
# =============================================================================

"""
    PatchPoisson3D{T,S,V}

Poisson solver for a single 3D refined patch.
Inherits the matrix structure A = L + D + L' from the base solver.

# Fields
- `L`, `D`, `iD`, `x`, `ϵ`, `r`, `z`: Standard Poisson arrays
- `n`: Iteration count history
- `level`: Refinement level (1=2x, 2=4x, 3=8x)
- `anchor`: Coarse cell anchor (3D)
- `coarse_extent`: Number of coarse cells covered
- `fine_dims`: Fine grid interior dimensions
- `Δx`: Grid spacing (relative to coarse = 1, so Δx = 1/ratio)

Note: Uses the same "unit spacing" convention as the base Poisson solver.
The L coefficients are NOT scaled by ratio² - the unit spacing convention
is consistent throughout and the Δx factors cancel in the solve/correct cycle.
"""
struct PatchPoisson3D{T,S<:AbstractArray{T},V<:AbstractArray{T}} <: AbstractPoisson{T,S,V}
    # Standard Poisson fields
    L :: V      # Lower diagonal coefficients (unit spacing convention)
    D :: S      # Diagonal coefficients
    iD :: S     # 1/Diagonal
    x :: S      # Solution (pressure)
    ϵ :: S      # Increment/error
    r :: S      # Residual
    z :: S      # Source term
    n :: Vector{Int16}  # Iteration count

    # Patch-specific fields
    level :: Int                    # Refinement level
    anchor :: NTuple{3,Int}         # Coarse anchor (3D)
    coarse_extent :: NTuple{3,Int}  # Coarse cells covered
    fine_dims :: NTuple{3,Int}      # Fine interior dimensions
    Δx :: T                         # Grid spacing (1/ratio relative to coarse)
end

"""
    PatchPoisson3D(anchor, coarse_extent, level, μ₀_coarse, T)

Create a PatchPoisson3D solver for a 3D refined patch.

# Arguments
- `anchor`: Coarse cell anchor position (1-indexed, global)
- `coarse_extent`: Number of coarse cells in (x, y, z)
- `level`: Refinement level (1, 2, or 3)
- `μ₀_coarse`: Coarse grid μ₀ coefficient array
- `T`: Element type (Float32 or Float64)

Uses the same "unit spacing" convention as the base Poisson solver.
The L coefficients are NOT scaled by ratio² - this is consistent with
the unit spacing convention used throughout the solver.
"""
function PatchPoisson3D(anchor::NTuple{3,Int},
                        coarse_extent::NTuple{3,Int},
                        level::Int,
                        μ₀_coarse::AbstractArray{Tc},
                        ::Type{T}=Float64;
                        mem=Array) where {Tc,T}
    ratio = 2^level
    fine_dims = coarse_extent .* ratio
    # Fine grid with ghost cells
    nx, ny, nz = fine_dims
    Ng = (nx + 2, ny + 2, nz + 2)

    # Grid spacing relative to coarse (coarse = 1)
    Δx = one(T) / ratio

    # Create arrays (use mem for GPU support)
    x = zeros(T, Ng) |> mem
    z = zeros(T, Ng) |> mem
    r = zeros(T, Ng) |> mem
    ϵ = zeros(T, Ng) |> mem
    D = zeros(T, Ng) |> mem
    iD = zeros(T, Ng) |> mem
    L = ones(T, Ng..., 3) |> mem

    # Initialize L from coarse μ₀ (interpolated)
    # Uses unit spacing convention - NO ratio² scaling
    initialize_patch_coefficients_3d!(L, μ₀_coarse, anchor, coarse_extent, ratio)

    # Compute diagonal
    patch_set_diag_3d!(D, iD, L, Ng)

    PatchPoisson3D{T, typeof(x), typeof(L)}(
        L, D, iD, x, ϵ, r, z, Int16[],
        level, anchor, coarse_extent, fine_dims, Δx
    )
end

# Helper for trilinear interpolation of coefficients (GPU-compatible)
@inline function _interp_coeff_3d(μ₀_coarse::AbstractArray, I::CartesianIndex{3},
                                   ai::Int, aj::Int, ak::Int, ratio::Int, d::Int, ::Type{T},
                                   nc_i::Int, nc_j::Int, nc_k::Int) where T
    fi, fj, fk = I.I
    inv_ratio = one(T) / ratio
    xf = (T(fi) - T(1.5)) * inv_ratio
    yf = (T(fj) - T(1.5)) * inv_ratio
    zf = (T(fk) - T(1.5)) * inv_ratio

    ic = clamp(floor(Int, xf) + ai, 1, nc_i - 1)
    jc = clamp(floor(Int, yf) + aj, 1, nc_j - 1)
    kc = clamp(floor(Int, zf) + ak, 1, nc_k - 1)
    wx = clamp(xf - floor(xf), zero(T), one(T))
    wy = clamp(yf - floor(yf), zero(T), one(T))
    wz = clamp(zf - floor(zf), zero(T), one(T))

    if ic >= 1 && ic < nc_i && jc >= 1 && jc < nc_j && kc >= 1 && kc < nc_k
        v000 = T(μ₀_coarse[ic, jc, kc, d])
        v100 = T(μ₀_coarse[ic+1, jc, kc, d])
        v010 = T(μ₀_coarse[ic, jc+1, kc, d])
        v110 = T(μ₀_coarse[ic+1, jc+1, kc, d])
        v001 = T(μ₀_coarse[ic, jc, kc+1, d])
        v101 = T(μ₀_coarse[ic+1, jc, kc+1, d])
        v011 = T(μ₀_coarse[ic, jc+1, kc+1, d])
        v111 = T(μ₀_coarse[ic+1, jc+1, kc+1, d])

        c00 = (1-wx)*v000 + wx*v100
        c10 = (1-wx)*v010 + wx*v110
        c01 = (1-wx)*v001 + wx*v101
        c11 = (1-wx)*v011 + wx*v111
        c0 = (1-wy)*c00 + wy*c10
        c1 = (1-wy)*c01 + wy*c11
        return (1-wz)*c0 + wz*c1
    else
        return one(T)
    end
end

"""
    initialize_patch_coefficients_3d!(L, μ₀_coarse, anchor, extent, ratio)

Initialize 3D patch coefficient array L by interpolating from coarse μ₀.
Uses trilinear interpolation. GPU-compatible via @loop.
"""
function initialize_patch_coefficients_3d!(L::AbstractArray{T},
                                            μ₀_coarse::AbstractArray,
                                            anchor::NTuple{3,Int},
                                            extent::NTuple{3,Int},
                                            ratio::Int) where T
    ai, aj, ak = anchor
    nx, ny, nz = size(L, 1) - 2, size(L, 2) - 2, size(L, 3) - 2
    nc_i, nc_j, nc_k = size(μ₀_coarse, 1), size(μ₀_coarse, 2), size(μ₀_coarse, 3)

    R = CartesianIndices((1:nx+2, 1:ny+2, 1:nz+2))
    @loop L[I, 1] = _interp_coeff_3d(μ₀_coarse, I, ai, aj, ak, ratio, 1, T, nc_i, nc_j, nc_k) over I ∈ R
    @loop L[I, 2] = _interp_coeff_3d(μ₀_coarse, I, ai, aj, ak, ratio, 2, T, nc_i, nc_j, nc_k) over I ∈ R
    @loop L[I, 3] = _interp_coeff_3d(μ₀_coarse, I, ai, aj, ak, ratio, 3, T, nc_i, nc_j, nc_k) over I ∈ R
end

# Helper for computing 3D diagonal entry (GPU-compatible)
@inline function _patch_diag_entry_3d(L::AbstractArray{T}, I::CartesianIndex{3}) where T
    i, j, k = I.I
    s = zero(T)
    s -= L[i, j, k, 1] + L[i+1, j, k, 1]  # x-direction
    s -= L[i, j, k, 2] + L[i, j+1, k, 2]  # y-direction
    s -= L[i, j, k, 3] + L[i, j, k+1, 3]  # z-direction
    return s
end

"""
    patch_set_diag_3d!(D, iD, L, Ng)

Compute diagonal and inverse diagonal for 3D patch Poisson matrix.
Same formula as base Poisson: D[I] = -Σᵢ(L[I,i] + L[I+δ(i),i])
GPU-compatible via @loop.
"""
function patch_set_diag_3d!(D::AbstractArray{T}, iD::AbstractArray{T},
                             L::AbstractArray{T}, Ng::NTuple{3,Int}) where T
    R = CartesianIndices((2:Ng[1]-1, 2:Ng[2]-1, 2:Ng[3]-1))
    @loop (s = _patch_diag_entry_3d(L, I);
           D[I] = s;
           iD[I] = abs2(s) < 2eps(T) ? zero(T) : inv(s)) over I ∈ R
end

"""
    refinement_ratio(patch::PatchPoisson3D)

Return the refinement ratio for this 3D patch.
"""
refinement_ratio(patch::PatchPoisson3D) = 2^patch.level

"""
    inside(patch::PatchPoisson3D)

Return CartesianIndices for interior cells (excluding ghost cells).
"""
inside(patch::PatchPoisson3D) = CartesianIndices((2:patch.fine_dims[1]+1,
                                                   2:patch.fine_dims[2]+1,
                                                   2:patch.fine_dims[3]+1))

"""
    patch_mult_3d!(patch::PatchPoisson3D, x)

Matrix-vector multiplication for 3D patch: z = A*x (GPU-compatible)
"""
function patch_mult_3d!(patch::PatchPoisson3D{T}, x::AbstractArray{T}) where T
    L, D, z = patch.L, patch.D, patch.z
    fill!(z, zero(T))
    R = inside(patch)
    @loop z[I] = x[I]*D[I] + x[I-δ(1,I)]*L[I,1] + x[I+δ(1,I)]*L[I+δ(1,I),1] +
                 x[I-δ(2,I)]*L[I,2] + x[I+δ(2,I)]*L[I+δ(2,I),2] +
                 x[I-δ(3,I)]*L[I,3] + x[I+δ(3,I)]*L[I+δ(3,I),3] over I ∈ R
    return z
end

"""
    patch_residual_3d!(patch::PatchPoisson3D)

Compute residual r = z - A*x for the 3D patch (GPU-compatible).
"""
function patch_residual_3d!(patch::PatchPoisson3D{T}) where T
    L, D, x, r, z, iD = patch.L, patch.D, patch.x, patch.r, patch.z, patch.iD
    R = inside(patch)

    # Compute residual: r = z - A*x (GPU-compatible via @loop)
    @loop r[I] = ifelse(iD[I] == zero(T), zero(T),
        z[I] - (x[I]*D[I] + x[I-δ(1,I)]*L[I,1] + x[I+δ(1,I)]*L[I+δ(1,I),1] +
                x[I-δ(2,I)]*L[I,2] + x[I+δ(2,I)]*L[I+δ(2,I),2] +
                x[I-δ(3,I)]*L[I,3] + x[I+δ(3,I)]*L[I+δ(3,I),3])) over I ∈ R

    # Enforce global solvability using GPU-compatible reductions
    r_inside = @view r[R]
    iD_inside = @view iD[R]
    count = sum(x -> x != zero(T) ? 1 : 0, iD_inside)
    if count > 0
        sum_r = sum(r_inside)
        mean_r = sum_r / count
        if abs(mean_r) > 2eps(T)
            @loop r[I] = r[I] - mean_r over I ∈ R
        end
    end
end

"""
    patch_increment_3d!(patch::PatchPoisson3D)

Apply increment: x += ϵ, r -= A*ϵ for 3D patch (GPU-compatible).
"""
function patch_increment_3d!(patch::PatchPoisson3D{T}) where T
    L, D, x, ϵ, r = patch.L, patch.D, patch.x, patch.ϵ, patch.r
    R = inside(patch)
    # Compute A*ϵ and update r, x (GPU-compatible via @loop)
    # Note: inline computation required - @loop macro doesn't support local variable definitions
    @loop (r[I] -= ϵ[I]*D[I] + ϵ[I-δ(1,I)]*L[I,1] + ϵ[I+δ(1,I)]*L[I+δ(1,I),1] +
                   ϵ[I-δ(2,I)]*L[I,2] + ϵ[I+δ(2,I)]*L[I+δ(2,I),2] +
                   ϵ[I-δ(3,I)]*L[I,3] + ϵ[I+δ(3,I)]*L[I+δ(3,I),3];
           x[I] += ϵ[I]) over I ∈ R
end

"""
    patch_jacobi_3d!(patch::PatchPoisson3D; it=1)

Jacobi smoother for 3D patch (GPU-compatible).
"""
function patch_jacobi_3d!(patch::PatchPoisson3D{T}; it::Int=1) where T
    R = inside(patch)
    for _ in 1:it
        @loop patch.ϵ[I] = patch.r[I] * patch.iD[I] over I ∈ R
        patch_increment_3d!(patch)
    end
end

"""
    patch_L₂_3d(patch::PatchPoisson3D)

Compute L₂ norm of residual for 3D patch (GPU-compatible).
"""
function patch_L₂_3d(patch::PatchPoisson3D{T}) where T
    R = inside(patch)
    return sum(abs2, @view patch.r[R])
end

"""
    patch_pcg_3d!(patch::PatchPoisson3D; it=6)

Preconditioned Conjugate Gradient smoother for 3D patch.
Uses Jacobi preconditioning. GPU-compatible via @loop and broadcast reductions.
"""
function patch_pcg_3d!(patch::PatchPoisson3D{T}; it::Int=6) where T
    x, r, ϵ, z, iD = patch.x, patch.r, patch.ϵ, patch.z, patch.iD
    L, D = patch.L, patch.D
    R = inside(patch)

    # z = M⁻¹r (Jacobi preconditioner), ϵ = z (search direction)
    @loop (z[I] = r[I] * iD[I]; ϵ[I] = z[I]) over I ∈ R

    # rho = r·z (GPU-compatible dot product)
    r_inside = @view r[R]
    z_inside = @view z[R]
    rho = sum(r_inside .* z_inside)
    abs(rho) < 10eps(T) && return

    for iter in 1:it
        # z = A*ϵ (reusing z array)
        @loop z[I] = ϵ[I]*D[I] + ϵ[I-δ(1,I)]*L[I,1] + ϵ[I+δ(1,I)]*L[I+δ(1,I),1] +
                     ϵ[I-δ(2,I)]*L[I,2] + ϵ[I+δ(2,I)]*L[I+δ(2,I),2] +
                     ϵ[I-δ(3,I)]*L[I,3] + ϵ[I+δ(3,I)]*L[I+δ(3,I),3] over I ∈ R

        # alpha = rho / (z·ϵ)
        ϵ_inside = @view ϵ[R]
        z_inside = @view z[R]
        zϵ = sum(z_inside .* ϵ_inside)
        abs(zϵ) < 10eps(T) && return
        alpha = rho / zϵ
        (!isfinite(alpha) || abs(alpha) < 1e-2 || abs(alpha) > 1e2) && return

        # x += alpha*ϵ, r -= alpha*z
        @loop (x[I] = x[I] + alpha * ϵ[I]; r[I] = r[I] - alpha * z[I]) over I ∈ R

        iter == it && return

        # z = M⁻¹r
        @loop z[I] = r[I] * iD[I] over I ∈ R

        # rho2 = r·z
        r_inside = @view r[R]
        z_inside = @view z[R]
        rho2 = sum(r_inside .* z_inside)
        abs(rho2) < 10eps(T) && return

        # beta = rho2/rho
        beta = rho2 / rho

        # ϵ = z + beta*ϵ
        @loop ϵ[I] = z[I] + beta * ϵ[I] over I ∈ R

        rho = rho2
    end
end

"""
    patch_smooth_3d!(patch::PatchPoisson3D)

Default smoother for 3D patch (PCG).
"""
patch_smooth_3d!(patch::PatchPoisson3D) = patch_pcg_3d!(patch)

# Helper for trilinear interpolation from coarse pressure (GPU-compatible)
@inline function _trilinear_interp_p(p_coarse::AbstractArray{T}, fi::Int, fj::Int, fk::Int,
                                      ai::Int, aj::Int, ak::Int, ratio::Int,
                                      nc_i::Int, nc_j::Int, nc_k::Int) where T
    inv_ratio = one(T) / ratio
    x_coarse = (T(fi) - T(1.5)) * inv_ratio + ai
    y_coarse = (T(fj) - T(1.5)) * inv_ratio + aj
    z_coarse = (T(fk) - T(1.5)) * inv_ratio + ak

    ic = clamp(floor(Int, x_coarse), 1, nc_i - 1)
    jc = clamp(floor(Int, y_coarse), 1, nc_j - 1)
    kc = clamp(floor(Int, z_coarse), 1, nc_k - 1)

    wx = clamp(x_coarse - T(ic), zero(T), one(T))
    wy = clamp(y_coarse - T(jc), zero(T), one(T))
    wz = clamp(z_coarse - T(kc), zero(T), one(T))

    v000 = p_coarse[ic, jc, kc]
    v100 = p_coarse[ic+1, jc, kc]
    v010 = p_coarse[ic, jc+1, kc]
    v110 = p_coarse[ic+1, jc+1, kc]
    v001 = p_coarse[ic, jc, kc+1]
    v101 = p_coarse[ic+1, jc, kc+1]
    v011 = p_coarse[ic, jc+1, kc+1]
    v111 = p_coarse[ic+1, jc+1, kc+1]

    c00 = (1-wx)*v000 + wx*v100
    c10 = (1-wx)*v010 + wx*v110
    c01 = (1-wx)*v001 + wx*v101
    c11 = (1-wx)*v011 + wx*v111
    c0 = (1-wy)*c00 + wy*c10
    c1 = (1-wy)*c01 + wy*c11
    return (1-wz)*c0 + wz*c1
end

"""
    set_patch_boundary_3d!(patch, p_coarse, anchor)

Set 3D patch ghost cell values from coarse pressure solution.
Uses trilinear interpolation for Dirichlet BCs at patch edges.
Falls back to Neumann BC (zero gradient) at domain boundaries.
GPU-compatible via @loop.
"""
function set_patch_boundary_3d!(patch::PatchPoisson3D{T},
                                 p_coarse::AbstractArray{T},
                                 anchor::NTuple{3,Int}) where T
    ratio = refinement_ratio(patch)
    ai, aj, ak = anchor
    nx, ny, nz = patch.fine_dims
    nc_i, nc_j, nc_k = size(p_coarse, 1), size(p_coarse, 2), size(p_coarse, 3)
    x = patch.x

    # Left boundary (i = 1)
    R_yz = CartesianIndices((1:ny+2, 1:nz+2))
    if ai > 2
        @loop x[1, I[1], I[2]] = _trilinear_interp_p(p_coarse, 1, I[1], I[2], ai, aj, ak, ratio, nc_i, nc_j, nc_k) over I ∈ R_yz
    else
        @loop x[1, I[1], I[2]] = x[2, I[1], I[2]] over I ∈ R_yz
    end

    # Right boundary (i = nx+2)
    right_coarse = ai + patch.coarse_extent[1]
    if right_coarse < nc_i
        @loop x[nx+2, I[1], I[2]] = _trilinear_interp_p(p_coarse, nx+2, I[1], I[2], ai, aj, ak, ratio, nc_i, nc_j, nc_k) over I ∈ R_yz
    else
        @loop x[nx+2, I[1], I[2]] = x[nx+1, I[1], I[2]] over I ∈ R_yz
    end

    # Front boundary (j = 1)
    R_xz = CartesianIndices((1:nx+2, 1:nz+2))
    if aj > 2
        @loop x[I[1], 1, I[2]] = _trilinear_interp_p(p_coarse, I[1], 1, I[2], ai, aj, ak, ratio, nc_i, nc_j, nc_k) over I ∈ R_xz
    else
        @loop x[I[1], 1, I[2]] = x[I[1], 2, I[2]] over I ∈ R_xz
    end

    # Back boundary (j = ny+2)
    back_coarse = aj + patch.coarse_extent[2]
    if back_coarse < nc_j
        @loop x[I[1], ny+2, I[2]] = _trilinear_interp_p(p_coarse, I[1], ny+2, I[2], ai, aj, ak, ratio, nc_i, nc_j, nc_k) over I ∈ R_xz
    else
        @loop x[I[1], ny+2, I[2]] = x[I[1], ny+1, I[2]] over I ∈ R_xz
    end

    # Bottom boundary (k = 1)
    R_xy = CartesianIndices((1:nx+2, 1:ny+2))
    if ak > 2
        @loop x[I[1], I[2], 1] = _trilinear_interp_p(p_coarse, I[1], I[2], 1, ai, aj, ak, ratio, nc_i, nc_j, nc_k) over I ∈ R_xy
    else
        @loop x[I[1], I[2], 1] = x[I[1], I[2], 2] over I ∈ R_xy
    end

    # Top boundary (k = nz+2)
    top_coarse = ak + patch.coarse_extent[3]
    if top_coarse < nc_k
        @loop x[I[1], I[2], nz+2] = _trilinear_interp_p(p_coarse, I[1], I[2], nz+2, ai, aj, ak, ratio, nc_i, nc_j, nc_k) over I ∈ R_xy
    else
        @loop x[I[1], I[2], nz+2] = x[I[1], I[2], nz+1] over I ∈ R_xy
    end
end

"""
    update_coefficients!(patch::PatchPoisson3D)

Recompute diagonal after coefficient update for 3D patch.
"""
function update_coefficients!(patch::PatchPoisson3D)
    Ng = (patch.fine_dims[1] + 2, patch.fine_dims[2] + 2, patch.fine_dims[3] + 2)
    patch_set_diag_3d!(patch.D, patch.iD, patch.L, Ng)
end
