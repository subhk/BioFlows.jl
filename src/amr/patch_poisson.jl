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
- `parent`: Reference to parent Poisson (coarse level or base)
"""
struct PatchPoisson{T,S<:AbstractArray{T},V<:AbstractArray{T}} <: AbstractPoisson{T,S,V}
    # Standard Poisson fields
    L :: V      # Lower diagonal coefficients
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
"""
function PatchPoisson(anchor::NTuple{2,Int},
                      coarse_extent::NTuple{2,Int},
                      level::Int,
                      μ₀_coarse::AbstractArray{Tc},
                      ::Type{T}=Float64) where {Tc,T}
    ratio = 2^level
    fine_dims = coarse_extent .* ratio
    # Fine grid with ghost cells
    nx, nz = fine_dims
    Ng = (nx + 2, nz + 2)

    # Create arrays
    x = zeros(T, Ng)
    z = zeros(T, Ng)
    r = zeros(T, Ng)
    ϵ = zeros(T, Ng)
    D = zeros(T, Ng)
    iD = zeros(T, Ng)
    L = ones(T, Ng..., 2)

    # Initialize L from coarse μ₀ (interpolated)
    initialize_patch_coefficients!(L, μ₀_coarse, anchor, coarse_extent, ratio)

    # Compute diagonal
    patch_set_diag!(D, iD, L, Ng)

    PatchPoisson{T, typeof(x), typeof(L)}(
        L, D, iD, x, ϵ, r, z, Int16[],
        level, anchor, coarse_extent, fine_dims
    )
end

"""
    initialize_patch_coefficients!(L, μ₀_coarse, anchor, extent, ratio)

Initialize patch coefficient array L by interpolating from coarse μ₀.
The coefficient at fine faces is interpolated from surrounding coarse values.
"""
function initialize_patch_coefficients!(L::AbstractArray{T},
                                         μ₀_coarse::AbstractArray,
                                         anchor::NTuple{2,Int},
                                         extent::NTuple{2,Int},
                                         ratio::Int) where T
    ai, aj = anchor
    nx, nz = size(L, 1) - 2, size(L, 2) - 2

    for fi in 1:nx+2, fj in 1:nz+2
        # Position in coarse cell units
        xf = (fi - 1.5) / ratio
        zf = (fj - 1.5) / ratio

        # Coarse cell and interpolation weights
        ic = clamp(floor(Int, xf) + ai, 1, size(μ₀_coarse, 1) - 1)
        jc = clamp(floor(Int, zf) + aj, 1, size(μ₀_coarse, 2) - 1)
        wx = clamp(xf - floor(xf), zero(T), one(T))
        wz = clamp(zf - floor(zf), zero(T), one(T))

        # Bilinear interpolation for each direction
        for d in 1:2
            if ic >= 1 && ic < size(μ₀_coarse, 1) && jc >= 1 && jc < size(μ₀_coarse, 2)
                v00 = T(μ₀_coarse[ic, jc, d])
                v10 = T(μ₀_coarse[ic+1, jc, d])
                v01 = T(μ₀_coarse[ic, jc+1, d])
                v11 = T(μ₀_coarse[ic+1, jc+1, d])
                L[fi, fj, d] = (1-wx)*(1-wz)*v00 + wx*(1-wz)*v10 +
                               (1-wx)*wz*v01 + wx*wz*v11
            else
                L[fi, fj, d] = one(T)
            end
        end
    end
end

"""
    patch_set_diag!(D, iD, L, Ng)

Compute diagonal and inverse diagonal for patch Poisson matrix.
Same formula as base Poisson: D[I] = -Σᵢ(L[I,i] + L[I+δ(i),i])
"""
function patch_set_diag!(D::AbstractArray{T}, iD::AbstractArray{T},
                          L::AbstractArray{T}, Ng::NTuple{2,Int}) where T
    for j in 2:Ng[2]-1, i in 2:Ng[1]-1
        s = zero(T)
        s -= L[i, j, 1] + L[i+1, j, 1]  # x-direction
        s -= L[i, j, 2] + L[i, j+1, 2]  # z-direction
        D[i, j] = s
        iD[i, j] = abs2(s) < 2eps(T) ? zero(T) : inv(s)
    end
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

    for I in inside(patch)
        i, j = I.I
        s = x[i, j] * D[i, j]
        # x-neighbors
        s += x[i-1, j] * L[i, j, 1]
        s += x[i+1, j] * L[i+1, j, 1]
        # z-neighbors
        s += x[i, j-1] * L[i, j, 2]
        s += x[i, j+1] * L[i, j+1, 2]
        z[I] = s
    end
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

    count = 0
    sum_r = zero(T)

    for I in inside(patch)
        i, j = I.I
        if iD[i, j] == zero(T)
            r[I] = zero(T)
        else
            # Compute A*x at this point
            ax = x[i, j] * D[i, j]
            ax += x[i-1, j] * L[i, j, 1] + x[i+1, j] * L[i+1, j, 1]
            ax += x[i, j-1] * L[i, j, 2] + x[i, j+1] * L[i, j+1, 2]
            r[I] = z[I] - ax
            sum_r += r[I]
            count += 1
        end
    end

    # Enforce global solvability
    if count > 0
        mean_r = sum_r / count
        if abs(mean_r) > 2eps(T)
            for I in inside(patch)
                r[I] -= mean_r
            end
        end
    end
end

"""
    patch_increment!(patch::PatchPoisson)

Apply increment: x += ϵ, r -= A*ϵ
"""
function patch_increment!(patch::PatchPoisson{T}) where T
    L, D, x, ϵ, r = patch.L, patch.D, patch.x, patch.ϵ, patch.r

    for I in inside(patch)
        i, j = I.I
        # Compute A*ϵ
        aϵ = ϵ[i, j] * D[i, j]
        aϵ += ϵ[i-1, j] * L[i, j, 1] + ϵ[i+1, j] * L[i+1, j, 1]
        aϵ += ϵ[i, j-1] * L[i, j, 2] + ϵ[i, j+1] * L[i, j+1, 2]
        r[I] -= aϵ
        x[I] += ϵ[I]
    end
end

"""
    patch_jacobi!(patch::PatchPoisson; it=1)

Jacobi smoother for patch.
"""
function patch_jacobi!(patch::PatchPoisson{T}; it::Int=1) where T
    for _ in 1:it
        for I in inside(patch)
            patch.ϵ[I] = patch.r[I] * patch.iD[I]
        end
        patch_increment!(patch)
    end
end

"""
    patch_L₂(patch::PatchPoisson)

Compute L₂ norm of residual.
"""
function patch_L₂(patch::PatchPoisson{T}) where T
    s = zero(T)
    for I in inside(patch)
        s += patch.r[I]^2
    end
    return s
end

"""
    patch_L∞(patch::PatchPoisson)

Compute L∞ norm of residual.
"""
function patch_L∞(patch::PatchPoisson{T}) where T
    m = zero(T)
    for I in inside(patch)
        m = max(m, abs(patch.r[I]))
    end
    return m
end

"""
    patch_pcg!(patch::PatchPoisson; it=6)

Preconditioned Conjugate Gradient smoother for patch.
Uses Jacobi preconditioning.
"""
function patch_pcg!(patch::PatchPoisson{T}; it::Int=6) where T
    x, r, ϵ, z, iD = patch.x, patch.r, patch.ϵ, patch.z, patch.iD
    L, D = patch.L, patch.D

    # z = M⁻¹r (Jacobi preconditioner), ϵ = z (search direction)
    for I in inside(patch)
        z[I] = r[I] * iD[I]
        ϵ[I] = z[I]
    end

    # rho = r·z
    rho = zero(T)
    for I in inside(patch)
        rho += r[I] * z[I]
    end
    abs(rho) < 10eps(T) && return

    for iter in 1:it
        # z = A*ϵ (reusing z array)
        for I in inside(patch)
            i, j = I.I
            aϵ = ϵ[i, j] * D[i, j]
            aϵ += ϵ[i-1, j] * L[i, j, 1] + ϵ[i+1, j] * L[i+1, j, 1]
            aϵ += ϵ[i, j-1] * L[i, j, 2] + ϵ[i, j+1] * L[i, j+1, 2]
            z[I] = aϵ
        end

        # alpha = rho / (z·ϵ)
        zϵ = zero(T)
        for I in inside(patch)
            zϵ += z[I] * ϵ[I]
        end
        alpha = rho / zϵ
        (abs(alpha) < 1e-2 || abs(alpha) > 1e2) && return

        # x += alpha*ϵ, r -= alpha*z
        for I in inside(patch)
            x[I] += alpha * ϵ[I]
            r[I] -= alpha * z[I]
        end

        iter == it && return

        # z = M⁻¹r
        for I in inside(patch)
            z[I] = r[I] * iD[I]
        end

        # rho2 = r·z
        rho2 = zero(T)
        for I in inside(patch)
            rho2 += r[I] * z[I]
        end
        abs(rho2) < 10eps(T) && return

        # beta = rho2/rho
        beta = rho2 / rho

        # ϵ = z + beta*ϵ
        for I in inside(patch)
            ϵ[I] = z[I] + beta * ϵ[I]
        end

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

"""
    set_patch_boundary!(patch, p_coarse, anchor)

Set patch ghost cell values from coarse pressure solution.
Uses 2D bilinear interpolation for Dirichlet BCs at patch edges.
Falls back to Neumann BC (zero gradient) at domain boundaries.
"""
function set_patch_boundary!(patch::PatchPoisson{T},
                              p_coarse::AbstractArray{T},
                              anchor::NTuple{2,Int}) where T
    ratio = refinement_ratio(patch)
    ai, aj = anchor
    nx, nz = patch.fine_dims
    nc_i, nc_j = size(p_coarse, 1), size(p_coarse, 2)

    # Helper for 2D bilinear interpolation with bounds checking
    function bilinear_interp(fi::Int, fj::Int, i_offset::Int, j_offset::Int)
        # Map fine cell to coarse location
        xf = (fi - 1.5) / ratio
        zf = (fj - 1.5) / ratio

        # Coarse indices (with offset for boundary position)
        ic = floor(Int, xf) + ai + i_offset
        jc = floor(Int, zf) + aj + j_offset

        # Bounds check
        ic = clamp(ic, 1, nc_i - 1)
        jc = clamp(jc, 1, nc_j - 1)

        # Ensure we don't go out of bounds for +1 access
        ic_next = min(ic + 1, nc_i)
        jc_next = min(jc + 1, nc_j)

        # Weights for bilinear interpolation
        wx = clamp(xf - floor(xf), zero(T), one(T))
        wz = clamp(zf - floor(zf), zero(T), one(T))

        # Bilinear interpolation
        v00 = p_coarse[ic, jc]
        v10 = p_coarse[ic_next, jc]
        v01 = p_coarse[ic, jc_next]
        v11 = p_coarse[ic_next, jc_next]

        return (1-wx)*(1-wz)*v00 + wx*(1-wz)*v10 + (1-wx)*wz*v01 + wx*wz*v11
    end

    # Left boundary (i = 1)
    if ai > 1  # Interior interface - use Dirichlet from coarse
        for fj in 1:nz+2
            patch.x[1, fj] = bilinear_interp(0, fj, -1, 0)
        end
    else  # Domain boundary - use Neumann BC (zero gradient)
        for fj in 1:nz+2
            patch.x[1, fj] = patch.x[2, fj]
        end
    end

    # Right boundary (i = nx+2)
    right_coarse = ai + patch.coarse_extent[1]
    if right_coarse < nc_i  # Interior interface - use Dirichlet
        for fj in 1:nz+2
            patch.x[nx+2, fj] = bilinear_interp(nx+2, fj, 0, 0)
        end
    else  # Domain boundary - use Neumann BC
        for fj in 1:nz+2
            patch.x[nx+2, fj] = patch.x[nx+1, fj]
        end
    end

    # Bottom boundary (j = 1)
    if aj > 1  # Interior interface - use Dirichlet
        for fi in 1:nx+2
            patch.x[fi, 1] = bilinear_interp(fi, 0, 0, -1)
        end
    else  # Domain boundary - use Neumann BC
        for fi in 1:nx+2
            patch.x[fi, 1] = patch.x[fi, 2]
        end
    end

    # Top boundary (j = nz+2)
    top_coarse = aj + patch.coarse_extent[2]
    if top_coarse < nc_j  # Interior interface - use Dirichlet
        for fi in 1:nx+2
            patch.x[fi, nz+2] = bilinear_interp(fi, nz+2, 0, 0)
        end
    else  # Domain boundary - use Neumann BC
        for fi in 1:nx+2
            patch.x[fi, nz+2] = patch.x[fi, nz+1]
        end
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
