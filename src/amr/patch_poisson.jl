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
- `T`: Element type (default: Float32)

Uses the same "unit spacing" convention as the base Poisson solver.
The L coefficients are NOT scaled by ratio² - this is consistent with
the unit spacing convention used throughout the solver.
"""
function PatchPoisson(anchor::NTuple{2,Int},
                      coarse_extent::NTuple{2,Int},
                      level::Int,
                      μ₀_coarse::AbstractArray{Tc},
                      ::Type{T}=Float32;
                      mem=Array) where {Tc,T}
    ratio = 2^level
    fine_dims = coarse_extent .* ratio
    # Fine grid with ghost cells
    nx, nz = fine_dims
    Ng = (nx + 2, nz + 2)

    # Grid spacing relative to coarse (coarse = 1)
    Δx = one(T) / ratio

    # Create arrays on CPU first for initialization
    # This avoids scalar indexing on GPU during initialization
    x_cpu = zeros(T, Ng)
    z_cpu = zeros(T, Ng)
    r_cpu = zeros(T, Ng)
    ϵ_cpu = zeros(T, Ng)
    D_cpu = zeros(T, Ng)
    iD_cpu = zeros(T, Ng)
    L_cpu = ones(T, Ng..., 2)

    # Run CPU-based initialization (uses scalar indexing)
    # Convert μ₀_coarse to CPU for initialization if needed
    μ₀_cpu = μ₀_coarse isa Array ? μ₀_coarse : Array(μ₀_coarse)
    initialize_patch_coefficients!(L_cpu, μ₀_cpu, anchor, coarse_extent, ratio)

    # Compute diagonal on CPU
    patch_set_diag!(D_cpu, iD_cpu, L_cpu, Ng)

    # Now pipe to GPU (or keep on CPU if mem=Array)
    x = x_cpu |> mem
    z = z_cpu |> mem
    r = r_cpu |> mem
    ϵ = ϵ_cpu |> mem
    D = D_cpu |> mem
    iD = iD_cpu |> mem
    L = L_cpu |> mem

    PatchPoisson{T, typeof(x), typeof(L)}(
        L, D, iD, x, ϵ, r, z, Int16[],
        level, anchor, coarse_extent, fine_dims, Δx
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
        xf = (fi - T(1.5)) / ratio
        zf = (fj - T(1.5)) / ratio

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

    # ρ = r·z (GPU-compatible dot product)
    r_inside = @view r[R]
    z_inside = @view z[R]
    ρ = sum(r_inside .* z_inside)
    abs(ρ) < 10eps(T) && return

    for iter in 1:it
        # z = A*ϵ (reusing z array)
        @loop z[I] = ϵ[I]*D[I] + ϵ[I-δ(1,I)]*L[I,1] + ϵ[I+δ(1,I)]*L[I+δ(1,I),1] +
                     ϵ[I-δ(2,I)]*L[I,2] + ϵ[I+δ(2,I)]*L[I+δ(2,I),2] over I ∈ R

        # α = ρ / (z·ϵ)
        ϵ_inside = @view ϵ[R]
        z_inside = @view z[R]
        σ = sum(z_inside .* ϵ_inside)  # σ = ϵᵀAϵ
        abs(σ) < 10eps(T) && return
        α = ρ / σ
        (!isfinite(α) || abs(α) < T(1e-2) || abs(α) > T(1e2)) && return

        # x += α*ϵ, r -= α*z
        @loop (x[I] = x[I] + α * ϵ[I]; r[I] = r[I] - α * z[I]) over I ∈ R

        iter == it && return

        # z = M⁻¹r
        @loop z[I] = r[I] * iD[I] over I ∈ R

        # ρ₂ = r·z
        r_inside = @view r[R]
        z_inside = @view z[R]
        ρ₂ = sum(r_inside .* z_inside)
        abs(ρ₂) < 10eps(T) && return

        # β = ρ₂/ρ
        β = ρ₂ / ρ

        # ϵ = z + β*ϵ
        @loop ϵ[I] = z[I] + β * ϵ[I] over I ∈ R

        ρ = ρ₂
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

    # Helper for bilinear interpolation at a specific fine cell position
    # The fine cell center is at (fi - 0.5) in 1-indexed fine coordinates
    # Mapping to coarse: coarse_pos = (fi - 1) / ratio + ai - 1 + 0.5
    # Simplified: coarse_pos = (fi - 1) / ratio + ai - 0.5
    function bilinear_interp_at(fi::Int, fj::Int)
        # Fine cell center in coarse coordinates (relative to ai, aj)
        # Fine cell fi has center at (fi - 1.5) / ratio relative to anchor
        x_coarse = (fi - T(1.5)) / ratio + ai
        z_coarse = (fj - T(1.5)) / ratio + aj

        # Get lower-left coarse cell for interpolation
        # Cell ic has center at position ic, so floor(x_coarse) gives the cell
        # whose center is at or below x_coarse
        ic = floor(Int, x_coarse)
        jc = floor(Int, z_coarse)

        # Clamp to valid range (leaving room for +1 access)
        ic = clamp(ic, 1, nc_i - 1)
        jc = clamp(jc, 1, nc_j - 1)

        # Interpolation weights (based on cell center positions)
        # Coarse cell ic has center at ic (in 1-based coordinates)
        wx = x_coarse - T(ic)
        wz = z_coarse - T(jc)
        wx = clamp(wx, zero(T), one(T))
        wz = clamp(wz, zero(T), one(T))

        # Bilinear interpolation
        v00 = p_coarse[ic, jc]
        v10 = p_coarse[ic+1, jc]
        v01 = p_coarse[ic, jc+1]
        v11 = p_coarse[ic+1, jc+1]

        return (1-wx)*(1-wz)*v00 + wx*(1-wz)*v10 + (1-wx)*wz*v01 + wx*wz*v11
    end

    # Left boundary (i = 1) - ghost cell to the left of interior
    if ai > 2  # Interior interface - use Dirichlet from coarse
        for fj in 1:nz+2
            patch.x[1, fj] = bilinear_interp_at(1, fj)
        end
    else  # Domain boundary - use Neumann BC (zero gradient)
        for fj in 1:nz+2
            patch.x[1, fj] = patch.x[2, fj]
        end
    end

    # Right boundary (i = nx+2) - ghost cell to the right of interior
    right_coarse = ai + patch.coarse_extent[1]
    if right_coarse < nc_i  # Interior interface - use Dirichlet
        for fj in 1:nz+2
            patch.x[nx+2, fj] = bilinear_interp_at(nx+2, fj)
        end
    else  # Domain boundary - use Neumann BC
        for fj in 1:nz+2
            patch.x[nx+2, fj] = patch.x[nx+1, fj]
        end
    end

    # Bottom boundary (j = 1) - ghost cell below interior
    if aj > 2  # Interior interface - use Dirichlet
        for fi in 1:nx+2
            patch.x[fi, 1] = bilinear_interp_at(fi, 1)
        end
    else  # Domain boundary - use Neumann BC
        for fi in 1:nx+2
            patch.x[fi, 1] = patch.x[fi, 2]
        end
    end

    # Top boundary (j = nz+2) - ghost cell above interior
    top_coarse = aj + patch.coarse_extent[2]
    if top_coarse < nc_j  # Interior interface - use Dirichlet
        for fi in 1:nx+2
            patch.x[fi, nz+2] = bilinear_interp_at(fi, nz+2)
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
- `T`: Element type (default: Float32)

Uses the same "unit spacing" convention as the base Poisson solver.
The L coefficients are NOT scaled by ratio² - this is consistent with
the unit spacing convention used throughout the solver.
"""
function PatchPoisson3D(anchor::NTuple{3,Int},
                        coarse_extent::NTuple{3,Int},
                        level::Int,
                        μ₀_coarse::AbstractArray{Tc},
                        ::Type{T}=Float32;
                        mem=Array) where {Tc,T}
    ratio = 2^level
    fine_dims = coarse_extent .* ratio
    # Fine grid with ghost cells
    nx, ny, nz = fine_dims
    Ng = (nx + 2, ny + 2, nz + 2)

    # Grid spacing relative to coarse (coarse = 1)
    Δx = one(T) / ratio

    # Create arrays on CPU first for initialization
    # This avoids scalar indexing on GPU during initialization
    x_cpu = zeros(T, Ng)
    z_cpu = zeros(T, Ng)
    r_cpu = zeros(T, Ng)
    ϵ_cpu = zeros(T, Ng)
    D_cpu = zeros(T, Ng)
    iD_cpu = zeros(T, Ng)
    L_cpu = ones(T, Ng..., 3)

    # Run CPU-based initialization (uses scalar indexing)
    # Convert μ₀_coarse to CPU for initialization if needed
    μ₀_cpu = μ₀_coarse isa Array ? μ₀_coarse : Array(μ₀_coarse)
    initialize_patch_coefficients_3d!(L_cpu, μ₀_cpu, anchor, coarse_extent, ratio)

    # Compute diagonal on CPU
    patch_set_diag_3d!(D_cpu, iD_cpu, L_cpu, Ng)

    # Now pipe to GPU (or keep on CPU if mem=Array)
    x = x_cpu |> mem
    z = z_cpu |> mem
    r = r_cpu |> mem
    ϵ = ϵ_cpu |> mem
    D = D_cpu |> mem
    iD = iD_cpu |> mem
    L = L_cpu |> mem

    PatchPoisson3D{T, typeof(x), typeof(L)}(
        L, D, iD, x, ϵ, r, z, Int16[],
        level, anchor, coarse_extent, fine_dims, Δx
    )
end

"""
    initialize_patch_coefficients_3d!(L, μ₀_coarse, anchor, extent, ratio)

Initialize 3D patch coefficient array L by interpolating from coarse μ₀.
Uses trilinear interpolation.
"""
function initialize_patch_coefficients_3d!(L::AbstractArray{T},
                                            μ₀_coarse::AbstractArray,
                                            anchor::NTuple{3,Int},
                                            extent::NTuple{3,Int},
                                            ratio::Int) where T
    ai, aj, ak = anchor
    nx, ny, nz = size(L, 1) - 2, size(L, 2) - 2, size(L, 3) - 2

    for fi in 1:nx+2, fj in 1:ny+2, fk in 1:nz+2
        # Position in coarse cell units
        xf = (fi - T(1.5)) / ratio
        yf = (fj - T(1.5)) / ratio
        zf = (fk - T(1.5)) / ratio

        # Coarse cell and interpolation weights
        ic = clamp(floor(Int, xf) + ai, 1, size(μ₀_coarse, 1) - 1)
        jc = clamp(floor(Int, yf) + aj, 1, size(μ₀_coarse, 2) - 1)
        kc = clamp(floor(Int, zf) + ak, 1, size(μ₀_coarse, 3) - 1)
        wx = clamp(xf - floor(xf), zero(T), one(T))
        wy = clamp(yf - floor(yf), zero(T), one(T))
        wz = clamp(zf - floor(zf), zero(T), one(T))

        # Trilinear interpolation for each direction
        for d in 1:3
            if ic >= 1 && ic < size(μ₀_coarse, 1) &&
               jc >= 1 && jc < size(μ₀_coarse, 2) &&
               kc >= 1 && kc < size(μ₀_coarse, 3)
                v000 = T(μ₀_coarse[ic, jc, kc, d])
                v100 = T(μ₀_coarse[ic+1, jc, kc, d])
                v010 = T(μ₀_coarse[ic, jc+1, kc, d])
                v110 = T(μ₀_coarse[ic+1, jc+1, kc, d])
                v001 = T(μ₀_coarse[ic, jc, kc+1, d])
                v101 = T(μ₀_coarse[ic+1, jc, kc+1, d])
                v011 = T(μ₀_coarse[ic, jc+1, kc+1, d])
                v111 = T(μ₀_coarse[ic+1, jc+1, kc+1, d])

                # Trilinear interpolation
                c00 = (1-wx)*v000 + wx*v100
                c10 = (1-wx)*v010 + wx*v110
                c01 = (1-wx)*v001 + wx*v101
                c11 = (1-wx)*v011 + wx*v111
                c0 = (1-wy)*c00 + wy*c10
                c1 = (1-wy)*c01 + wy*c11
                L[fi, fj, fk, d] = (1-wz)*c0 + wz*c1
            else
                L[fi, fj, fk, d] = one(T)
            end
        end
    end
end

"""
    patch_set_diag_3d!(D, iD, L, Ng)

Compute diagonal and inverse diagonal for 3D patch Poisson matrix.
Same formula as base Poisson: D[I] = -Σᵢ(L[I,i] + L[I+δ(i),i])
"""
function patch_set_diag_3d!(D::AbstractArray{T}, iD::AbstractArray{T},
                             L::AbstractArray{T}, Ng::NTuple{3,Int}) where T
    for k in 2:Ng[3]-1, j in 2:Ng[2]-1, i in 2:Ng[1]-1
        s = zero(T)
        s -= L[i, j, k, 1] + L[i+1, j, k, 1]  # x-direction
        s -= L[i, j, k, 2] + L[i, j+1, k, 2]  # y-direction
        s -= L[i, j, k, 3] + L[i, j, k+1, 3]  # z-direction
        D[i, j, k] = s
        iD[i, j, k] = abs2(s) < 2eps(T) ? zero(T) : inv(s)
    end
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

    # ρ = r·z (GPU-compatible dot product)
    r_inside = @view r[R]
    z_inside = @view z[R]
    ρ = sum(r_inside .* z_inside)
    abs(ρ) < 10eps(T) && return

    for iter in 1:it
        # z = A*ϵ (reusing z array)
        @loop z[I] = ϵ[I]*D[I] + ϵ[I-δ(1,I)]*L[I,1] + ϵ[I+δ(1,I)]*L[I+δ(1,I),1] +
                     ϵ[I-δ(2,I)]*L[I,2] + ϵ[I+δ(2,I)]*L[I+δ(2,I),2] +
                     ϵ[I-δ(3,I)]*L[I,3] + ϵ[I+δ(3,I)]*L[I+δ(3,I),3] over I ∈ R

        # α = ρ / (z·ϵ)
        ϵ_inside = @view ϵ[R]
        z_inside = @view z[R]
        σ = sum(z_inside .* ϵ_inside)  # σ = ϵᵀAϵ
        abs(σ) < 10eps(T) && return
        α = ρ / σ
        (!isfinite(α) || abs(α) < T(1e-2) || abs(α) > T(1e2)) && return

        # x += α*ϵ, r -= α*z
        @loop (x[I] = x[I] + α * ϵ[I]; r[I] = r[I] - α * z[I]) over I ∈ R

        iter == it && return

        # z = M⁻¹r
        @loop z[I] = r[I] * iD[I] over I ∈ R

        # ρ₂ = r·z
        r_inside = @view r[R]
        z_inside = @view z[R]
        ρ₂ = sum(r_inside .* z_inside)
        abs(ρ₂) < 10eps(T) && return

        # β = ρ₂/ρ
        β = ρ₂ / ρ

        # ϵ = z + β*ϵ
        @loop ϵ[I] = z[I] + β * ϵ[I] over I ∈ R

        ρ = ρ₂
    end
end

"""
    patch_smooth_3d!(patch::PatchPoisson3D)

Default smoother for 3D patch (PCG).
"""
patch_smooth_3d!(patch::PatchPoisson3D) = patch_pcg_3d!(patch)

"""
    set_patch_boundary_3d!(patch, p_coarse, anchor)

Set 3D patch ghost cell values from coarse pressure solution.
Uses trilinear interpolation for Dirichlet BCs at patch edges.
Falls back to Neumann BC (zero gradient) at domain boundaries.
"""
function set_patch_boundary_3d!(patch::PatchPoisson3D{T},
                                 p_coarse::AbstractArray{T},
                                 anchor::NTuple{3,Int}) where T
    ratio = refinement_ratio(patch)
    ai, aj, ak = anchor
    nx, ny, nz = patch.fine_dims
    nc_i, nc_j, nc_k = size(p_coarse, 1), size(p_coarse, 2), size(p_coarse, 3)

    # Helper for trilinear interpolation
    function trilinear_interp_at(fi::Int, fj::Int, fk::Int)
        x_coarse = (fi - T(1.5)) / ratio + ai
        y_coarse = (fj - T(1.5)) / ratio + aj
        z_coarse = (fk - T(1.5)) / ratio + ak

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

    # Left boundary (i = 1)
    if ai > 2
        for fj in 1:ny+2, fk in 1:nz+2
            patch.x[1, fj, fk] = trilinear_interp_at(1, fj, fk)
        end
    else
        for fj in 1:ny+2, fk in 1:nz+2
            patch.x[1, fj, fk] = patch.x[2, fj, fk]
        end
    end

    # Right boundary (i = nx+2)
    right_coarse = ai + patch.coarse_extent[1]
    if right_coarse < nc_i
        for fj in 1:ny+2, fk in 1:nz+2
            patch.x[nx+2, fj, fk] = trilinear_interp_at(nx+2, fj, fk)
        end
    else
        for fj in 1:ny+2, fk in 1:nz+2
            patch.x[nx+2, fj, fk] = patch.x[nx+1, fj, fk]
        end
    end

    # Front boundary (j = 1)
    if aj > 2
        for fi in 1:nx+2, fk in 1:nz+2
            patch.x[fi, 1, fk] = trilinear_interp_at(fi, 1, fk)
        end
    else
        for fi in 1:nx+2, fk in 1:nz+2
            patch.x[fi, 1, fk] = patch.x[fi, 2, fk]
        end
    end

    # Back boundary (j = ny+2)
    back_coarse = aj + patch.coarse_extent[2]
    if back_coarse < nc_j
        for fi in 1:nx+2, fk in 1:nz+2
            patch.x[fi, ny+2, fk] = trilinear_interp_at(fi, ny+2, fk)
        end
    else
        for fi in 1:nx+2, fk in 1:nz+2
            patch.x[fi, ny+2, fk] = patch.x[fi, ny+1, fk]
        end
    end

    # Bottom boundary (k = 1)
    if ak > 2
        for fi in 1:nx+2, fj in 1:ny+2
            patch.x[fi, fj, 1] = trilinear_interp_at(fi, fj, 1)
        end
    else
        for fi in 1:nx+2, fj in 1:ny+2
            patch.x[fi, fj, 1] = patch.x[fi, fj, 2]
        end
    end

    # Top boundary (k = nz+2)
    top_coarse = ak + patch.coarse_extent[3]
    if top_coarse < nc_k
        for fi in 1:nx+2, fj in 1:ny+2
            patch.x[fi, fj, nz+2] = trilinear_interp_at(fi, fj, nz+2)
        end
    else
        for fi in 1:nx+2, fj in 1:ny+2
            patch.x[fi, fj, nz+2] = patch.x[fi, fj, nz+1]
        end
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
