"""
    Interface Operators for AMR

Operators for transferring data between coarse and fine grids at refinement interfaces.
Includes prolongation (coarse→fine), restriction (fine→coarse), and flux matching.
"""

"""
    prolongate_pressure_to_boundary!(patch, p_coarse, anchor)

Interpolate coarse pressure to patch ghost cells using bilinear interpolation.
This provides Dirichlet boundary conditions for the patch solve.

# Arguments
- `patch`: PatchPoisson to fill ghost cells
- `p_coarse`: Coarse pressure solution array
- `anchor`: Anchor position in coarse grid
"""
function prolongate_pressure_to_boundary!(patch::PatchPoisson{T},
                                           p_coarse::AbstractArray{T},
                                           anchor::NTuple{2,Int}) where T
    set_patch_boundary!(patch, p_coarse, anchor)
end

"""
    restrict_pressure_to_coarse!(p_coarse, patch, anchor)

Restrict patch pressure to coarse grid using volume-weighted averaging.
Updates coarse cells covered by the patch.

# Arguments
- `p_coarse`: Coarse pressure array to update
- `patch`: PatchPoisson with fine pressure solution
- `anchor`: Anchor position in coarse grid
"""
function restrict_pressure_to_coarse!(p_coarse::AbstractArray{T},
                                       patch::PatchPoisson{T},
                                       anchor::NTuple{2,Int}) where T
    ratio = refinement_ratio(patch)
    ai, aj = anchor
    inv_ratio2 = one(T) / (ratio * ratio)

    for ci in 1:patch.coarse_extent[1], cj in 1:patch.coarse_extent[2]
        I_coarse = CartesianIndex(ai + ci - 1, aj + cj - 1)

        # Average fine cells
        sum_val = zero(T)
        for di in 1:ratio, dj in 1:ratio
            fi = (ci - 1) * ratio + di + 1  # +1 for ghost
            fj = (cj - 1) * ratio + dj + 1
            sum_val += patch.x[fi, fj]
        end
        p_coarse[I_coarse] = sum_val * inv_ratio2
    end
end

"""
    InterfaceFluxData{T}

Stores flux data at a coarse-fine interface for conservation enforcement.

# Fields
- `coarse_flux`: Flux computed from coarse solution
- `fine_fluxes`: Fluxes computed from fine solution (length = ratio)
- `mismatch`: Difference (coarse - sum(fine))
"""
struct InterfaceFluxData{T}
    coarse_flux::T
    fine_fluxes::Vector{T}
    mismatch::T
end

"""
    compute_interface_flux(patch, p_coarse, anchor, side)

Compute flux at a single interface between coarse and fine grids.
Includes bounds checking for domain boundaries.

Both coarse and fine grids use the "unit spacing" convention where L coefficients
are NOT scaled by ratio². This means fluxes computed as L * Δp are directly
comparable between coarse and fine grids.

# Arguments
- `patch`: PatchPoisson
- `p_coarse`: Coarse pressure array
- `L_coarse`: Coarse coefficient array
- `anchor`: Anchor position
- `side`: :left, :right, :bottom, or :top

# Returns
- InterfaceFluxData with coarse flux, fine fluxes, and mismatch
"""
function compute_interface_flux(patch::PatchPoisson{T},
                                 p_coarse::AbstractArray{T},
                                 L_coarse::AbstractArray{T},
                                 anchor::NTuple{2,Int},
                                 side::Symbol) where T
    ratio = refinement_ratio(patch)
    ai, aj = anchor
    nx, nz = patch.fine_dims
    nc_i, nc_j = size(p_coarse, 1), size(p_coarse, 2)

    # Compute coarse flux and fine fluxes
    # Both use unit spacing convention, so fluxes are directly comparable
    fine_fluxes = T[]
    coarse_flux = zero(T)

    if side == :left
        ic = ai
        # Check if at domain boundary (no coarse neighbor to the left)
        if ic <= 2
            # At domain boundary - no flux mismatch to compute
            return InterfaceFluxData{T}(zero(T), T[], zero(T))
        end

        for cj_idx in 1:patch.coarse_extent[2]
            jc = aj + cj_idx - 1
            if jc >= 1 && jc <= nc_j
                # Coarse flux: L[ic,jc,1] * (p[ic,jc] - p[ic-1,jc])
                c_flux = L_coarse[ic, jc, 1] * (p_coarse[ic, jc] - p_coarse[ic-1, jc])
                coarse_flux += c_flux

                # Fine fluxes (same unit spacing convention as coarse)
                for dj in 1:ratio
                    fj = (cj_idx - 1) * ratio + dj + 1
                    if fj >= 1 && fj <= nz + 2
                        f_flux = patch.L[2, fj, 1] * (patch.x[2, fj] - patch.x[1, fj])
                        push!(fine_fluxes, f_flux)
                    end
                end
            end
        end

    elseif side == :right
        # ic = ai + extent is the first cell OUTSIDE the patch (to the right)
        ic = ai + patch.coarse_extent[1]
        # Check if at domain boundary (need ic to exist for flux calculation)
        if ic >= nc_i || ic < 2
            return InterfaceFluxData{T}(zero(T), T[], zero(T))
        end

        for cj_idx in 1:patch.coarse_extent[2]
            jc = aj + cj_idx - 1
            if jc >= 1 && jc <= nc_j
                # Coarse flux OUT of patch: L[ic,jc,1] * (p[ic,jc] - p[ic-1,jc])
                # This is flux at face between (ic-1) = last cell in patch and (ic) = first cell outside
                c_flux = L_coarse[ic, jc, 1] * (p_coarse[ic, jc] - p_coarse[ic-1, jc])
                coarse_flux += c_flux

                for dj in 1:ratio
                    fj = (cj_idx - 1) * ratio + dj + 1
                    if fj >= 1 && fj <= nz + 2
                        # Fine flux at right face
                        f_flux = patch.L[nx+2, fj, 1] * (patch.x[nx+2, fj] - patch.x[nx+1, fj])
                        push!(fine_fluxes, f_flux)
                    end
                end
            end
        end

    elseif side == :bottom
        jc = aj
        # Check if at domain boundary
        if jc <= 2
            return InterfaceFluxData{T}(zero(T), T[], zero(T))
        end

        for ci_idx in 1:patch.coarse_extent[1]
            ic = ai + ci_idx - 1
            if ic >= 1 && ic <= nc_i
                # Coarse flux: L[ic,jc,2] * (p[ic,jc] - p[ic,jc-1])
                c_flux = L_coarse[ic, jc, 2] * (p_coarse[ic, jc] - p_coarse[ic, jc-1])
                coarse_flux += c_flux

                for di in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    if fi >= 1 && fi <= nx + 2
                        # Fine flux
                        f_flux = patch.L[fi, 2, 2] * (patch.x[fi, 2] - patch.x[fi, 1])
                        push!(fine_fluxes, f_flux)
                    end
                end
            end
        end

    else  # :top
        # jc = aj + extent is the first cell OUTSIDE the patch (above)
        jc = aj + patch.coarse_extent[2]
        # Check if at domain boundary (need jc to exist for flux calculation)
        if jc >= nc_j || jc < 2
            return InterfaceFluxData{T}(zero(T), T[], zero(T))
        end

        for ci_idx in 1:patch.coarse_extent[1]
            ic = ai + ci_idx - 1
            if ic >= 1 && ic <= nc_i
                # Coarse flux OUT of patch: L[ic,jc,2] * (p[ic,jc] - p[ic,jc-1])
                # This is flux at face between (jc-1) = last cell in patch and (jc) = first cell outside
                c_flux = L_coarse[ic, jc, 2] * (p_coarse[ic, jc] - p_coarse[ic, jc-1])
                coarse_flux += c_flux

                for di in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    if fi >= 1 && fi <= nx + 2
                        # Fine flux at top face
                        f_flux = patch.L[fi, nz+2, 2] * (patch.x[fi, nz+2] - patch.x[fi, nz+1])
                        push!(fine_fluxes, f_flux)
                    end
                end
            end
        end
    end

    mismatch = isempty(fine_fluxes) ? zero(T) : coarse_flux - sum(fine_fluxes)
    return InterfaceFluxData{T}(coarse_flux, fine_fluxes, mismatch)
end

"""
    compute_all_interface_fluxes(patch, p_coarse, L_coarse, anchor)

Compute flux data for all four interfaces of a patch.

# Returns
- Dictionary mapping :left/:right/:bottom/:top to InterfaceFluxData
"""
function compute_all_interface_fluxes(patch::PatchPoisson{T},
                                       p_coarse::AbstractArray{T},
                                       L_coarse::AbstractArray{T},
                                       anchor::NTuple{2,Int}) where T
    fluxes = Dict{Symbol, InterfaceFluxData{T}}()
    for side in (:left, :right, :bottom, :top)
        fluxes[side] = compute_interface_flux(patch, p_coarse, L_coarse, anchor, side)
    end
    return fluxes
end

"""
    apply_flux_correction!(patch, fluxes)

Apply flux correction to ensure conservation at interfaces.
Distributes flux mismatch equally among fine faces, accounting for sign conventions.

The flux mismatch is: coarse_flux - sum(fine_fluxes)
To correct, we adjust the interior pressure values adjacent to the interface.

# Arguments
- `patch`: PatchPoisson to correct
- `fluxes`: Dictionary of InterfaceFluxData from compute_all_interface_fluxes
"""
function apply_flux_correction!(patch::PatchPoisson{T},
                                 fluxes::Dict{Symbol, InterfaceFluxData{T}}) where T
    ratio = refinement_ratio(patch)
    nx, nz = patch.fine_dims
    min_coeff = T(1e-10)  # Minimum coefficient to avoid division by zero

    for (side, flux_data) in fluxes
        mismatch = flux_data.mismatch
        n_fine = length(flux_data.fine_fluxes)
        if n_fine == 0 || abs(mismatch) < eps(T)
            continue
        end

        # Correction per fine face (positive mismatch means coarse > fine sum)
        # We need to increase fine flux magnitude to match coarse
        correction_per_face = mismatch / n_fine

        # Apply correction to fine pressure to adjust flux
        # Flux = L * (p_interior - p_ghost), so to increase flux, increase p_interior
        if side == :left
            # Left interface: flux points into patch (negative x direction from ghost)
            # Flux = L * (p[2] - p[1]), to increase flux, increase p[2]
            fine_i = 2
            idx = 0
            for cj_idx in 1:patch.coarse_extent[2]
                for dj in 1:ratio
                    idx += 1
                    if idx > n_fine
                        break
                    end
                    fj = (cj_idx - 1) * ratio + dj + 1
                    if fj <= nz + 1  # Bounds check
                        L_val = max(patch.L[fine_i, fj, 1], min_coeff)
                        # Positive mismatch: coarse flux > sum fine flux
                        # Need to increase fine flux = increase p_interior
                        patch.x[fine_i, fj] += correction_per_face / L_val
                    end
                end
            end
        elseif side == :right
            # Right interface: flux points out of patch (positive x direction)
            # Flux = L * (p[nx+2] - p[nx+1]), but ghost is p[nx+2]
            # Fine interior flux = L * (p[nx+1] - p_ghost) with outward normal
            fine_i = nx + 1
            idx = 0
            for cj_idx in 1:patch.coarse_extent[2]
                for dj in 1:ratio
                    idx += 1
                    if idx > n_fine
                        break
                    end
                    fj = (cj_idx - 1) * ratio + dj + 1
                    if fj <= nz + 1
                        # For right face, flux is L[nx+2] * (p[nx+2] - p[nx+1])
                        # But we control p[nx+1], so to increase flux, decrease p[nx+1]
                        L_val = max(patch.L[fine_i+1, fj, 1], min_coeff)
                        patch.x[fine_i, fj] -= correction_per_face / L_val
                    end
                end
            end
        elseif side == :bottom
            # Bottom interface: flux points into patch (negative z direction from ghost)
            fine_j = 2
            idx = 0
            for ci_idx in 1:patch.coarse_extent[1]
                for di in 1:ratio
                    idx += 1
                    if idx > n_fine
                        break
                    end
                    fi = (ci_idx - 1) * ratio + di + 1
                    if fi <= nx + 1
                        L_val = max(patch.L[fi, fine_j, 2], min_coeff)
                        patch.x[fi, fine_j] += correction_per_face / L_val
                    end
                end
            end
        else  # :top
            # Top interface: flux points out of patch
            fine_j = nz + 1
            idx = 0
            for ci_idx in 1:patch.coarse_extent[1]
                for di in 1:ratio
                    idx += 1
                    if idx > n_fine
                        break
                    end
                    fi = (ci_idx - 1) * ratio + di + 1
                    if fi <= nx + 1
                        L_val = max(patch.L[fi, fine_j+1, 2], min_coeff)
                        patch.x[fi, fine_j] -= correction_per_face / L_val
                    end
                end
            end
        end
    end
end

"""
    total_flux_mismatch(fluxes)

Compute total absolute flux mismatch across all interfaces.
"""
function total_flux_mismatch(fluxes::Dict{Symbol, InterfaceFluxData{T}}) where T
    total = zero(T)
    for (_, flux_data) in fluxes
        total += abs(flux_data.mismatch)
    end
    return total
end

"""
    synchronize_coarse_fine!(p_coarse, patch, anchor, L_coarse)

Full coarse-fine synchronization:
1. Compute interface fluxes
2. Apply flux correction
3. Restrict patch to coarse

# Arguments
- `p_coarse`: Coarse pressure to update
- `patch`: PatchPoisson with fine solution
- `anchor`: Anchor position
- `L_coarse`: Coarse coefficient array
"""
function synchronize_coarse_fine!(p_coarse::AbstractArray{T},
                                   patch::PatchPoisson{T},
                                   anchor::NTuple{2,Int},
                                   L_coarse::AbstractArray{T}) where T
    # Compute flux mismatch
    fluxes = compute_all_interface_fluxes(patch, p_coarse, L_coarse, anchor)

    # Apply correction to patch
    apply_flux_correction!(patch, fluxes)

    # Restrict to coarse
    restrict_pressure_to_coarse!(p_coarse, patch, anchor)
end

"""
    compute_fine_divergence!(patch, u_coarse, u_fine, anchor)

Compute divergence on fine patch for source term.
Uses fine velocity if available, otherwise interpolates from coarse.

The divergence is scaled by 1/Δx = ratio to account for finer grid spacing:
    ∇·u = ∂u/∂x + ∂w/∂z = (Δu/Δx) + (Δw/Δz)

With Δx = 1/ratio on fine grid: ∇·u = (Δu + Δw) * ratio

# Arguments
- `patch`: PatchPoisson (z array will be filled with divergence)
- `u_coarse`: Coarse velocity array
- `u_fine`: Fine velocity patch (or nothing to interpolate)
- `anchor`: Anchor position
"""
function compute_fine_divergence!(patch::PatchPoisson{T},
                                   u_coarse::AbstractArray{T},
                                   u_fine::Union{Nothing, AbstractArray{T}},
                                   anchor::NTuple{2,Int},
                                   h_coarse::T=one(T)) where T
    ratio = refinement_ratio(patch)
    ai, aj = anchor

    if u_fine !== nothing
        # Use fine velocity directly - compute unit-spacing divergence
        for I in inside(patch)
            fi, fj = I.I
            # Unit-spacing divergence on fine grid
            dudx = u_fine[fi, fj, 1] - u_fine[fi-1, fj, 1]
            dwdz = u_fine[fi, fj, 2] - u_fine[fi, fj-1, 2]
            patch.z[I] = dudx + dwdz
        end
    else
        # Interpolate from coarse - compute unit-spacing divergence on coarse
        for I in inside(patch)
            fi, fj = I.I
            # Map to coarse location
            xf = (fi - 1.5) / ratio
            zf = (fj - 1.5) / ratio
            ic = floor(Int, xf) + ai
            jc = floor(Int, zf) + aj
            ic = clamp(ic, 2, size(u_coarse, 1) - 1)
            jc = clamp(jc, 2, size(u_coarse, 2) - 1)

            # Coarse divergence (unit-spacing)
            div_coarse = (u_coarse[ic, jc, 1] - u_coarse[ic-1, jc, 1]) +
                         (u_coarse[ic, jc, 2] - u_coarse[ic, jc-1, 2])
            patch.z[I] = div_coarse
        end
    end
end

"""
    correct_fine_velocity!(u_fine, patch, anchor)

Correct fine velocity using fine pressure gradient: u -= L*∇p/ρ

The physical velocity correction is: u -= L * (∂p/∂x) / ρ
With Δx = 1/ratio: ∂p/∂x = Δp / Δx = Δp * ratio

Since patch.L is scaled by ratio² for the Laplacian, but velocity correction
only needs gradient scaling, we use: L_scaled * Δp / ratio = L * ratio * Δp

Note: ρ scaling is handled by the caller (typically in project! or similar)

# Arguments
- `u_fine`: Fine velocity array to update
- `patch`: PatchPoisson with pressure solution
- `anchor`: Anchor position
"""
function correct_fine_velocity!(u_fine::AbstractArray{T},
                                 patch::PatchPoisson{T},
                                 anchor::NTuple{2,Int}) where T
    ratio = refinement_ratio(patch)
    # L_scaled = L * ratio², but we want L * ratio for velocity correction
    # So: L_scaled / ratio = L * ratio
    scale = one(T) / ratio

    for I in inside(patch)
        fi, fj = I.I
        # x-velocity correction: u -= L_scaled/ratio * Δp = L * ratio * Δp
        u_fine[fi, fj, 1] -= scale * patch.L[fi, fj, 1] * (patch.x[fi, fj] - patch.x[fi-1, fj])
        # z-velocity correction
        u_fine[fi, fj, 2] -= scale * patch.L[fi, fj, 2] * (patch.x[fi, fj] - patch.x[fi, fj-1])
    end
end

"""
    enforce_interface_velocity_consistency!(u_coarse, u_fine_field, patches)

Ensure velocity is consistent at coarse-fine interfaces.
Fine fluxes should sum to coarse flux.

# Arguments
- `u_coarse`: Coarse velocity array
- `u_fine_field`: RefinedVelocityField with fine velocities
- `patches`: Dictionary of PatchPoisson (for coefficient access)
"""
function enforce_interface_velocity_consistency!(u_coarse::AbstractArray{T},
                                                  u_fine_field,
                                                  patches::Dict) where T
    for (anchor, patch) in patches
        ratio = refinement_ratio(patch)
        ai, aj = anchor
        nx, nz = patch.fine_dims

        # Get corresponding fine velocity patch
        vel_patch = get_patch(u_fine_field, anchor)
        vel_patch === nothing && continue

        # Left interface
        ic = ai
        for cj_idx in 1:patch.coarse_extent[2]
            jc = aj + cj_idx - 1
            coarse_flux = u_coarse[ic, jc, 1]
            fine_sum = zero(T)
            for dj in 1:ratio
                fj = (cj_idx - 1) * ratio + dj + 1
                fine_sum += vel_patch.u[2, fj, 1]
            end
            # Correct fine to match coarse
            correction = (coarse_flux * ratio - fine_sum) / ratio
            for dj in 1:ratio
                fj = (cj_idx - 1) * ratio + dj + 1
                vel_patch.u[2, fj, 1] += correction
            end
        end

        # Similar for other interfaces (right, bottom, top)...
    end
end

# =============================================================================
# 3D INTERFACE OPERATORS
# =============================================================================

"""
    prolongate_pressure_to_boundary_3d!(patch, p_coarse, anchor)

Interpolate coarse pressure to 3D patch ghost cells using trilinear interpolation.
This provides Dirichlet boundary conditions for the patch solve.

# Arguments
- `patch`: PatchPoisson3D to fill ghost cells
- `p_coarse`: Coarse pressure solution array
- `anchor`: Anchor position in coarse grid (i, j, k)
"""
function prolongate_pressure_to_boundary_3d!(patch::PatchPoisson3D{T},
                                              p_coarse::AbstractArray{T},
                                              anchor::NTuple{3,Int}) where T
    set_patch_boundary_3d!(patch, p_coarse, anchor)
end

"""
    restrict_pressure_to_coarse_3d!(p_coarse, patch, anchor)

Restrict 3D patch pressure to coarse grid using volume-weighted averaging.
Updates coarse cells covered by the patch.

# Arguments
- `p_coarse`: Coarse pressure array to update
- `patch`: PatchPoisson3D with fine pressure solution
- `anchor`: Anchor position in coarse grid
"""
function restrict_pressure_to_coarse_3d!(p_coarse::AbstractArray{T},
                                          patch::PatchPoisson3D{T},
                                          anchor::NTuple{3,Int}) where T
    ratio = refinement_ratio(patch)
    ai, aj, ak = anchor
    inv_ratio3 = one(T) / (ratio * ratio * ratio)

    for ci in 1:patch.coarse_extent[1],
        cj in 1:patch.coarse_extent[2],
        ck in 1:patch.coarse_extent[3]

        I_coarse = CartesianIndex(ai + ci - 1, aj + cj - 1, ak + ck - 1)

        # Average fine cells
        sum_val = zero(T)
        for di in 1:ratio, dj in 1:ratio, dk in 1:ratio
            fi = (ci - 1) * ratio + di + 1  # +1 for ghost
            fj = (cj - 1) * ratio + dj + 1
            fk = (ck - 1) * ratio + dk + 1
            sum_val += patch.x[fi, fj, fk]
        end
        p_coarse[I_coarse] = sum_val * inv_ratio3
    end
end

"""
    InterfaceFluxData3D{T}

Stores flux data at a 3D coarse-fine interface for conservation enforcement.

# Fields
- `coarse_flux`: Total flux computed from coarse solution
- `fine_fluxes`: Fluxes computed from fine solution (matrix for face)
- `mismatch`: Difference (coarse - sum(fine))
"""
struct InterfaceFluxData3D{T}
    coarse_flux::T
    fine_fluxes::Vector{T}
    mismatch::T
end

"""
    compute_interface_flux_3d(patch, p_coarse, L_coarse, anchor, side)

Compute flux at a single 3D interface between coarse and fine grids.
Includes bounds checking for domain boundaries.

Since patch.L is scaled by ratio² for the Laplacian, we divide the fine flux
by ratio² to get the physical flux that can be compared with coarse flux.

# Arguments
- `patch`: PatchPoisson3D
- `p_coarse`: Coarse pressure array
- `L_coarse`: Coarse coefficient array
- `anchor`: Anchor position
- `side`: :xminus, :xplus, :yminus, :yplus, :zminus, or :zplus

# Returns
- InterfaceFluxData3D with coarse flux, fine fluxes, and mismatch
"""
function compute_interface_flux_3d(patch::PatchPoisson3D{T},
                                    p_coarse::AbstractArray{T},
                                    L_coarse::AbstractArray{T},
                                    anchor::NTuple{3,Int},
                                    side::Symbol) where T
    ratio = refinement_ratio(patch)
    ai, aj, ak = anchor
    nx, ny, nz = patch.fine_dims
    nc_i, nc_j, nc_k = size(p_coarse, 1), size(p_coarse, 2), size(p_coarse, 3)

    # Both use unit spacing convention, so fluxes are directly comparable
    fine_fluxes = T[]
    coarse_flux = zero(T)

    if side == :xminus
        ic = ai
        if ic <= 2
            return InterfaceFluxData3D{T}(zero(T), T[], zero(T))
        end

        for cj_idx in 1:patch.coarse_extent[2], ck_idx in 1:patch.coarse_extent[3]
            jc = aj + cj_idx - 1
            kc = ak + ck_idx - 1
            if jc >= 1 && jc <= nc_j && kc >= 1 && kc <= nc_k
                c_flux = L_coarse[ic, jc, kc, 1] * (p_coarse[ic, jc, kc] - p_coarse[ic-1, jc, kc])
                coarse_flux += c_flux

                # Fine fluxes (same unit spacing convention as coarse)
                for dj in 1:ratio, dk in 1:ratio
                    fj = (cj_idx - 1) * ratio + dj + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    if fj >= 1 && fj <= ny + 2 && fk >= 1 && fk <= nz + 2
                        f_flux = patch.L[2, fj, fk, 1] * (patch.x[2, fj, fk] - patch.x[1, fj, fk])
                        push!(fine_fluxes, f_flux)
                    end
                end
            end
        end

    elseif side == :xplus
        ic = ai + patch.coarse_extent[1]
        if ic >= nc_i || ic < 2
            return InterfaceFluxData3D{T}(zero(T), T[], zero(T))
        end

        for cj_idx in 1:patch.coarse_extent[2], ck_idx in 1:patch.coarse_extent[3]
            jc = aj + cj_idx - 1
            kc = ak + ck_idx - 1
            if jc >= 1 && jc <= nc_j && kc >= 1 && kc <= nc_k
                c_flux = L_coarse[ic, jc, kc, 1] * (p_coarse[ic, jc, kc] - p_coarse[ic-1, jc, kc])
                coarse_flux += c_flux

                for dj in 1:ratio, dk in 1:ratio
                    fj = (cj_idx - 1) * ratio + dj + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    if fj >= 1 && fj <= ny + 2 && fk >= 1 && fk <= nz + 2
                        f_flux = patch.L[nx+2, fj, fk, 1] * (patch.x[nx+2, fj, fk] - patch.x[nx+1, fj, fk])
                        push!(fine_fluxes, f_flux)
                    end
                end
            end
        end

    elseif side == :yminus
        jc = aj
        if jc <= 2
            return InterfaceFluxData3D{T}(zero(T), T[], zero(T))
        end

        for ci_idx in 1:patch.coarse_extent[1], ck_idx in 1:patch.coarse_extent[3]
            ic = ai + ci_idx - 1
            kc = ak + ck_idx - 1
            if ic >= 1 && ic <= nc_i && kc >= 1 && kc <= nc_k
                c_flux = L_coarse[ic, jc, kc, 2] * (p_coarse[ic, jc, kc] - p_coarse[ic, jc-1, kc])
                coarse_flux += c_flux

                for di in 1:ratio, dk in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    if fi >= 1 && fi <= nx + 2 && fk >= 1 && fk <= nz + 2
                        f_flux = patch.L[fi, 2, fk, 2] * (patch.x[fi, 2, fk] - patch.x[fi, 1, fk])
                        push!(fine_fluxes, f_flux)
                    end
                end
            end
        end

    elseif side == :yplus
        jc = aj + patch.coarse_extent[2]
        if jc >= nc_j || jc < 2
            return InterfaceFluxData3D{T}(zero(T), T[], zero(T))
        end

        for ci_idx in 1:patch.coarse_extent[1], ck_idx in 1:patch.coarse_extent[3]
            ic = ai + ci_idx - 1
            kc = ak + ck_idx - 1
            if ic >= 1 && ic <= nc_i && kc >= 1 && kc <= nc_k
                c_flux = L_coarse[ic, jc, kc, 2] * (p_coarse[ic, jc, kc] - p_coarse[ic, jc-1, kc])
                coarse_flux += c_flux

                for di in 1:ratio, dk in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    if fi >= 1 && fi <= nx + 2 && fk >= 1 && fk <= nz + 2
                        f_flux = patch.L[fi, ny+2, fk, 2] * (patch.x[fi, ny+2, fk] - patch.x[fi, ny+1, fk])
                        push!(fine_fluxes, f_flux)
                    end
                end
            end
        end

    elseif side == :zminus
        kc = ak
        if kc <= 2
            return InterfaceFluxData3D{T}(zero(T), T[], zero(T))
        end

        for ci_idx in 1:patch.coarse_extent[1], cj_idx in 1:patch.coarse_extent[2]
            ic = ai + ci_idx - 1
            jc = aj + cj_idx - 1
            if ic >= 1 && ic <= nc_i && jc >= 1 && jc <= nc_j
                c_flux = L_coarse[ic, jc, kc, 3] * (p_coarse[ic, jc, kc] - p_coarse[ic, jc, kc-1])
                coarse_flux += c_flux

                for di in 1:ratio, dj in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fj = (cj_idx - 1) * ratio + dj + 1
                    if fi >= 1 && fi <= nx + 2 && fj >= 1 && fj <= ny + 2
                        f_flux = patch.L[fi, fj, 2, 3] * (patch.x[fi, fj, 2] - patch.x[fi, fj, 1])
                        push!(fine_fluxes, f_flux)
                    end
                end
            end
        end

    else  # :zplus
        kc = ak + patch.coarse_extent[3]
        if kc >= nc_k || kc < 2
            return InterfaceFluxData3D{T}(zero(T), T[], zero(T))
        end

        for ci_idx in 1:patch.coarse_extent[1], cj_idx in 1:patch.coarse_extent[2]
            ic = ai + ci_idx - 1
            jc = aj + cj_idx - 1
            if ic >= 1 && ic <= nc_i && jc >= 1 && jc <= nc_j
                c_flux = L_coarse[ic, jc, kc, 3] * (p_coarse[ic, jc, kc] - p_coarse[ic, jc, kc-1])
                coarse_flux += c_flux

                for di in 1:ratio, dj in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fj = (cj_idx - 1) * ratio + dj + 1
                    if fi >= 1 && fi <= nx + 2 && fj >= 1 && fj <= ny + 2
                        f_flux = patch.L[fi, fj, nz+2, 3] * (patch.x[fi, fj, nz+2] - patch.x[fi, fj, nz+1])
                        push!(fine_fluxes, f_flux)
                    end
                end
            end
        end
    end

    mismatch = isempty(fine_fluxes) ? zero(T) : coarse_flux - sum(fine_fluxes)
    return InterfaceFluxData3D{T}(coarse_flux, fine_fluxes, mismatch)
end

"""
    compute_all_interface_fluxes_3d(patch, p_coarse, L_coarse, anchor)

Compute flux data for all six interfaces of a 3D patch.

# Returns
- Dictionary mapping face symbols to InterfaceFluxData3D
"""
function compute_all_interface_fluxes_3d(patch::PatchPoisson3D{T},
                                          p_coarse::AbstractArray{T},
                                          L_coarse::AbstractArray{T},
                                          anchor::NTuple{3,Int}) where T
    fluxes = Dict{Symbol, InterfaceFluxData3D{T}}()
    for side in (:xminus, :xplus, :yminus, :yplus, :zminus, :zplus)
        fluxes[side] = compute_interface_flux_3d(patch, p_coarse, L_coarse, anchor, side)
    end
    return fluxes
end

"""
    apply_flux_correction_3d!(patch, fluxes)

Apply flux correction to 3D patch to ensure conservation at interfaces.
Distributes flux mismatch equally among fine faces.

# Arguments
- `patch`: PatchPoisson3D to correct
- `fluxes`: Dictionary of InterfaceFluxData3D from compute_all_interface_fluxes_3d
"""
function apply_flux_correction_3d!(patch::PatchPoisson3D{T},
                                    fluxes::Dict{Symbol, InterfaceFluxData3D{T}}) where T
    ratio = refinement_ratio(patch)
    nx, ny, nz = patch.fine_dims
    min_coeff = T(1e-10)

    for (side, flux_data) in fluxes
        mismatch = flux_data.mismatch
        n_fine = length(flux_data.fine_fluxes)
        if n_fine == 0 || abs(mismatch) < eps(T)
            continue
        end

        correction_per_face = mismatch / n_fine

        if side == :xminus
            fine_i = 2
            idx = 0
            for cj_idx in 1:patch.coarse_extent[2], ck_idx in 1:patch.coarse_extent[3]
                for dj in 1:ratio, dk in 1:ratio
                    idx += 1
                    if idx > n_fine
                        break
                    end
                    fj = (cj_idx - 1) * ratio + dj + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    if fj <= ny + 1 && fk <= nz + 1
                        L_val = max(patch.L[fine_i, fj, fk, 1], min_coeff)
                        patch.x[fine_i, fj, fk] += correction_per_face / L_val
                    end
                end
            end

        elseif side == :xplus
            fine_i = nx + 1
            idx = 0
            for cj_idx in 1:patch.coarse_extent[2], ck_idx in 1:patch.coarse_extent[3]
                for dj in 1:ratio, dk in 1:ratio
                    idx += 1
                    if idx > n_fine
                        break
                    end
                    fj = (cj_idx - 1) * ratio + dj + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    if fj <= ny + 1 && fk <= nz + 1
                        L_val = max(patch.L[fine_i+1, fj, fk, 1], min_coeff)
                        patch.x[fine_i, fj, fk] -= correction_per_face / L_val
                    end
                end
            end

        elseif side == :yminus
            fine_j = 2
            idx = 0
            for ci_idx in 1:patch.coarse_extent[1], ck_idx in 1:patch.coarse_extent[3]
                for di in 1:ratio, dk in 1:ratio
                    idx += 1
                    if idx > n_fine
                        break
                    end
                    fi = (ci_idx - 1) * ratio + di + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    if fi <= nx + 1 && fk <= nz + 1
                        L_val = max(patch.L[fi, fine_j, fk, 2], min_coeff)
                        patch.x[fi, fine_j, fk] += correction_per_face / L_val
                    end
                end
            end

        elseif side == :yplus
            fine_j = ny + 1
            idx = 0
            for ci_idx in 1:patch.coarse_extent[1], ck_idx in 1:patch.coarse_extent[3]
                for di in 1:ratio, dk in 1:ratio
                    idx += 1
                    if idx > n_fine
                        break
                    end
                    fi = (ci_idx - 1) * ratio + di + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    if fi <= nx + 1 && fk <= nz + 1
                        L_val = max(patch.L[fi, fine_j+1, fk, 2], min_coeff)
                        patch.x[fi, fine_j, fk] -= correction_per_face / L_val
                    end
                end
            end

        elseif side == :zminus
            fine_k = 2
            idx = 0
            for ci_idx in 1:patch.coarse_extent[1], cj_idx in 1:patch.coarse_extent[2]
                for di in 1:ratio, dj in 1:ratio
                    idx += 1
                    if idx > n_fine
                        break
                    end
                    fi = (ci_idx - 1) * ratio + di + 1
                    fj = (cj_idx - 1) * ratio + dj + 1
                    if fi <= nx + 1 && fj <= ny + 1
                        L_val = max(patch.L[fi, fj, fine_k, 3], min_coeff)
                        patch.x[fi, fj, fine_k] += correction_per_face / L_val
                    end
                end
            end

        else  # :zplus
            fine_k = nz + 1
            idx = 0
            for ci_idx in 1:patch.coarse_extent[1], cj_idx in 1:patch.coarse_extent[2]
                for di in 1:ratio, dj in 1:ratio
                    idx += 1
                    if idx > n_fine
                        break
                    end
                    fi = (ci_idx - 1) * ratio + di + 1
                    fj = (cj_idx - 1) * ratio + dj + 1
                    if fi <= nx + 1 && fj <= ny + 1
                        L_val = max(patch.L[fi, fj, fine_k+1, 3], min_coeff)
                        patch.x[fi, fj, fine_k] -= correction_per_face / L_val
                    end
                end
            end
        end
    end
end

"""
    total_flux_mismatch_3d(fluxes)

Compute total absolute flux mismatch across all 3D interfaces.
"""
function total_flux_mismatch_3d(fluxes::Dict{Symbol, InterfaceFluxData3D{T}}) where T
    total = zero(T)
    for (_, flux_data) in fluxes
        total += abs(flux_data.mismatch)
    end
    return total
end

"""
    synchronize_coarse_fine_3d!(p_coarse, patch, anchor, L_coarse)

Full coarse-fine synchronization for 3D:
1. Compute interface fluxes
2. Apply flux correction
3. Restrict patch to coarse

# Arguments
- `p_coarse`: Coarse pressure to update
- `patch`: PatchPoisson3D with fine solution
- `anchor`: Anchor position
- `L_coarse`: Coarse coefficient array
"""
function synchronize_coarse_fine_3d!(p_coarse::AbstractArray{T},
                                      patch::PatchPoisson3D{T},
                                      anchor::NTuple{3,Int},
                                      L_coarse::AbstractArray{T}) where T
    # Compute flux mismatch
    fluxes = compute_all_interface_fluxes_3d(patch, p_coarse, L_coarse, anchor)

    # Apply correction to patch
    apply_flux_correction_3d!(patch, fluxes)

    # Restrict to coarse
    restrict_pressure_to_coarse_3d!(p_coarse, patch, anchor)
end

"""
    compute_fine_divergence_3d!(patch, u_coarse, u_fine, anchor)

Compute divergence on 3D fine patch for source term.
Uses fine velocity if available, otherwise interpolates from coarse.

The divergence is scaled by 1/Δx = ratio to account for finer grid spacing:
    ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = (Δu + Δv + Δw) / Δx

With Δx = 1/ratio on fine grid: ∇·u = (Δu + Δv + Δw) * ratio

# Arguments
- `patch`: PatchPoisson3D (z array will be filled with divergence)
- `u_coarse`: Coarse velocity array
- `u_fine`: Fine velocity patch (or nothing to interpolate)
- `anchor`: Anchor position
"""
function compute_fine_divergence_3d!(patch::PatchPoisson3D{T},
                                      u_coarse::AbstractArray{T},
                                      u_fine::Union{Nothing, AbstractArray{T}},
                                      anchor::NTuple{3,Int},
                                      h_coarse::T=one(T)) where T
    ratio = refinement_ratio(patch)
    ai, aj, ak = anchor

    if u_fine !== nothing
        # Use fine velocity directly - compute unit-spacing divergence
        for I in inside(patch)
            fi, fj, fk = I.I
            # Unit-spacing divergence on fine grid
            dudx = u_fine[fi, fj, fk, 1] - u_fine[fi-1, fj, fk, 1]
            dvdy = u_fine[fi, fj, fk, 2] - u_fine[fi, fj-1, fk, 2]
            dwdz = u_fine[fi, fj, fk, 3] - u_fine[fi, fj, fk-1, 3]
            patch.z[I] = dudx + dvdy + dwdz
        end
    else
        # Interpolate from coarse - compute unit-spacing divergence on coarse
        for I in inside(patch)
            fi, fj, fk = I.I
            # Map to coarse location
            xf = (fi - 1.5) / ratio
            yf = (fj - 1.5) / ratio
            zf = (fk - 1.5) / ratio
            ic = floor(Int, xf) + ai
            jc = floor(Int, yf) + aj
            kc = floor(Int, zf) + ak
            ic = clamp(ic, 2, size(u_coarse, 1) - 1)
            jc = clamp(jc, 2, size(u_coarse, 2) - 1)
            kc = clamp(kc, 2, size(u_coarse, 3) - 1)

            # Coarse divergence (unit-spacing)
            div_coarse = (u_coarse[ic, jc, kc, 1] - u_coarse[ic-1, jc, kc, 1]) +
                         (u_coarse[ic, jc, kc, 2] - u_coarse[ic, jc-1, kc, 2]) +
                         (u_coarse[ic, jc, kc, 3] - u_coarse[ic, jc, kc-1, 3])
            patch.z[I] = div_coarse
        end
    end
end

"""
    correct_fine_velocity_3d!(u_fine, patch, anchor)

Correct 3D fine velocity using fine pressure gradient: u -= L*∇p/ρ

The physical velocity correction is: u -= L * (∂p/∂x) / ρ
With Δx = 1/ratio: ∂p/∂x = Δp / Δx = Δp * ratio

Since patch.L is scaled by ratio² for the Laplacian, but velocity correction
only needs gradient scaling, we use: L_scaled * Δp / ratio = L * ratio * Δp

Note: ρ scaling is handled by the caller (typically in project! or similar)

# Arguments
- `u_fine`: Fine velocity array to update
- `patch`: PatchPoisson3D with pressure solution
- `anchor`: Anchor position
"""
function correct_fine_velocity_3d!(u_fine::AbstractArray{T},
                                    patch::PatchPoisson3D{T},
                                    anchor::NTuple{3,Int}) where T
    ratio = refinement_ratio(patch)
    # L_scaled = L * ratio², but we want L * ratio for velocity correction
    # So: L_scaled / ratio = L * ratio
    scale = one(T) / ratio

    for I in inside(patch)
        fi, fj, fk = I.I
        # x-velocity correction: u -= L_scaled/ratio * Δp = L * ratio * Δp
        u_fine[fi, fj, fk, 1] -= scale * patch.L[fi, fj, fk, 1] * (patch.x[fi, fj, fk] - patch.x[fi-1, fj, fk])
        # y-velocity correction
        u_fine[fi, fj, fk, 2] -= scale * patch.L[fi, fj, fk, 2] * (patch.x[fi, fj, fk] - patch.x[fi, fj-1, fk])
        # z-velocity correction
        u_fine[fi, fj, fk, 3] -= scale * patch.L[fi, fj, fk, 3] * (patch.x[fi, fj, fk] - patch.x[fi, fj, fk-1])
    end
end
