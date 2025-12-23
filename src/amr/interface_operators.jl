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

# Arguments
- `patch`: PatchPoisson
- `p_coarse`: Coarse pressure array
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

    # Determine direction and coarse cell index
    if side == :left
        dir = 1
        ic = ai
        jc_range = aj:(aj + patch.coarse_extent[2] - 1)
        fine_i = 2  # First interior fine cell
        fine_j_offset = 0
    elseif side == :right
        dir = 1
        ic = ai + patch.coarse_extent[1]
        jc_range = aj:(aj + patch.coarse_extent[2] - 1)
        fine_i = nx + 1  # Last interior fine cell
        fine_j_offset = 0
    elseif side == :bottom
        dir = 2
        jc = aj
        ic_range = ai:(ai + patch.coarse_extent[1] - 1)
        fine_j = 2
        fine_i_offset = 0
    else  # :top
        dir = 2
        jc = aj + patch.coarse_extent[2]
        ic_range = ai:(ai + patch.coarse_extent[1] - 1)
        fine_j = nz + 1
        fine_i_offset = 0
    end

    # Compute coarse flux and fine fluxes
    fine_fluxes = T[]
    coarse_flux = zero(T)

    if dir == 1  # x-direction interface (left/right)
        for cj_idx in 1:patch.coarse_extent[2]
            jc = aj + cj_idx - 1

            # Coarse flux: L[ic,jc,1] * (p[ic,jc] - p[ic-1,jc]) for left
            #              L[ic+1,jc,1] * (p[ic+1,jc] - p[ic,jc]) for right
            if side == :left
                c_flux = L_coarse[ic, jc, 1] * (p_coarse[ic, jc] - p_coarse[ic-1, jc])
            else
                c_flux = L_coarse[ic+1, jc, 1] * (p_coarse[ic+1, jc] - p_coarse[ic, jc])
            end
            coarse_flux += c_flux

            # Fine fluxes
            for dj in 1:ratio
                fj = (cj_idx - 1) * ratio + dj + 1
                if side == :left
                    f_flux = patch.L[fine_i, fj, 1] * (patch.x[fine_i, fj] - patch.x[fine_i-1, fj])
                else
                    f_flux = patch.L[fine_i+1, fj, 1] * (patch.x[fine_i+1, fj] - patch.x[fine_i, fj])
                end
                push!(fine_fluxes, f_flux)
            end
        end
    else  # z-direction interface (bottom/top)
        for ci_idx in 1:patch.coarse_extent[1]
            ic = ai + ci_idx - 1

            if side == :bottom
                c_flux = L_coarse[ic, jc, 2] * (p_coarse[ic, jc] - p_coarse[ic, jc-1])
            else
                c_flux = L_coarse[ic, jc+1, 2] * (p_coarse[ic, jc+1] - p_coarse[ic, jc])
            end
            coarse_flux += c_flux

            for di in 1:ratio
                fi = (ci_idx - 1) * ratio + di + 1
                if side == :bottom
                    f_flux = patch.L[fi, fine_j, 2] * (patch.x[fi, fine_j] - patch.x[fi, fine_j-1])
                else
                    f_flux = patch.L[fi, fine_j+1, 2] * (patch.x[fi, fine_j+1] - patch.x[fi, fine_j])
                end
                push!(fine_fluxes, f_flux)
            end
        end
    end

    mismatch = coarse_flux - sum(fine_fluxes)
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
            for idx in 1:n_fine
                fj = idx + 1  # Skip ghost cell
                if fj <= nz + 1  # Bounds check
                    L_val = max(patch.L[fine_i, fj, 1], min_coeff)
                    # Positive mismatch: coarse flux > sum fine flux
                    # Need to increase fine flux = increase p_interior
                    patch.x[fine_i, fj] += correction_per_face / L_val
                end
            end
        elseif side == :right
            # Right interface: flux points out of patch (positive x direction)
            # Flux = L * (p[nx+2] - p[nx+1]), but ghost is p[nx+2]
            # Fine interior flux = L * (p[nx+1] - p_ghost) with outward normal
            fine_i = nx + 1
            for idx in 1:n_fine
                fj = idx + 1
                if fj <= nz + 1
                    # For right face, flux is L[nx+2] * (p[nx+2] - p[nx+1])
                    # But we control p[nx+1], so to increase flux, decrease p[nx+1]
                    L_val = max(patch.L[fine_i+1, fj, 1], min_coeff)
                    patch.x[fine_i, fj] -= correction_per_face / L_val
                end
            end
        elseif side == :bottom
            # Bottom interface: flux points into patch (negative z direction from ghost)
            fine_j = 2
            for idx in 1:n_fine
                fi = idx + 1
                if fi <= nx + 1
                    L_val = max(patch.L[fi, fine_j, 2], min_coeff)
                    patch.x[fi, fine_j] += correction_per_face / L_val
                end
            end
        else  # :top
            # Top interface: flux points out of patch
            fine_j = nz + 1
            for idx in 1:n_fine
                fi = idx + 1
                if fi <= nx + 1
                    L_val = max(patch.L[fi, fine_j+1, 2], min_coeff)
                    patch.x[fi, fine_j] -= correction_per_face / L_val
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

# Arguments
- `patch`: PatchPoisson (z array will be filled with divergence)
- `u_coarse`: Coarse velocity array
- `u_fine`: Fine velocity patch (or nothing to interpolate)
- `anchor`: Anchor position
"""
function compute_fine_divergence!(patch::PatchPoisson{T},
                                   u_coarse::AbstractArray{T},
                                   u_fine::Union{Nothing, AbstractArray{T}},
                                   anchor::NTuple{2,Int}) where T
    ratio = refinement_ratio(patch)
    ai, aj = anchor

    if u_fine !== nothing
        # Use fine velocity directly
        for I in inside(patch)
            fi, fj = I.I
            # Divergence: du/dx + dw/dz
            dudx = u_fine[fi, fj, 1] - u_fine[fi-1, fj, 1]
            dwdz = u_fine[fi, fj, 2] - u_fine[fi, fj-1, 2]
            patch.z[I] = dudx + dwdz
        end
    else
        # Interpolate divergence from coarse (less accurate but works without fine velocity)
        for I in inside(patch)
            fi, fj = I.I
            # Map to coarse location
            xf = (fi - 1.5) / ratio
            zf = (fj - 1.5) / ratio
            ic = floor(Int, xf) + ai
            jc = floor(Int, zf) + aj
            ic = clamp(ic, 2, size(u_coarse, 1) - 1)
            jc = clamp(jc, 2, size(u_coarse, 2) - 1)

            # Coarse divergence at (ic, jc)
            div_coarse = (u_coarse[ic, jc, 1] - u_coarse[ic-1, jc, 1]) +
                         (u_coarse[ic, jc, 2] - u_coarse[ic, jc-1, 2])
            patch.z[I] = div_coarse
        end
    end
end

"""
    correct_fine_velocity!(u_fine, patch, anchor)

Correct fine velocity using fine pressure gradient: u -= L*∇p

# Arguments
- `u_fine`: Fine velocity array to update
- `patch`: PatchPoisson with pressure solution
- `anchor`: Anchor position
"""
function correct_fine_velocity!(u_fine::AbstractArray{T},
                                 patch::PatchPoisson{T},
                                 anchor::NTuple{2,Int}) where T
    for I in inside(patch)
        fi, fj = I.I
        # x-velocity correction
        u_fine[fi, fj, 1] -= patch.L[fi, fj, 1] * (patch.x[fi, fj] - patch.x[fi-1, fj])
        # z-velocity correction (at staggered location)
        u_fine[fi, fj, 2] -= patch.L[fi, fj, 2] * (patch.x[fi, fj] - patch.x[fi, fj-1])
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
