# =============================================================================
# COMPOSITE SOLVER ALGORITHM
# =============================================================================
# This file implements the main solve algorithm for the composite AMR system.
#
# Algorithm overview:
# 1. V-cycle on base grid (multigrid captures smooth errors efficiently)
# 2. For each patch (coarse to fine order):
#    a. Prolongate parent pressure to patch ghost cells
#    b. Compute patch residual (defect)
#    c. Local PCG smoothing on patch
#    d. Compute flux correction at interfaces
#    e. Restrict correction to parent level
# 3. Check convergence on base grid
# 4. Repeat until converged
#
# Key functions:
# - solver!: Main iterative solve loop
# - patch_defect_correction!: Process all patches
# - project!: AMR-aware velocity projection step
# - enforce_velocity_consistency!: Flux matching at interfaces
#
# The projection step (project!) is the main entry point for time-stepping,
# as it projects the velocity field to be divergence-free.
# =============================================================================

"""
    Composite Solver Algorithm for AMR

Main solver algorithm combining base grid V-cycle with patch defect correction.
Implements iterative defect correction with flux matching at coarse-fine interfaces.
"""

"""
    solver!(cp::CompositePoisson; tol=1e-4, itmx=32)

Solve the composite grid Poisson system using defect correction.

Algorithm:
1. V-cycle on base grid
2. Patch defect corrections (prolongate, smooth, flux match, restrict)
3. Repeat until convergence

# Arguments
- `cp`: CompositePoisson solver
- `tol`: Convergence tolerance (relative to initial residual)
- `itmx`: Maximum number of outer iterations
"""
function solver!(cp::CompositePoisson{T}; tol::T=T(1e-4), itmx::Int=32) where T
    base = cp.base

    # Initial residual on base grid
    residual!(base.levels[1])
    r0 = max(L₂(base.levels[1]), 10eps(T))

    np = 0
    for _ in 1:itmx
        np += 1

        # 1. V-cycle on base grid (captures smooth errors)
        Vcycle!(base)

        # 2. Patch defect corrections (captures fine-scale errors)
        if has_patches(cp)
            patch_defect_correction!(cp)
        end

        # 3. Check convergence
        r2 = L₂(base.levels[1])
        if r2 / r0 < tol || r2 < 10eps(T)
            break
        end
    end

    push!(cp.n, Int16(np))
end

"""
    patch_defect_correction!(cp::CompositePoisson)

Perform defect correction on all patches.
For multi-level AMR, processes patches from coarse to fine levels.
"""
function patch_defect_correction!(cp::CompositePoisson{T}) where T
    base_level = cp.base.levels[1]

    # Process 2D patches by level (coarse to fine)
    for level in 1:cp.max_level
        for (anchor, patch) in patches_at_level(cp, level)
            # Determine parent pressure (base for level 1, parent patch for higher levels)
            p_parent = base_level.x
            L_parent = base_level.L

            # 1. Prolongate parent pressure to patch ghost cells
            prolongate_pressure_to_boundary!(patch, p_parent, anchor)

            # 2. Compute patch residual (defect)
            patch_residual!(patch)

            # 3. Local smoothing on patch (use PCG for good convergence)
            for _ in 1:4
                patch_pcg!(patch; it=2)
            end

            # 4. Compute and apply flux correction
            fluxes = compute_all_interface_fluxes(patch, p_parent, L_parent, anchor)
            apply_flux_correction!(patch, fluxes)

            # 5. Restrict correction to parent level
            restrict_pressure_to_coarse!(p_parent, patch, anchor)
        end

        # Process 3D patches at this level
        for (anchor, patch) in patches_at_level_3d(cp, level)
            p_parent = base_level.x
            L_parent = base_level.L

            # 1. Prolongate parent pressure to patch ghost cells
            prolongate_pressure_to_boundary_3d!(patch, p_parent, anchor)

            # 2. Compute patch residual (defect)
            patch_residual_3d!(patch)

            # 3. Local smoothing on patch (use PCG for good convergence)
            for _ in 1:4
                patch_pcg_3d!(patch; it=2)
            end

            # 4. Compute and apply flux correction
            fluxes = compute_all_interface_fluxes_3d(patch, p_parent, L_parent, anchor)
            apply_flux_correction_3d!(patch, fluxes)

            # 5. Restrict correction to parent level
            restrict_pressure_to_coarse_3d!(p_parent, patch, anchor)
        end
    end

    # Recompute base residual after restriction
    residual!(base_level)
end

# Note: patch_smooth! is defined in patch_poisson.jl
# This function provides the default smoothing for defect correction

"""
    project!(flow::Flow, cp::CompositePoisson, refined_grid::RefinedGrid, w=1)

AMR-aware projection step.
Projects velocity to divergence-free state using composite solver.

# Arguments
- `flow`: Flow object (base grid velocity)
- `cp`: CompositePoisson solver
- `refined_grid`: RefinedGrid with cell refinement info
- `w`: Time step weight (1 for predictor, 0.5 for corrector)
"""
function project!(flow::Flow{D,T}, cp::CompositePoisson{T},
                  refined_grid, w::Real=1) where {D,T}
    dt = T(w * flow.Δt[end])
    ρ = flow.ρ
    inv_ρ = inv(ρ)

    # 1. Set divergence on base grid
    @inside flow.σ[I] = ρ * div(I, flow.u)
    cp.base.z .= flow.σ

    # 2. Scale pressure initial guess for warm start
    cp.base.x .*= dt
    for (_, patch) in cp.patches
        patch.x .*= dt
    end
    for (_, patch) in cp.patches_3d
        patch.x .*= dt
    end

    # 3. Interpolate velocity to refined patches
    interpolate_velocity_to_patches!(cp.refined_velocity, flow.u)

    # 4. Set divergence on refined patches
    for (anchor, patch) in cp.patches
        vel_patch = get_patch(cp.refined_velocity, anchor)
        u_fine = vel_patch === nothing ? nothing : vel_patch.u
        set_patch_divergence!(patch, flow.u, u_fine, anchor, ρ)
    end
    for (anchor, patch) in cp.patches_3d
        vel_patch = get_patch(cp.refined_velocity, anchor)
        u_fine = vel_patch === nothing ? nothing : vel_patch.u
        set_patch_divergence_3d!(patch, flow.u, u_fine, anchor, ρ)
    end

    # 5. Solve composite Poisson system
    solver!(cp)

    # 6. Correct base grid velocity
    correct_velocity!(flow, cp.base.x, cp.base.L, inv_ρ)

    # 7. Correct refined velocity patches
    for (anchor, patch) in cp.patches
        vel_patch = get_patch(cp.refined_velocity, anchor)
        if vel_patch !== nothing
            correct_refined_velocity!(vel_patch, patch, inv_ρ)
        end
    end
    for (anchor, patch) in cp.patches_3d
        vel_patch = get_patch(cp.refined_velocity, anchor)
        if vel_patch !== nothing
            correct_refined_velocity_3d!(vel_patch, patch, inv_ρ)
        end
    end

    # 8. Enforce interface velocity consistency
    enforce_velocity_consistency!(flow.u, cp.refined_velocity, cp.patches)
    enforce_velocity_consistency_3d!(flow.u, cp.refined_velocity, cp.patches_3d)

    # 9. Unscale pressure for storage
    cp.base.x ./= dt
    for (_, patch) in cp.patches
        patch.x ./= dt
    end
    for (_, patch) in cp.patches_3d
        patch.x ./= dt
    end

    # 10. Store pressure
    flow.p .= cp.base.x
end

"""
    set_patch_divergence!(patch, u_coarse, u_fine, anchor, ρ)

Set divergence source term on patch.
Uses fine velocity when available, otherwise interpolates from coarse.
"""
function set_patch_divergence!(patch::PatchPoisson{T},
                               u_coarse::AbstractArray{T},
                               u_fine::Union{Nothing, AbstractArray{T}},
                               anchor::NTuple{2,Int},
                               ρ::T) where T
    compute_fine_divergence!(patch, u_coarse, u_fine, anchor)
    @inside patch.z[I] *= ρ
end

"""
    set_patch_divergence_3d!(patch, u_coarse, u_fine, anchor, ρ)

Set divergence source term on 3D patch.
Uses fine velocity when available, otherwise interpolates from coarse.
"""
function set_patch_divergence_3d!(patch::PatchPoisson3D{T},
                                   u_coarse::AbstractArray{T},
                                   u_fine::Union{Nothing, AbstractArray{T}},
                                   anchor::NTuple{3,Int},
                                   ρ::T) where T
    compute_fine_divergence_3d!(patch, u_coarse, u_fine, anchor)
    @inside patch.z[I] *= ρ
end

"""
    correct_velocity!(flow, p, L, scale)

Correct velocity using pressure gradient: u -= scale * L * ∇p
Uses the same formula as standard project!.
"""
function correct_velocity!(flow::Flow{D,T}, p::AbstractArray{T},
                           L::AbstractArray{T}, scale::T) where {D,T}
    # Use the same formula as standard project!: u -= L*∂(d,I,p)*scale
    for d in 1:D
        @loop flow.u[I, d] -= scale * L[I, d] * ∂(d, I, p) over I ∈ inside(p)
    end
end

"""
    correct_refined_velocity!(vel_patch, pois_patch, scale)

Correct refined velocity using fine pressure gradient.
Uses the same formula as standard project!.
"""
function correct_refined_velocity!(vel_patch::RefinedVelocityPatch{T,2},
                                   pois_patch::PatchPoisson{T},
                                   scale::T) where T
    p, L = pois_patch.x, pois_patch.L

    # Use backward difference like standard project!
    for I in inside(pois_patch)
        fi, fj = I.I
        # x-velocity correction: ∂(1,I,p) = p[I] - p[I-1]
        vel_patch.u[fi, fj, 1] -= scale * L[fi, fj, 1] * (p[fi, fj] - p[fi-1, fj])
        # z-velocity correction: ∂(2,I,p) = p[I] - p[I-(0,1)]
        vel_patch.u[fi, fj, 2] -= scale * L[fi, fj, 2] * (p[fi, fj] - p[fi, fj-1])
    end
end

"""
    correct_refined_velocity_3d!(vel_patch, pois_patch, scale)

Correct refined 3D velocity using fine pressure gradient.
Uses the same formula as standard project!.
"""
function correct_refined_velocity_3d!(vel_patch::RefinedVelocityPatch{T,3},
                                       pois_patch::PatchPoisson3D{T},
                                       scale::T) where T
    p, L = pois_patch.x, pois_patch.L

    # Use backward difference like standard project!
    for I in inside(pois_patch)
        fi, fj, fk = I.I
        # x-velocity correction
        vel_patch.u[fi, fj, fk, 1] -= scale * L[fi, fj, fk, 1] * (p[fi, fj, fk] - p[fi-1, fj, fk])
        # y-velocity correction
        vel_patch.u[fi, fj, fk, 2] -= scale * L[fi, fj, fk, 2] * (p[fi, fj, fk] - p[fi, fj-1, fk])
        # z-velocity correction
        vel_patch.u[fi, fj, fk, 3] -= scale * L[fi, fj, fk, 3] * (p[fi, fj, fk] - p[fi, fj, fk-1])
    end
end

"""
    enforce_velocity_consistency!(u_coarse, refined_velocity, patches)

Ensure velocity flux is consistent at coarse-fine interfaces.
Fine fluxes should sum to match coarse flux.
"""
function enforce_velocity_consistency!(u_coarse::AbstractArray{T},
                                       refined_velocity::RefinedVelocityField,
                                       patches::Dict) where T
    for (anchor, patch) in patches
        vel_patch = get_patch(refined_velocity, anchor)
        vel_patch === nothing && continue

        ratio = refinement_ratio(patch)
        ai, aj = anchor

        # Process each interface
        # Left interface (x-direction)
        ic = ai
        if ic >= 2
            for cj_idx in 1:patch.coarse_extent[2]
                jc = aj + cj_idx - 1
                coarse_flux = u_coarse[ic, jc, 1]

                # Sum fine fluxes
                fine_sum = zero(T)
                for dj in 1:ratio
                    fj = (cj_idx - 1) * ratio + dj + 1
                    fine_sum += vel_patch.u[2, fj, 1]
                end

                # Distribute correction
                correction = (coarse_flux * ratio - fine_sum) / ratio
                for dj in 1:ratio
                    fj = (cj_idx - 1) * ratio + dj + 1
                    vel_patch.u[2, fj, 1] += correction
                end
            end
        end

        # Right interface
        ic = ai + patch.coarse_extent[1]
        if ic <= size(u_coarse, 1) - 1
            for cj_idx in 1:patch.coarse_extent[2]
                jc = aj + cj_idx - 1
                coarse_flux = u_coarse[ic, jc, 1]

                fine_sum = zero(T)
                for dj in 1:ratio
                    fj = (cj_idx - 1) * ratio + dj + 1
                    fi = patch.fine_dims[1] + 1
                    fine_sum += vel_patch.u[fi, fj, 1]
                end

                correction = (coarse_flux * ratio - fine_sum) / ratio
                for dj in 1:ratio
                    fj = (cj_idx - 1) * ratio + dj + 1
                    fi = patch.fine_dims[1] + 1
                    vel_patch.u[fi, fj, 1] += correction
                end
            end
        end

        # Bottom interface (z-direction)
        jc = aj
        if jc >= 2
            for ci_idx in 1:patch.coarse_extent[1]
                ic = ai + ci_idx - 1
                coarse_flux = u_coarse[ic, jc, 2]

                fine_sum = zero(T)
                for di in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fine_sum += vel_patch.u[fi, 2, 2]
                end

                correction = (coarse_flux * ratio - fine_sum) / ratio
                for di in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    vel_patch.u[fi, 2, 2] += correction
                end
            end
        end

        # Top interface
        jc = aj + patch.coarse_extent[2]
        if jc <= size(u_coarse, 2) - 1
            for ci_idx in 1:patch.coarse_extent[1]
                ic = ai + ci_idx - 1
                coarse_flux = u_coarse[ic, jc, 2]

                fine_sum = zero(T)
                for di in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fj = patch.fine_dims[2] + 1
                    fine_sum += vel_patch.u[fi, fj, 2]
                end

                correction = (coarse_flux * ratio - fine_sum) / ratio
                for di in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fj = patch.fine_dims[2] + 1
                    vel_patch.u[fi, fj, 2] += correction
                end
            end
        end
    end
end

"""
    enforce_velocity_consistency_3d!(u_coarse, refined_velocity, patches_3d)

Ensure velocity flux is consistent at 3D coarse-fine interfaces.
Fine fluxes should sum to match coarse flux across all 6 faces.
"""
function enforce_velocity_consistency_3d!(u_coarse::AbstractArray{T},
                                           refined_velocity::RefinedVelocityField,
                                           patches_3d::Dict) where T
    for (anchor, patch) in patches_3d
        vel_patch = get_patch(refined_velocity, anchor)
        vel_patch === nothing && continue

        ratio = refinement_ratio(patch)
        ratio2 = ratio * ratio
        ai, aj, ak = anchor
        nx, ny, nz = patch.fine_dims

        # X-minus interface
        ic = ai
        if ic >= 2
            for cj_idx in 1:patch.coarse_extent[2], ck_idx in 1:patch.coarse_extent[3]
                jc = aj + cj_idx - 1
                kc = ak + ck_idx - 1
                coarse_flux = u_coarse[ic, jc, kc, 1]

                fine_sum = zero(T)
                for dj in 1:ratio, dk in 1:ratio
                    fj = (cj_idx - 1) * ratio + dj + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    fine_sum += vel_patch.u[2, fj, fk, 1]
                end

                correction = (coarse_flux * ratio2 - fine_sum) / ratio2
                for dj in 1:ratio, dk in 1:ratio
                    fj = (cj_idx - 1) * ratio + dj + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    vel_patch.u[2, fj, fk, 1] += correction
                end
            end
        end

        # X-plus interface
        ic = ai + patch.coarse_extent[1]
        if ic <= size(u_coarse, 1) - 1
            for cj_idx in 1:patch.coarse_extent[2], ck_idx in 1:patch.coarse_extent[3]
                jc = aj + cj_idx - 1
                kc = ak + ck_idx - 1
                coarse_flux = u_coarse[ic, jc, kc, 1]

                fine_sum = zero(T)
                for dj in 1:ratio, dk in 1:ratio
                    fj = (cj_idx - 1) * ratio + dj + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    fine_sum += vel_patch.u[nx+1, fj, fk, 1]
                end

                correction = (coarse_flux * ratio2 - fine_sum) / ratio2
                for dj in 1:ratio, dk in 1:ratio
                    fj = (cj_idx - 1) * ratio + dj + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    vel_patch.u[nx+1, fj, fk, 1] += correction
                end
            end
        end

        # Y-minus interface
        jc = aj
        if jc >= 2
            for ci_idx in 1:patch.coarse_extent[1], ck_idx in 1:patch.coarse_extent[3]
                ic = ai + ci_idx - 1
                kc = ak + ck_idx - 1
                coarse_flux = u_coarse[ic, jc, kc, 2]

                fine_sum = zero(T)
                for di in 1:ratio, dk in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    fine_sum += vel_patch.u[fi, 2, fk, 2]
                end

                correction = (coarse_flux * ratio2 - fine_sum) / ratio2
                for di in 1:ratio, dk in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    vel_patch.u[fi, 2, fk, 2] += correction
                end
            end
        end

        # Y-plus interface
        jc = aj + patch.coarse_extent[2]
        if jc <= size(u_coarse, 2) - 1
            for ci_idx in 1:patch.coarse_extent[1], ck_idx in 1:patch.coarse_extent[3]
                ic = ai + ci_idx - 1
                kc = ak + ck_idx - 1
                coarse_flux = u_coarse[ic, jc, kc, 2]

                fine_sum = zero(T)
                for di in 1:ratio, dk in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    fine_sum += vel_patch.u[fi, ny+1, fk, 2]
                end

                correction = (coarse_flux * ratio2 - fine_sum) / ratio2
                for di in 1:ratio, dk in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fk = (ck_idx - 1) * ratio + dk + 1
                    vel_patch.u[fi, ny+1, fk, 2] += correction
                end
            end
        end

        # Z-minus interface
        kc = ak
        if kc >= 2
            for ci_idx in 1:patch.coarse_extent[1], cj_idx in 1:patch.coarse_extent[2]
                ic = ai + ci_idx - 1
                jc = aj + cj_idx - 1
                coarse_flux = u_coarse[ic, jc, kc, 3]

                fine_sum = zero(T)
                for di in 1:ratio, dj in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fj = (cj_idx - 1) * ratio + dj + 1
                    fine_sum += vel_patch.u[fi, fj, 2, 3]
                end

                correction = (coarse_flux * ratio2 - fine_sum) / ratio2
                for di in 1:ratio, dj in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fj = (cj_idx - 1) * ratio + dj + 1
                    vel_patch.u[fi, fj, 2, 3] += correction
                end
            end
        end

        # Z-plus interface
        kc = ak + patch.coarse_extent[3]
        if kc <= size(u_coarse, 3) - 1
            for ci_idx in 1:patch.coarse_extent[1], cj_idx in 1:patch.coarse_extent[2]
                ic = ai + ci_idx - 1
                jc = aj + cj_idx - 1
                coarse_flux = u_coarse[ic, jc, kc, 3]

                fine_sum = zero(T)
                for di in 1:ratio, dj in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fj = (cj_idx - 1) * ratio + dj + 1
                    fine_sum += vel_patch.u[fi, fj, nz+1, 3]
                end

                correction = (coarse_flux * ratio2 - fine_sum) / ratio2
                for di in 1:ratio, dj in 1:ratio
                    fi = (ci_idx - 1) * ratio + di + 1
                    fj = (cj_idx - 1) * ratio + dj + 1
                    vel_patch.u[fi, fj, nz+1, 3] += correction
                end
            end
        end
    end
end

"""
    divergence_at_all_levels(flow, cp)

Compute maximum divergence at base and all refined levels.
Returns tuple (base_div, patch_divs) for verification.
"""
function divergence_at_all_levels(flow::Flow{D,T}, cp::CompositePoisson{T}) where {D,T}
    # Base grid divergence
    base_div = zero(T)
    for I in inside(flow.p)
        base_div = max(base_div, abs(div(I, flow.u)))
    end

    # Patch divergences
    patch_divs = Dict{Tuple{Int,Int}, T}()
    for (anchor, patch) in cp.patches
        vel_patch = get_patch(cp.refined_velocity, anchor)
        vel_patch === nothing && continue

        max_div = zero(T)
        for I in inside(patch)
            max_div = max(max_div, abs(div(I, vel_patch.u)))
        end
        patch_divs[anchor] = max_div
    end

    return base_div, patch_divs
end

"""
    total_solve_iterations(cp::CompositePoisson)

Return total iterations from last solve across all components.
"""
function total_solve_iterations(cp::CompositePoisson)
    total = isempty(cp.n) ? 0 : Int(cp.n[end])
    for (_, patch) in cp.patches
        if !isempty(patch.n)
            total += Int(patch.n[end])
        end
    end
    return total
end
