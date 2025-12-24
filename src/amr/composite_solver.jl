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

    # Process patches by level (coarse to fine)
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

    # 1. Set divergence on base grid
    @inside flow.σ[I] = div(I, flow.u) / dt
    cp.base.z .= flow.σ

    # 2. Set divergence on refined patches (interpolated from base for now)
    for (anchor, patch) in cp.patches
        set_patch_divergence!(patch, flow.u, anchor, dt)
    end

    # 3. Solve composite Poisson system
    solver!(cp)

    # 4. Correct base grid velocity
    scale = inv(dt)
    correct_velocity!(flow, cp.base.x, cp.base.L, scale)

    # 5. Correct refined velocity patches
    for (anchor, patch) in cp.patches
        vel_patch = get_patch(cp.refined_velocity, anchor)
        if vel_patch !== nothing
            correct_refined_velocity!(vel_patch, patch, scale)
        end
    end

    # 6. Enforce interface velocity consistency
    enforce_velocity_consistency!(flow.u, cp.refined_velocity, cp.patches)

    # 7. Store pressure
    flow.p .= cp.base.x
end

"""
    set_patch_divergence!(patch, u_coarse, anchor, dt)

Set divergence source term on patch.
Currently interpolates from coarse; will use fine velocity when available.
"""
function set_patch_divergence!(patch::PatchPoisson{T},
                               u_coarse::AbstractArray{T},
                               anchor::NTuple{2,Int},
                               dt::T) where T
    ratio = refinement_ratio(patch)
    ai, aj = anchor
    inv_dt = inv(dt)

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
        patch.z[I] = div_coarse * inv_dt
    end
end

"""
    correct_velocity!(flow, p, L, scale)

Correct velocity using pressure gradient: u -= scale * L * ∇p
Uses forward difference to match standard project! convention.
"""
function correct_velocity!(flow::Flow{D,T}, p::AbstractArray{T},
                           L::AbstractArray{T}, scale::T) where {D,T}
    for I in inside(p)
        for d in 1:D
            δd = δ(d, I)  # Unit offset in direction d
            # FORWARD difference: p[I+1] - p[I]
            flow.u[I, d] -= scale * L[I, d] * (p[I+δd] - p[I])
        end
    end
end

"""
    correct_refined_velocity!(vel_patch, pois_patch, scale)

Correct refined velocity using fine pressure gradient.
Uses forward difference to match standard project! convention.
"""
function correct_refined_velocity!(vel_patch::RefinedVelocityPatch{T,2},
                                   pois_patch::PatchPoisson{T},
                                   scale::T) where T
    p, L = pois_patch.x, pois_patch.L

    for I in inside(pois_patch)
        fi, fj = I.I
        # x-velocity correction - FORWARD difference
        vel_patch.u[fi, fj, 1] -= scale * L[fi, fj, 1] * (p[fi+1, fj] - p[fi, fj])
        # z-velocity correction - FORWARD difference
        vel_patch.u[fi, fj, 2] -= scale * L[fi, fj, 2] * (p[fi, fj+1] - p[fi, fj])
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
                coarse_flux = u_coarse[ic+1, jc, 1]

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
                coarse_flux = u_coarse[ic, jc+1, 2]

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
            fi, fj = I.I
            # Fine divergence
            d = (vel_patch.u[fi, fj, 1] - vel_patch.u[fi-1, fj, 1]) +
                (vel_patch.u[fi, fj, 2] - vel_patch.u[fi, fj-1, 2])
            max_div = max(max_div, abs(d))
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
