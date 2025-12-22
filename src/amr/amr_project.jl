"""
    AMR Projection Functions

AMR-aware projection step that maintains divergence-free velocity
at both base and refined levels.
"""

"""
    amr_project!(flow::Flow, cp::CompositePoisson, w=1)

AMR-aware projection step using composite Poisson solver.
Projects velocity to divergence-free state at all refinement levels.

# Arguments
- `flow`: Flow object (base grid velocity and fields)
- `cp`: CompositePoisson solver (manages base + patches)
- `w`: Time step weight (1 for predictor, 0.5 for corrector)
"""
function amr_project!(flow::Flow{D,T}, cp::CompositePoisson{T}, w::Real=1) where {D,T}
    dt = w * flow.Δt[end]
    inv_dt = inv(dt)

    # 1. Set divergence source on base grid
    @inside flow.σ[I] = div(I, flow.u) * inv_dt
    cp.base.z .= flow.σ

    # 2. Set divergence on refined patches
    set_all_patch_divergence!(cp, flow.u, dt)

    # 3. Interpolate velocity to refined patches (before solve)
    interpolate_velocity_to_patches!(cp.refined_velocity, flow.u)

    # 4. Solve composite Poisson system
    solver!(cp)

    # 5. Correct base grid velocity: u -= dt * L * ∇p
    correct_base_velocity!(flow, cp.base.x, cp.base.L, dt)

    # 6. Correct refined velocity on patches
    correct_all_refined_velocity!(cp, dt)

    # 7. Enforce interface consistency (flux matching)
    enforce_all_interface_consistency!(flow.u, cp)

    # 8. Store pressure for output
    flow.p .= cp.base.x
end

"""
    set_all_patch_divergence!(cp, u_coarse, dt)

Set divergence source term on all patches.
"""
function set_all_patch_divergence!(cp::CompositePoisson{T}, u_coarse::AbstractArray{T}, dt::T) where T
    inv_dt = inv(dt)

    for (anchor, patch) in cp.patches
        ratio = refinement_ratio(patch)
        ai, aj = anchor

        for I in inside(patch)
            fi, fj = I.I

            # Map fine cell center to coarse location
            xf = (fi - 1.5) / ratio
            zf = (fj - 1.5) / ratio
            ic = floor(Int, xf) + ai
            jc = floor(Int, zf) + aj
            ic = clamp(ic, 2, size(u_coarse, 1) - 1)
            jc = clamp(jc, 2, size(u_coarse, 2) - 1)

            # Interpolated divergence from coarse
            # TODO: Use fine velocity when available
            div_c = (u_coarse[ic, jc, 1] - u_coarse[ic-1, jc, 1]) +
                    (u_coarse[ic, jc, 2] - u_coarse[ic, jc-1, 2])

            patch.z[I] = div_c * inv_dt
        end
    end
end

"""
    interpolate_velocity_to_patches!(refined_velocity, u_coarse)

Interpolate coarse velocity to all refined patches.
"""
function interpolate_velocity_to_patches!(refined_velocity::RefinedVelocityField{T,2},
                                           u_coarse::AbstractArray{T}) where T
    for (anchor, vel_patch) in refined_velocity.patches
        interpolate_from_coarse!(vel_patch, u_coarse, anchor)
        fill_ghost_cells!(vel_patch, u_coarse, anchor)
    end
end

"""
    correct_base_velocity!(flow, p, L, dt)

Correct base velocity using pressure gradient.
"""
function correct_base_velocity!(flow::Flow{D,T}, p::AbstractArray{T},
                                L::AbstractArray{T}, dt::T) where {D,T}
    scale = inv(dt)
    for I in inside(p)
        for d in 1:D
            δ = δi(d, I)
            flow.u[I, d] -= scale * L[I, d] * (p[I] - p[I-δ])
        end
    end
end

"""
    correct_all_refined_velocity!(cp, dt)

Correct velocity on all refined patches.
"""
function correct_all_refined_velocity!(cp::CompositePoisson{T}, dt::T) where T
    scale = inv(dt)

    for (anchor, patch) in cp.patches
        vel_patch = get_patch(cp.refined_velocity, anchor)
        vel_patch === nothing && continue

        p, L = patch.x, patch.L

        for I in inside(patch)
            fi, fj = I.I
            # x-velocity correction
            vel_patch.u[fi, fj, 1] -= scale * L[fi, fj, 1] * (p[fi, fj] - p[fi-1, fj])
            # z-velocity correction
            vel_patch.u[fi, fj, 2] -= scale * L[fi, fj, 2] * (p[fi, fj] - p[fi, fj-1])
        end
    end
end

"""
    enforce_all_interface_consistency!(u_coarse, cp)

Ensure velocity flux consistency at all interfaces.
"""
function enforce_all_interface_consistency!(u_coarse::AbstractArray{T},
                                             cp::CompositePoisson{T}) where T
    enforce_velocity_consistency!(u_coarse, cp.refined_velocity, cp.patches)
end

"""
    amr_mom_step!(flow::Flow, cp::CompositePoisson, body=nothing)

AMR-aware momentum step using predictor-corrector scheme.

# Arguments
- `flow`: Flow object
- `cp`: CompositePoisson solver
- `body`: Optional body for immersed boundary
"""
function amr_mom_step!(flow::Flow{D,T}, cp::CompositePoisson{T};
                       body=nothing, λ=quick) where {D,T}
    # Predictor step
    conv_diff!(flow.f, flow.u, flow.σ, λ; ν=flow.ν)

    # Apply BDIM if body present
    if body !== nothing && typeof(body) != NoBody
        BDIM!(flow, body)
    elseif hasfield(typeof(flow), :V)
        BDIM!(flow)
    end

    # Project to divergence-free
    amr_project!(flow, cp)

    # Corrector step
    conv_diff!(flow.f, flow.u, flow.σ, λ; ν=flow.ν)

    if body !== nothing && typeof(body) != NoBody
        BDIM!(flow, body)
    elseif hasfield(typeof(flow), :V)
        BDIM!(flow)
    end

    amr_project!(flow, cp, 0.5)

    # Update time step
    push!(flow.Δt, CFL(flow))
end

"""
    check_amr_divergence(flow, cp; verbose=false)

Check maximum divergence at base and all refined levels.
Useful for verification.

# Returns
- Maximum divergence across all levels
"""
function check_amr_divergence(flow::Flow{D,T}, cp::CompositePoisson{T};
                              verbose::Bool=false) where {D,T}
    # Base grid divergence
    base_div = zero(T)
    for I in inside(flow.p)
        base_div = max(base_div, abs(div(I, flow.u)))
    end

    if verbose
        println("Base grid max |∇·u|: ", base_div)
    end

    max_div = base_div

    # Patch divergences
    for (anchor, patch) in cp.patches
        vel_patch = get_patch(cp.refined_velocity, anchor)
        vel_patch === nothing && continue

        patch_div = zero(T)
        for I in inside(patch)
            fi, fj = I.I
            # Fine divergence
            d = (vel_patch.u[fi, fj, 1] - vel_patch.u[fi-1, fj, 1]) +
                (vel_patch.u[fi, fj, 2] - vel_patch.u[fi, fj-1, 2])
            patch_div = max(patch_div, abs(d))
        end

        if verbose
            println("Patch at ", anchor, " max |∇·u|: ", patch_div)
        end

        max_div = max(max_div, patch_div)
    end

    return max_div
end

"""
    amr_cfl(flow, cp)

Compute CFL time step considering refined patches.
"""
function amr_cfl(flow::Flow{D,T}, cp::CompositePoisson{T}) where {D,T}
    # Base CFL
    dt = CFL(flow)

    # Consider refined patches (finer grid = smaller dt)
    for (anchor, patch) in cp.patches
        vel_patch = get_patch(cp.refined_velocity, anchor)
        vel_patch === nothing && continue

        ratio = refinement_ratio(patch)
        # Fine grid spacing is 1/ratio of coarse
        # CFL scales linearly with grid spacing
        dt = min(dt, dt / ratio)
    end

    return dt
end

"""
    synchronize_base_and_patches!(flow, cp)

Synchronize solution between base grid and patches.
Ensures consistency after regridding.
"""
function synchronize_base_and_patches!(flow::Flow{D,T}, cp::CompositePoisson{T}) where {D,T}
    # Interpolate base pressure to patches
    for (anchor, patch) in cp.patches
        set_patch_boundary!(patch, cp.base.x, anchor)

        # Initialize patch interior from coarse
        ratio = refinement_ratio(patch)
        ai, aj = anchor

        for I in inside(patch)
            fi, fj = I.I
            xf = (fi - 1.5) / ratio
            zf = (fj - 1.5) / ratio
            ic = floor(Int, xf) + ai
            jc = floor(Int, zf) + aj
            ic = clamp(ic, 1, size(cp.base.x, 1))
            jc = clamp(jc, 1, size(cp.base.x, 2))

            # Simple copy (could use interpolation)
            patch.x[I] = cp.base.x[ic, jc]
        end
    end

    # Interpolate base velocity to refined patches
    interpolate_velocity_to_patches!(cp.refined_velocity, flow.u)
end

"""
    regrid_amr!(cp, rg, flow, indicators, threshold)

Full AMR regridding: mark cells, create patches, synchronize.

# Arguments
- `cp`: CompositePoisson to update
- `rg`: RefinedGrid to update
- `flow`: Flow object (provides μ₀ and velocity)
- `indicators`: Dict of cell -> indicator values
- `threshold`: Refinement threshold
"""
function regrid_amr!(cp::CompositePoisson{T}, rg::RefinedGrid,
                     flow::Flow{D,T}, indicators::Dict, threshold::Real) where {D,T}
    # 1. Mark cells for refinement
    mark_cells_for_refinement!(rg, indicators, threshold, cp.max_level)

    # 2. Ensure proper nesting
    ensure_proper_nesting!(rg)

    # 3. Create patches from marked cells
    create_patches!(cp, rg, flow.μ₀)

    # 4. Synchronize data
    synchronize_base_and_patches!(flow, cp)
end
