# =============================================================================
# AMR PROJECTION FUNCTIONS
# =============================================================================
# This file implements AMR-aware projection to enforce velocity divergence-free
# constraint across all refinement levels. The projection step is critical for
# incompressible flow simulation.
#
# Algorithm:
# 1. Compute divergence on base grid and refined patches
# 2. Interpolate velocity to refined patches (for consistency)
# 3. Solve composite Poisson system (pressure correction)
# 4. Correct velocity: u -= (1/dt) * L * ∇p on all levels
# 5. Enforce flux matching at coarse-fine interfaces
#
# Key functions:
# - amr_project!: Main projection step (used in time-stepping)
# - amr_mom_step!: Full momentum step with AMR projection
# - check_amr_divergence: Verify divergence-free constraint
# - amr_cfl: Compute time step considering refined grids
# - regrid_amr!: Dynamic regridding based on indicators
#
# The projection ensures:
# - ∇·u = 0 at all grid levels
# - Flux conservation at coarse-fine interfaces
# - Proper boundary conditions on patches
# =============================================================================

"""
    AMR Projection Functions

AMR-aware projection step that maintains divergence-free velocity
at both base and refined levels.
"""

"""
    amr_project!(flow::Flow, cp::CompositePoisson, w=1)

AMR-aware projection step using composite Poisson solver.
Projects velocity to divergence-free state at all refinement levels.

Matches the standard project! convention:
- Poisson equation: L·p = ρ·div(u)
- Velocity correction: u -= (L/ρ)·∇p

# Arguments
- `flow`: Flow object (base grid velocity and fields)
- `cp`: CompositePoisson solver (manages base + patches)
- `w`: Time step weight (1 for predictor, 0.5 for corrector)
"""
function amr_project!(flow::Flow{D,T}, cp::CompositePoisson{T}, w::Real=1) where {D,T}
    dt = T(w * flow.Δt[end])
    ρ = flow.ρ
    inv_ρ = inv(ρ)

    # 1. Set divergence source on base grid: RHS = ρ * div(u)
    # This matches the standard project! convention
    @inside flow.σ[I] = ρ * div(I, flow.u)
    cp.base.z .= flow.σ

    # 2. Scale pressure initial guess for warm start (like standard project!)
    cp.base.x .*= dt
    # Also scale patch pressures for consistency
    for (_, patch) in cp.patches
        patch.x .*= dt
    end

    # 3. Interpolate velocity to refined patches (before divergence/solve)
    interpolate_velocity_to_patches!(cp.refined_velocity, flow.u)

    # 4. Set divergence on refined patches (using same ρ scaling)
    set_all_patch_divergence!(cp, flow.u, ρ)

    # 5. Solve composite Poisson system
    solver!(cp)

    # 6. Correct base grid velocity: u -= (L/ρ) * ∇p
    correct_base_velocity!(flow, cp.base.x, cp.base.L, inv_ρ)

    # 7. Correct refined velocity on patches
    correct_all_refined_velocity!(cp, inv_ρ)

    # 8. Enforce interface consistency (flux matching)
    enforce_all_interface_consistency!(flow.u, cp)

    # 9. Unscale pressure for storage (like standard project!)
    cp.base.x ./= dt
    # Also unscale patch pressures
    for (_, patch) in cp.patches
        patch.x ./= dt
    end

    # 10. Store pressure for output
    flow.p .= cp.base.x
end

"""
    set_all_patch_divergence!(cp, u_coarse, ρ)

Set divergence source term on all patches.
RHS = ρ * div(u) to match standard projection convention.
Uses refined patch velocity when available; otherwise falls back to coarse.
"""
function set_all_patch_divergence!(cp::CompositePoisson{T}, u_coarse::AbstractArray{T}, ρ::T) where T
    for (anchor, patch) in cp.patches
        vel_patch = get_patch(cp.refined_velocity, anchor)
        ratio = refinement_ratio(patch)
        ai, aj = anchor

        for I in inside(patch)
            if vel_patch !== nothing
                # Use fine velocity directly when available.
                patch.z[I] = ρ * div(I, vel_patch.u)
            else
                fi, fj = I.I

                # Map fine cell center to coarse location
                xf = (fi - 1.5) / ratio
                zf = (fj - 1.5) / ratio
                ic = floor(Int, xf) + ai
                jc = floor(Int, zf) + aj
                ic = clamp(ic, 2, size(u_coarse, 1) - 1)
                jc = clamp(jc, 2, size(u_coarse, 2) - 1)

                # RHS = ρ * div(u) to match standard convention
                patch.z[I] = ρ * div(CartesianIndex(ic, jc), u_coarse)
            end
        end
    end
end

"""
    interpolate_velocity_to_patches!(refined_velocity, u_coarse)

Interpolate coarse velocity to all refined patches (2D version).
"""
function interpolate_velocity_to_patches!(refined_velocity::RefinedVelocityField{T,2},
                                           u_coarse::AbstractArray{T}) where T
    for (anchor, vel_patch) in refined_velocity.patches
        interpolate_from_coarse!(vel_patch, u_coarse, anchor)
        fill_ghost_cells!(vel_patch, u_coarse, anchor)
    end
end

"""
    interpolate_velocity_to_patches!(refined_velocity, u_coarse) - 3D version

Interpolate coarse velocity to all refined patches (3D version).
"""
function interpolate_velocity_to_patches!(refined_velocity::RefinedVelocityField{T,3},
                                           u_coarse::AbstractArray{T}) where T
    for (anchor, vel_patch) in refined_velocity.patches
        interpolate_from_coarse!(vel_patch, u_coarse, anchor)
        fill_ghost_cells!(vel_patch, u_coarse, anchor)
    end
end

"""
    correct_base_velocity!(flow, p, L, inv_ρ)

Correct base velocity using pressure gradient.
Velocity correction: u -= (L/ρ) * ∇p = inv_ρ * L * ∇p
Uses the same formula as standard project!: u -= L*∂(d,I,p)/ρ
"""
function correct_base_velocity!(flow::Flow{D,T}, p::AbstractArray{T},
                                L::AbstractArray{T}, inv_ρ::T) where {D,T}
    # Use the same formula as standard project!
    # a.u[I,i] -= b.L[I,i]*∂(i,I,b.x)/ρ where ∂(i,I,x) = x[I+δ(i,I)] - x[I]
    for d in 1:D
        @loop flow.u[I, d] -= inv_ρ * L[I, d] * ∂(d, I, p) over I ∈ inside(p)
    end
end

"""
    correct_all_refined_velocity!(cp, inv_ρ)

Correct velocity on all refined patches.
Velocity correction: u -= (L/ρ) * ∇p = inv_ρ * L * ∇p
Uses the same formula as standard project!.
"""
function correct_all_refined_velocity!(cp::CompositePoisson{T}, inv_ρ::T) where T
    for (anchor, patch) in cp.patches
        vel_patch = get_patch(cp.refined_velocity, anchor)
        vel_patch === nothing && continue

        p, L = patch.x, patch.L

        # Use the same formula as standard project!
        # ∂(d,I,p) = p[I] - p[I-δ(d,I)]
        for I in inside(patch)
            fi, fj = I.I
            # x-velocity correction
            vel_patch.u[fi, fj, 1] -= inv_ρ * L[fi, fj, 1] * (p[fi, fj] - p[fi-1, fj])
            # z-velocity correction
            vel_patch.u[fi, fj, 2] -= inv_ρ * L[fi, fj, 2] * (p[fi, fj] - p[fi, fj-1])
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
    amr_mom_step!(flow::Flow, cp::CompositePoisson; λ=quick)

AMR-aware momentum step using predictor-corrector scheme.
Closely follows the standard mom_step! algorithm with proper Δx scaling.
The body is already incorporated into flow.μ₀ and flow.V via measure!().

# Arguments
- `flow`: Flow object (should have measure! called beforehand for body)
- `cp`: CompositePoisson solver
- `λ`: Convective scheme function (default: quick)
"""
function amr_mom_step!(flow::Flow{D,T}, cp::CompositePoisson{T};
                       λ=quick) where {D,T}
    # Save current velocity for predictor-corrector (like standard mom_step!)
    flow.u⁰ .= flow.u
    scale_u!(flow, zero(T))  # Zero out u for predictor start

    t₁ = sum(flow.Δt)
    t₀ = t₁ - flow.Δt[end]

    # Predictor step - compute forcing from u⁰
    conv_diff!(flow.f, flow.u⁰, flow.σ, λ; ν=flow.ν, Δx=flow.Δx, perdir=flow.perdir)

    # Apply body acceleration (gravity, etc.)
    accelerate!(flow.f, t₀, flow.g, flow.inletBC)

    # Apply BDIM (body info is in flow.μ₀ and flow.V)
    BDIM!(flow)
    BC!(flow.u, flow.inletBC, flow.outletBC, flow.perdir, t₁)

    # Apply convective outlet BC if enabled
    flow.outletBC && exitBC!(flow.u, flow.u⁰, flow.Δt[end])

    # Project to divergence-free
    amr_project!(flow, cp)
    BC!(flow.u, flow.inletBC, flow.outletBC, flow.perdir, t₁)

    # Corrector step - compute forcing from predicted u
    conv_diff!(flow.f, flow.u, flow.σ, λ; ν=flow.ν, Δx=flow.Δx, perdir=flow.perdir)

    # Apply body acceleration
    accelerate!(flow.f, t₁, flow.g, flow.inletBC)

    BDIM!(flow)
    scale_u!(flow, T(0.5))  # Average predictor and corrector
    BC!(flow.u, flow.inletBC, flow.outletBC, flow.perdir, t₁)

    amr_project!(flow, cp, 0.5)
    BC!(flow.u, flow.inletBC, flow.outletBC, flow.perdir, t₁)

    # Update time step - use amr_cfl to account for refined patches
    push!(flow.Δt, amr_cfl(flow, cp))
end

# Helper to scale velocity field (matches Flow.jl scale_u!)
@inline function scale_u!(flow::Flow, scale)
    @loop flow.u[Ii] *= scale over Ii ∈ inside_u(size(flow.p))
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
            patch_div = max(patch_div, abs(div(I, vel_patch.u)))
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
The time step is limited by the finest grid level.
"""
function amr_cfl(flow::Flow{D,T}, cp::CompositePoisson{T}) where {D,T}
    # Base CFL
    dt_base = CFL(flow)

    # Find the maximum refinement ratio across all patches
    # CFL condition: dt ∝ Δx, so finer grids need smaller dt
    max_ratio = one(T)
    for (anchor, patch) in cp.patches
        ratio = T(refinement_ratio(patch))
        max_ratio = max(max_ratio, ratio)
    end

    # Time step is limited by finest grid: dt = dt_base / max_ratio
    return dt_base / max_ratio
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
