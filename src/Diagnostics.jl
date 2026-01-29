# =============================================================================
# FLOW DIAGNOSTICS AND ANALYSIS
# =============================================================================
# This module provides functions to analyze simulation results:
#
# - Field extraction: Convert staggered grid fields to cell-centered format
# - Vorticity: Compute vorticity components and magnitude
# - Forces: Compute and record force coefficients on immersed bodies
# - Statistics: Summarize force history (mean, standard deviation)
#
# All functions work with AbstractSimulation and handle ghost cell stripping
# automatically. Results are returned in physical (non-ghost) grid dimensions.
# =============================================================================

using StaticArrays

# =============================================================================
# GHOST CELL HANDLING
# =============================================================================
# BioFlows arrays include ghost cells for boundary conditions.
# These helpers strip ghost cells to return only physical domain values.
# GPU arrays are explicitly synchronized and transferred to CPU.
# =============================================================================

# Internal helper to strip ghost layers from BioFlows arrays.
# For GPU arrays, this ensures synchronization and returns a CPU Array.
_strip_ghosts(A, spatial_dims) = begin
    ranges = ntuple(i -> i <= spatial_dims && size(A, i) > 2 ? (2:size(A, i)-1) : (1:size(A, i)), ndims(A))
    # Use to_cpu for explicit GPU→CPU transfer with synchronization
    to_cpu(view(A, ranges...))
end

@inline _vorticity2d(I, u) = ∂(2,1,I,u) - ∂(1,2,I,u)
@inline _vorticity2d(I, u, Δx) = ∂(2,1,I,u)/Δx[1] - ∂(1,2,I,u)/Δx[2]

"""
    vorticity_component(sim, component; strip_ghosts=true, physical=true)

Return the requested vorticity component evaluated at cell centres.
For 2D simulations use `component=3` to obtain the out-of-plane vorticity.

# Arguments
- `physical=true`: If true, return physical vorticity in units of 1/s (scaled by Δx).
                   If false, return unit-spacing vorticity (raw differences).
"""
function vorticity_component(sim::AbstractSimulation, component::Integer;
                             strip_ghosts::Bool=true, physical::Bool=true)
    spatial_dims = ndims(sim.flow.p)
    spatial_dims == 2 && component != 3 &&
        throw(ArgumentError("2D vorticity has only component=3 (out-of-plane)."))
    ω_field = similar(sim.flow.p)
    fill!(ω_field, zero(eltype(ω_field)))
    if spatial_dims == 2
        if physical
            Δx = sim.flow.Δx
            @inside ω_field[I] = _vorticity2d(I, sim.flow.u, Δx)
        else
            @inside ω_field[I] = _vorticity2d(I, sim.flow.u)
        end
    elseif physical
        Δx = sim.flow.Δx
        @inside ω_field[I] = ω(I, sim.flow.u, Δx)[component]
    else
        @inside ω_field[I] = ω(I, sim.flow.u)[component]
    end
    return strip_ghosts ? _strip_ghosts(ω_field, spatial_dims) : ω_field
end

"""
    vorticity_magnitude(sim; strip_ghosts=true, physical=true)

Compute the vorticity magnitude field for the current simulation state.

# Arguments
- `physical=true`: If true, return physical vorticity magnitude in units of 1/s (scaled by Δx).
                   If false, return unit-spacing vorticity magnitude (raw differences).
"""
function vorticity_magnitude(sim::AbstractSimulation; strip_ghosts::Bool=true, physical::Bool=true)
    spatial_dims = ndims(sim.flow.p)
    ω_field = similar(sim.flow.p)
    fill!(ω_field, zero(eltype(ω_field)))
    if spatial_dims == 2
        if physical
            Δx = sim.flow.Δx
            @inside ω_field[I] = abs(_vorticity2d(I, sim.flow.u, Δx))
        else
            @inside ω_field[I] = abs(_vorticity2d(I, sim.flow.u))
        end
    else
        if physical
            Δx = sim.flow.Δx
            @inside ω_field[I] = ω_mag(I, sim.flow.u, Δx)
        else
            @inside ω_field[I] = ω_mag(I, sim.flow.u)
        end
    end
    return strip_ghosts ? _strip_ghosts(ω_field, spatial_dims) : ω_field
end

"""
    cell_center_velocity(sim; strip_ghosts=true)

Return the velocity field averaged to cell centres for each component.
The last dimension indexes the velocity components.
"""
function cell_center_velocity(sim::AbstractSimulation; strip_ghosts::Bool=true)
    spatial_dims = ndims(sim.flow.p)
    vel = similar(sim.flow.u, (size(sim.flow.p)..., spatial_dims))
    fill!(vel, zero(eltype(vel)))
    for comp in 1:spatial_dims
        @loop vel[I, comp] = 0.5 * (sim.flow.u[I, comp] + sim.flow.u[I + δ(comp, I), comp]) over I ∈ inside(sim.flow.p)
    end
    return strip_ghosts ? _strip_ghosts(vel, spatial_dims) : vel
end

"""
    cell_center_pressure(sim; strip_ghosts=true)

Return the pressure field at cell centres. Pressure is already cell-centred
in the staggered grid, so this simply returns the field with optional ghost
cell stripping.
"""
function cell_center_pressure(sim::AbstractSimulation; strip_ghosts::Bool=true)
    spatial_dims = ndims(sim.flow.p)
    return strip_ghosts ? _strip_ghosts(sim.flow.p, spatial_dims) : copy(sim.flow.p)
end

"""
    cell_center_vorticity(sim; strip_ghosts=true, physical=true)

Return the vorticity at cell centres. For 2D simulations this returns a scalar
field (the `ω₃` component); for 3D it returns the full vector with the last
dimension indexing components.

# Arguments
- `physical=true`: If true, return physical vorticity in units of 1/s (scaled by Δx).
                   If false, return unit-spacing vorticity (raw differences).
"""
function cell_center_vorticity(sim::AbstractSimulation; strip_ghosts::Bool=true, physical::Bool=true)
    spatial_dims = ndims(sim.flow.p)
    Δx = sim.flow.Δx
    if spatial_dims == 2
        ω_field = similar(sim.flow.p)
        fill!(ω_field, zero(eltype(ω_field)))
        if physical
            @inside ω_field[I] = _vorticity2d(I, sim.flow.u, Δx)
        else
            @inside ω_field[I] = _vorticity2d(I, sim.flow.u)
        end
        return strip_ghosts ? _strip_ghosts(ω_field, spatial_dims) : ω_field
    else
        ω_field = similar(sim.flow.p, (size(sim.flow.p)..., 3))
        fill!(ω_field, zero(eltype(ω_field)))
        if physical
            for comp in 1:3
                @loop ω_field[I, comp] = ω(I, sim.flow.u, Δx)[comp] over I ∈ inside(sim.flow.p)
            end
        else
            for comp in 1:3
                @loop ω_field[I, comp] = ω(I, sim.flow.u)[comp] over I ∈ inside(sim.flow.p)
            end
        end
        return strip_ghosts ? _strip_ghosts(ω_field, spatial_dims) : ω_field
    end
end

"""
    force_components(sim; reference_area=sim.L, with_coefficients=true)

Collect pressure, viscous, and total force vectors for `sim` in Newtons.
When `with_coefficients=true`, dimensionless coefficients are returned using
the reference area `½ρU²A`. Uses the density from the simulation (sim.flow.ρ).
"""
function force_components(sim::AbstractSimulation;
                          reference_area::Real=sim.L, with_coefficients::Bool=true)
    pressure = pressure_force(sim)
    viscous = viscous_force(sim)
    total = pressure .+ viscous
    ρ = sim.flow.ρ  # Use density from simulation
    coeff_scale = with_coefficients ? (0.5 * ρ * sim.U^2 * reference_area) : nothing
    coefficients = isnothing(coeff_scale) ? nothing : (
        pressure ./ coeff_scale,
        viscous ./ coeff_scale,
        total ./ coeff_scale
    )
    return (
        pressure = pressure,
        viscous = viscous,
        total = total,
        coefficients = coefficients
    )
end

"""
    force_coefficients(sim; reference_area=sim.L)

Convenience wrapper returning only the total force coefficients.
Uses the density from the simulation (sim.flow.ρ).
"""
function force_coefficients(sim::AbstractSimulation; reference_area::Real=sim.L)
    components = force_components(sim; reference_area)
    return components.coefficients === nothing ? nothing : components.coefficients[3]
end

"""
    record_force!(history, sim; reference_area=sim.L)

Append a force sample to `history`, which should be a `Vector` of `NamedTuple`s.
Stores time, raw forces (in Newtons), and dimensionless coefficients.
Uses the density from the simulation (sim.flow.ρ).
"""
function record_force!(history::Vector, sim::AbstractSimulation;
                       reference_area::Real=sim.L)
    components = force_components(sim; reference_area)
    push!(history, (
        time = sim_time(sim),
        pressure = components.pressure,
        viscous = components.viscous,
        total = components.total,
        pressure_coeff = components.coefficients === nothing ? nothing : components.coefficients[1],
        viscous_coeff = components.coefficients === nothing ? nothing : components.coefficients[2],
        total_coeff = components.coefficients === nothing ? nothing : components.coefficients[3]
    ))
    return history
end

"""
    compute_diagnostics(sim)

Return a named tuple containing simple diagnostic quantities for the current
state of `sim`, including maximum velocity components and an estimated CFL
number.
"""
function compute_diagnostics(sim::AbstractSimulation)
    vel = sim.flow.u
    spatial_dims = ndims(sim.flow.p)
    colon_dims = ntuple(_ -> Colon(), spatial_dims)
    max_components = Vector{eltype(sim.flow.p)}(undef, spatial_dims)
    for comp in 1:spatial_dims
        slice = view(vel, colon_dims..., comp)
        max_components[comp] = maximum(abs, slice)
    end
    Δt = sim.flow.Δt[end]
    h = minimum(sim.flow.Δx)
    cfl = maximum(max_components) * Δt / h
    return (
        max_u = max_components[1],
        max_w = spatial_dims ≥ 2 ? max_components[2] : 0.0,
        max_v = spatial_dims ≥ 3 ? max_components[3] : 0.0,
        CFL = cfl,
        Δt = Δt,
        length_scale = sim.L,
        grid = map(x -> max(x - 2, 0), size(sim.flow.p)[1:spatial_dims])
    )
end

"""
    summarize_force_history(history; discard=0)

Compute summary statistics from a force history vector. Optionally discard
the first `discard` fraction of samples (0.0 to 1.0).
"""
function summarize_force_history(history::Vector; discard::Real=0.0)
    n = length(history)
    start_idx = max(1, round(Int, discard * n) + 1)
    subset = history[start_idx:end]

    if isempty(subset)
        return (drag_mean=NaN, drag_std=NaN, lift_mean=NaN, lift_std=NaN)
    end

    # Extract force components (assuming 2D: [drag, lift] or 3D: [drag, lift, side])
    totals = [s.total for s in subset]
    drag = [t[1] for t in totals]
    lift = length(totals[1]) >= 2 ? [t[2] for t in totals] : zeros(length(totals))

    return (
        drag_mean = mean(drag),
        drag_std = std(drag),
        lift_mean = mean(lift),
        lift_std = std(lift)
    )
end
