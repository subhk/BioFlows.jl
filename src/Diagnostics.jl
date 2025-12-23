using StaticArrays

# Internal helper to strip ghost layers from BioFlows arrays.
_strip_ghosts(A, spatial_dims) = begin
    ranges = ntuple(i -> i <= spatial_dims && size(A, i) > 2 ? (2:size(A, i)-1) : (1:size(A, i)), ndims(A))
    copy(view(A, ranges...))
end

"""
    vorticity_component(sim, component; strip_ghosts=true)

Return the requested vorticity component evaluated at cell centres.
For 2D simulations use `component=3` to obtain the out-of-plane vorticity.
"""
function vorticity_component(sim::AbstractSimulation, component::Integer; strip_ghosts::Bool=true)
    ω_field = similar(sim.flow.p)
    @inside ω_field[I] = curl(component, I, sim.flow.u)
    return strip_ghosts ? _strip_ghosts(ω_field, ndims(sim.flow.p)) : ω_field
end

"""
    vorticity_magnitude(sim; strip_ghosts=true)

Compute the vorticity magnitude field for the current simulation state.
"""
function vorticity_magnitude(sim::AbstractSimulation; strip_ghosts::Bool=true)
    ω_field = similar(sim.flow.p)
    @inside ω_field[I] = ω_mag(I, sim.flow.u)
    return strip_ghosts ? _strip_ghosts(ω_field, ndims(sim.flow.p)) : ω_field
end

"""
    cell_center_velocity(sim; strip_ghosts=true)

Return the velocity field averaged to cell centres for each component.
The last dimension indexes the velocity components.
"""
function cell_center_velocity(sim::AbstractSimulation; strip_ghosts::Bool=true)
    spatial_dims = ndims(sim.flow.p)
    vel = zeros(eltype(sim.flow.p), (size(sim.flow.p)..., spatial_dims))
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
    cell_center_vorticity(sim; strip_ghosts=true)

Return the vorticity at cell centres. For 2D simulations this returns a scalar
field (the `ω₃` component); for 3D it returns the full vector with the last
dimension indexing components.
"""
function cell_center_vorticity(sim::AbstractSimulation; strip_ghosts::Bool=true)
    spatial_dims = ndims(sim.flow.p)
    if spatial_dims == 2
        ω_field = zeros(eltype(sim.flow.p), size(sim.flow.p))
        @inside ω_field[I] = curl(3, I, sim.flow.u)
        return strip_ghosts ? _strip_ghosts(ω_field, spatial_dims) : ω_field
    else
        ω_field = zeros(eltype(sim.flow.p), (size(sim.flow.p)..., 3))
        for comp in 1:3
            @loop ω_field[I, comp] = ω(I, sim.flow.u)[comp] over I ∈ inside(sim.flow.p)
        end
        return strip_ghosts ? _strip_ghosts(ω_field, spatial_dims) : ω_field
    end
end

"""
    force_components(sim; ρ=1.0, reference_area=sim.L, with_coefficients=true)

Collect pressure, viscous, and total force vectors for `sim`. When
`with_coefficients=true`, dimensionless coefficients are returned using the
reference area `½ρU²A`.
"""
function force_components(sim::AbstractSimulation; ρ::Real=1.0,
                          reference_area::Real=sim.L, with_coefficients::Bool=true)
    pressure = pressure_force(sim)
    viscous = viscous_force(sim)
    total = pressure .+ viscous
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
    force_coefficients(sim; ρ=1.0, reference_area=sim.L)

Convenience wrapper returning only the total force coefficients.
"""
function force_coefficients(sim::AbstractSimulation; ρ::Real=1.0,
                            reference_area::Real=sim.L)
    components = force_components(sim; ρ, reference_area)
    return components.coefficients === nothing ? nothing : components.coefficients[3]
end

"""
    record_force!(history, sim; ρ=1.0, reference_area=sim.L)

Append a force sample to `history`, which should be a `Vector` of `NamedTuple`s.
Stores time, raw forces, and coefficients.
"""
function record_force!(history::Vector, sim::AbstractSimulation;
                       ρ::Real=1.0, reference_area::Real=sim.L)
    components = force_components(sim; ρ, reference_area)
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
    grid_counts = map(x -> max(x - 2, 1), size(sim.flow.p)[1:spatial_dims])
    spacing = fill(float(sim.L), spatial_dims) ./ grid_counts
    h = minimum(spacing)
    cfl = maximum(max_components) * Δt / h
    return (
        max_u = max_components[1],
        max_w = spatial_dims ≥ 2 ? max_components[2] : 0.0,
        max_v = spatial_dims ≥ 3 ? max_components[3] : 0.0,
        CFL = cfl,
        Δt = Δt,
        length_scale = sim.L,
        grid = size(sim.flow.p)[1:spatial_dims]
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
