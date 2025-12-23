#!/usr/bin/env julia
"""
Swimming Fish Example - Flexible Body with Traveling Wave Motion

This example demonstrates how to model a flexible swimming fish using
BioFlows.jl. The fish body undergoes undulatory motion with a traveling
wave propagating from head to tail, mimicking carangiform/anguilliform swimming.

The body centerline follows:
    y(x,t) = A(x) * sin(k*x - ω*t + φ)

where:
- A(x) = amplitude envelope (increases from head to tail)
- k = 2π/λ = wave number
- ω = 2π*f = angular frequency
- φ = phase offset

For fish schooling, multiple fish can be created with different phase offsets.
"""

using BioFlows
using StaticArrays: SVector

"""
    swimming_fish_sim(; nx=256, nz=128, ν=0.001, U=1.0,
                        fish_length=0.2, fish_thickness=0.02,
                        amplitude=0.1, frequency=1.0, wavelength=1.0,
                        center=(0.3, 0.5))

Create a simulation of a swimming fish with traveling wave body motion.

# Arguments
- `nx, nz`: Grid dimensions
- `ν`: Kinematic viscosity (m²/s)
- `U`: Inflow velocity (m/s)
- `fish_length`: Fish body length (m)
- `fish_thickness`: Maximum fish body thickness (m)
- `amplitude`: Tail amplitude relative to body length
- `frequency`: Oscillation frequency (Hz)
- `wavelength`: Wavelength relative to body length
- `center`: Fish head position as fraction of domain (x, z)
"""
function swimming_fish_sim(; nx::Int=256, nz::Int=128,
                             Lx::Real=1.0, Lz::Real=0.5,
                             ν::Real=0.001, U::Real=1.0, ρ::Real=1000.0,
                             fish_length::Real=0.2,
                             fish_thickness::Real=0.02,
                             amplitude::Real=0.1,
                             frequency::Real=1.0,
                             wavelength::Real=1.0,
                             center::Tuple{Real,Real}=(0.3, 0.5))

    # Fish parameters
    L = fish_length
    h = fish_thickness / 2  # half-thickness
    A_tail = amplitude * L   # tail amplitude
    k = 2π / (wavelength * L)  # wave number
    ω = 2π * frequency         # angular frequency

    # Head position in physical coordinates
    x_head = center[1] * Lx
    z_center = center[2] * Lz

    # Amplitude envelope: zero at head, max at tail
    # Using quadratic envelope: A(s) = A_tail * (s/L)^2
    function amplitude_envelope(s)
        s_norm = clamp(s / L, 0, 1)
        return A_tail * s_norm^2
    end

    # Thickness envelope: NACA-like profile (thicker in middle)
    # h(s) = h_max * 4 * (s/L) * (1 - s/L)
    function thickness_envelope(s)
        s_norm = clamp(s / L, 0, 1)
        return h * 4 * s_norm * (1 - s_norm) + 0.001  # Small offset to avoid zero thickness
    end

    """
    SDF for a flexible fish body with traveling wave motion.

    The body centerline is: z_body(x, t) = z_center + A(s) * sin(k*s - ω*t)
    where s = x - x_head is the distance along the body from the head.
    """
    function fish_sdf(x, t)
        # Position relative to head
        s = x[1] - x_head

        # Outside the fish body (before head or after tail)
        if s < 0
            # Distance to head (approximated as semicircle)
            return sqrt(s^2 + (x[2] - z_center)^2) - thickness_envelope(0)
        elseif s > L
            # Distance to tail
            z_tail = z_center + amplitude_envelope(L) * sin(k * L - ω * t)
            return sqrt((s - L)^2 + (x[2] - z_tail)^2) - thickness_envelope(L)
        end

        # Body centerline position at this x
        A_local = amplitude_envelope(s)
        z_body = z_center + A_local * sin(k * s - ω * t)

        # Local body thickness
        h_local = thickness_envelope(s)

        # Distance to body surface (approximate as distance to centerline minus thickness)
        return abs(x[2] - z_body) - h_local
    end

    # Create the fish body
    fish = AutoBody(fish_sdf)

    # Create simulation
    sim = Simulation((nx, nz), (Lx, Lz);
                     inletBC=(U, 0.0),
                     ν=ν,
                     ρ=ρ,
                     body=fish,
                     L_char=fish_length,
                     outletBC=true)

    return sim
end

"""
    fish_school_sim(; n_fish=3, spacing=0.15, phase_offset=π/3, kwargs...)

Create a simulation of multiple swimming fish (fish school).

# Arguments
- `n_fish`: Number of fish in the school
- `spacing`: Spacing between fish (fraction of domain width)
- `phase_offset`: Phase difference between adjacent fish
- `kwargs`: Additional arguments passed to each fish
"""
function fish_school_sim(; nx::Int=512, nz::Int=256,
                           Lx::Real=2.0, Lz::Real=1.0,
                           n_fish::Int=3,
                           spacing::Real=0.15,
                           phase_offset::Real=π/3,
                           ν::Real=0.001, U::Real=1.0, ρ::Real=1000.0,
                           fish_length::Real=0.15,
                           fish_thickness::Real=0.015,
                           amplitude::Real=0.1,
                           frequency::Real=1.0,
                           wavelength::Real=1.0)

    L = fish_length
    h = fish_thickness / 2
    A_tail = amplitude * L
    k = 2π / (wavelength * L)
    ω = 2π * frequency

    # Fish positions (staggered formation)
    fish_positions = []
    for i in 1:n_fish
        x_pos = 0.2 + (i - 1) * 0.2  # Staggered in x
        z_pos = 0.5 + (i - (n_fish + 1) / 2) * spacing  # Spread in z
        phase = (i - 1) * phase_offset
        push!(fish_positions, (x_pos, z_pos, phase))
    end

    function amplitude_envelope(s)
        s_norm = clamp(s / L, 0, 1)
        return A_tail * s_norm^2
    end

    function thickness_envelope(s)
        s_norm = clamp(s / L, 0, 1)
        return h * 4 * s_norm * (1 - s_norm) + 0.001
    end

    """
    Combined SDF for multiple fish (union of all fish bodies).
    """
    function school_sdf(x, t)
        min_dist = Inf

        for (x_frac, z_frac, φ) in fish_positions
            x_head = x_frac * Lx
            z_center = z_frac * Lz
            s = x[1] - x_head

            local d
            if s < 0
                d = sqrt(s^2 + (x[2] - z_center)^2) - thickness_envelope(0)
            elseif s > L
                z_tail = z_center + amplitude_envelope(L) * sin(k * L - ω * t + φ)
                d = sqrt((s - L)^2 + (x[2] - z_tail)^2) - thickness_envelope(L)
            else
                A_local = amplitude_envelope(s)
                z_body = z_center + A_local * sin(k * s - ω * t + φ)
                h_local = thickness_envelope(s)
                d = abs(x[2] - z_body) - h_local
            end

            min_dist = min(min_dist, d)
        end

        return min_dist
    end

    school = AutoBody(school_sdf)

    sim = Simulation((nx, nz), (Lx, Lz);
                     inletBC=(U, 0.0),
                     ν=ν,
                     ρ=ρ,
                     body=school,
                     L_char=fish_length,
                     outletBC=true)

    return sim
end

"""
    run_swimming_fish(; steps=500, kwargs...)

Run the swimming fish simulation and record force history.
"""
function run_swimming_fish(; steps::Int=500, kwargs...)
    sim = swimming_fish_sim(; kwargs...)
    history = NamedTuple[]

    for k in 1:steps
        sim_step!(sim; remeasure=true)  # remeasure=true for moving body
        record_force!(history, sim)

        if k % 100 == 0
            t = sim_time(sim)
            forces = history[end]
            @info "Step $k: t=$(round(t, digits=3)), Cd=$(round(forces.total_coeff[1], digits=4)), Cl=$(round(forces.total_coeff[2], digits=4))"
        end
    end

    return sim, history
end

"""
    run_fish_school(; steps=500, kwargs...)

Run the fish school simulation and record force history.
"""
function run_fish_school(; steps::Int=500, kwargs...)
    sim = fish_school_sim(; kwargs...)
    history = NamedTuple[]

    for k in 1:steps
        sim_step!(sim; remeasure=true)
        record_force!(history, sim)

        if k % 100 == 0
            t = sim_time(sim)
            forces = history[end]
            @info "Step $k: t=$(round(t, digits=3)), Cd=$(round(forces.total_coeff[1], digits=4)), Cl=$(round(forces.total_coeff[2], digits=4))"
        end
    end

    return sim, history
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    @info "Running single swimming fish simulation..."
    sim, history = run_swimming_fish(steps=200)

    stats = summarize_force_history(history; discard=0.2)
    @info "Swimming fish results:" mean_drag=stats.drag_mean mean_lift=stats.lift_mean

    @info "\nRunning fish school simulation (3 fish)..."
    sim_school, history_school = run_fish_school(steps=200, n_fish=3)

    stats_school = summarize_force_history(history_school; discard=0.2)
    @info "Fish school results:" mean_drag=stats_school.drag_mean mean_lift=stats_school.lift_mean
end
