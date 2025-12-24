#!/usr/bin/env julia
"""
Swimming Fish Example - Flexible Body with Traveling Wave Motion

This example demonstrates how to model a flexible swimming fish using
BioFlows.jl. The fish body undergoes undulatory motion with a traveling
wave propagating from head to tail, supporting various swimming modes:

- **Carangiform**: Motion concentrated at tail (amplitude_envelope=:carangiform)
- **Anguilliform**: Motion along entire body (amplitude_envelope=:anguilliform)
- **Leading edge oscillation**: Head heave and pitch motion

The body centerline follows:
    y(x,t) = y_head(t) + A(x) * sin(k*x - ω*t + φ) + pitch_contribution(x,t)

where:
- y_head(t) = heave_amplitude * sin(ω*t + heave_phase) - leading edge heave
- A(x) = amplitude envelope (varies based on swimming mode)
- k = 2π/λ = wave number
- ω = 2π*f = angular frequency
- φ = phase offset
- pitch_contribution = s * sin(θ(t)) where θ is the pitch angle

For fish schooling, multiple fish can be created with different phase offsets.
"""

using BioFlows
using StaticArrays: SVector

# =============================================================================
# AMPLITUDE ENVELOPE FUNCTIONS
# =============================================================================
# Different swimming modes have different amplitude distributions along the body

"""
    carangiform_envelope(s, L, A_tail)

Carangiform swimming: amplitude increases quadratically from head to tail.
Typical of tuna, mackerel - stiff body with tail propulsion.
"""
carangiform_envelope(s, L, A_tail) = A_tail * clamp(s/L, 0, 1)^2

"""
    anguilliform_envelope(s, L, A_head, A_tail)

Anguilliform swimming: amplitude increases linearly with non-zero head amplitude.
Typical of eels, lampreys - entire body participates in propulsion.
"""
anguilliform_envelope(s, L, A_head, A_tail) = A_head + (A_tail - A_head) * clamp(s/L, 0, 1)

"""
    subcarangiform_envelope(s, L, A_tail)

Subcarangiform swimming: amplitude increases as s^1.5 - between carangiform and anguilliform.
Typical of trout, carp.
"""
subcarangiform_envelope(s, L, A_tail) = A_tail * clamp(s/L, 0, 1)^1.5

"""
    uniform_envelope(s, L, A_uniform)

Uniform amplitude along the entire body - for anguilliform swimmers with constant amplitude.
"""
uniform_envelope(s, L, A_uniform) = A_uniform

# =============================================================================
# SINGLE FISH SIMULATION
# =============================================================================

"""
    swimming_fish_sim(; nx=256, nz=128, ν=0.001, U=1.0,
                        fish_length=0.2, fish_thickness=0.02,
                        amplitude=0.1, frequency=1.0, wavelength=1.0,
                        center=(0.3, 0.5),
                        heave_amplitude=0.0, heave_phase=0.0,
                        pitch_amplitude=0.0, pitch_phase=π/2,
                        amplitude_envelope=:carangiform,
                        head_amplitude=0.05)

Create a simulation of a swimming fish with flexible body motion.

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

# Leading Edge Motion (sinusoidal motion at the head)
- `heave_amplitude`: Vertical oscillation amplitude at leading edge (relative to length)
- `heave_phase`: Phase of heave motion relative to body wave
- `pitch_amplitude`: Angular oscillation amplitude at leading edge (radians)
- `pitch_phase`: Phase of pitch motion relative to body wave (default π/2 for optimal thrust)

# Swimming Mode
- `amplitude_envelope`: Swimming mode - `:carangiform`, `:anguilliform`, `:subcarangiform`, or `:uniform`
- `head_amplitude`: Amplitude at head for anguilliform mode (relative to length)
"""
function swimming_fish_sim(; nx::Int=256, nz::Int=128,
                             Lx::Real=1.0, Lz::Real=0.5,
                             ν::Real=0.001, U::Real=1.0, ρ::Real=1000.0,
                             fish_length::Real=0.2,
                             fish_thickness::Real=0.02,
                             amplitude::Real=0.1,
                             frequency::Real=1.0,
                             wavelength::Real=1.0,
                             center::Tuple{Real,Real}=(0.3, 0.5),
                             # Leading edge motion parameters
                             heave_amplitude::Real=0.0,
                             heave_phase::Real=0.0,
                             pitch_amplitude::Real=0.0,
                             pitch_phase::Real=π/2,
                             # Swimming mode
                             amplitude_envelope::Symbol=:carangiform,
                             head_amplitude::Real=0.05)

    # Fish parameters
    L = fish_length
    h = fish_thickness / 2  # half-thickness
    A_tail = amplitude * L   # tail amplitude
    A_head = head_amplitude * L  # head amplitude for anguilliform
    k = 2π / (wavelength * L)  # wave number
    ω = 2π * frequency         # angular frequency

    # Leading edge motion amplitudes (in physical units)
    h_heave = heave_amplitude * L  # heave amplitude
    θ_max = pitch_amplitude        # pitch amplitude (radians)

    # Head position in physical coordinates
    x_head = center[1] * Lx
    z_center = center[2] * Lz

    # Select amplitude envelope function based on swimming mode
    function get_amplitude(s)
        if amplitude_envelope == :carangiform
            return carangiform_envelope(s, L, A_tail)
        elseif amplitude_envelope == :anguilliform
            return anguilliform_envelope(s, L, A_head, A_tail)
        elseif amplitude_envelope == :subcarangiform
            return subcarangiform_envelope(s, L, A_tail)
        elseif amplitude_envelope == :uniform
            return uniform_envelope(s, L, A_tail)
        else
            # Default to carangiform
            return carangiform_envelope(s, L, A_tail)
        end
    end

    # Thickness envelope: NACA-like profile (thicker in middle)
    # h(s) = h_max * 4 * (s/L) * (1 - s/L)
    function thickness_envelope(s)
        s_norm = clamp(s / L, 0, 1)
        return h * 4 * s_norm * (1 - s_norm) + 0.001  # Small offset to avoid zero thickness
    end

    """
    Compute body centerline displacement at position s along body at time t.

    Includes:
    1. Leading edge heave: y_heave(t) = h_heave * sin(ω*t + heave_phase)
    2. Leading edge pitch: contributes s * sin(θ(t)) where θ(t) = θ_max * sin(ω*t + pitch_phase)
    3. Traveling wave: A(s) * sin(k*s - ω*t)
    """
    function centerline_displacement(s, t)
        # Leading edge heave motion (sinusoidal vertical oscillation at head)
        y_heave = h_heave * sin(ω * t + heave_phase)

        # Leading edge pitch motion (angular oscillation at head)
        # θ(t) = θ_max * sin(ω*t + pitch_phase)
        # Contribution at distance s from pivot: s * sin(θ(t)) ≈ s * θ(t) for small angles
        θ_t = θ_max * sin(ω * t + pitch_phase)
        y_pitch = s * sin(θ_t)  # Exact formula, not small-angle approximation

        # Traveling wave deformation
        A_local = get_amplitude(s)
        y_wave = A_local * sin(k * s - ω * t)

        return y_heave + y_pitch + y_wave
    end

    """
    SDF for a flexible fish body with traveling wave motion and leading edge oscillation.

    The body centerline is: z_body(x, t) = z_center + centerline_displacement(s, t)
    where s = x - x_head is the distance along the body from the head.
    """
    function fish_sdf(x, t)
        # Position relative to head
        s = x[1] - x_head

        # Leading edge position at current time (for head/tail caps)
        y_head_current = centerline_displacement(0, t)

        # Outside the fish body (before head or after tail)
        if s < 0
            # Distance to head (approximated as semicircle centered on displaced head position)
            z_head = z_center + y_head_current
            return sqrt(s^2 + (x[2] - z_head)^2) - thickness_envelope(0)
        elseif s > L
            # Distance to tail
            z_tail = z_center + centerline_displacement(L, t)
            return sqrt((s - L)^2 + (x[2] - z_tail)^2) - thickness_envelope(L)
        end

        # Body centerline position at this x
        z_body = z_center + centerline_displacement(s, t)

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

# =============================================================================
# FISH SCHOOL SIMULATION
# =============================================================================

"""
    FishConfig

Configuration for a single fish in a school.
"""
struct FishConfig
    x_pos::Float64      # x position as fraction of domain
    z_pos::Float64      # z position as fraction of domain
    phase::Float64      # Phase offset for traveling wave
    heave_phase::Float64  # Phase offset for heave motion
    pitch_phase::Float64  # Phase offset for pitch motion
end

# Constructor with default phases
FishConfig(x, z, phase) = FishConfig(x, z, phase, phase, phase + π/2)

"""
    fish_school_sim(; n_fish=3, spacing=0.15, phase_offset=π/3, formation=:staggered, kwargs...)

Create a simulation of multiple swimming fish (fish school) with flexible body motion.

# Arguments
- `n_fish`: Number of fish in the school
- `spacing`: Spacing between fish (fraction of domain width)
- `phase_offset`: Phase difference between adjacent fish for body wave
- `formation`: School formation - `:staggered`, `:inline`, `:diamond`, or `:custom`
- `custom_positions`: Vector of FishConfig for custom formations

# Leading Edge Motion (applied to all fish)
- `heave_amplitude`: Vertical oscillation amplitude at leading edge (relative to length)
- `heave_phase_offset`: Phase offset increment for heave between adjacent fish
- `pitch_amplitude`: Angular oscillation amplitude at leading edge (radians)
- `pitch_phase_offset`: Phase offset increment for pitch between adjacent fish

# Swimming Mode
- `amplitude_envelope`: Swimming mode - `:carangiform`, `:anguilliform`, `:subcarangiform`, or `:uniform`
- `head_amplitude`: Amplitude at head for anguilliform mode (relative to length)

# Other arguments (same as swimming_fish_sim)
"""
function fish_school_sim(; nx::Int=512, nz::Int=256,
                           Lx::Real=2.0, Lz::Real=1.0,
                           n_fish::Int=3,
                           spacing::Real=0.15,
                           x_spacing::Real=0.2,
                           phase_offset::Real=π/3,
                           formation::Symbol=:staggered,
                           custom_positions::Union{Nothing, Vector{FishConfig}}=nothing,
                           ν::Real=0.001, U::Real=1.0, ρ::Real=1000.0,
                           fish_length::Real=0.15,
                           fish_thickness::Real=0.015,
                           amplitude::Real=0.1,
                           frequency::Real=1.0,
                           wavelength::Real=1.0,
                           # Leading edge motion parameters
                           heave_amplitude::Real=0.0,
                           heave_phase_offset::Real=0.0,
                           pitch_amplitude::Real=0.0,
                           pitch_phase_offset::Real=0.0,
                           # Swimming mode
                           amplitude_envelope::Symbol=:carangiform,
                           head_amplitude::Real=0.05)

    L = fish_length
    h = fish_thickness / 2
    A_tail = amplitude * L
    A_head = head_amplitude * L
    k = 2π / (wavelength * L)
    ω = 2π * frequency

    # Leading edge motion amplitudes
    h_heave = heave_amplitude * L
    θ_max = pitch_amplitude

    # Generate fish positions based on formation
    fish_configs = if !isnothing(custom_positions)
        custom_positions
    else
        configs = FishConfig[]
        for i in 1:n_fish
            if formation == :staggered
                # Staggered formation (diagonal)
                x_pos = 0.2 + (i - 1) * x_spacing
                z_pos = 0.5 + (i - (n_fish + 1) / 2) * spacing
            elseif formation == :inline
                # Inline (tandem) formation
                x_pos = 0.15 + (i - 1) * (L/Lx + 0.1)
                z_pos = 0.5
            elseif formation == :diamond
                # Diamond formation (for 4+ fish)
                if n_fish >= 4
                    # Leader at front, then side-by-side, then follower
                    positions = [
                        (0.15, 0.5),                    # Leader
                        (0.3, 0.5 - spacing),           # Left wing
                        (0.3, 0.5 + spacing),           # Right wing
                        (0.45, 0.5),                    # Tail
                    ]
                    if i <= 4
                        x_pos, z_pos = positions[i]
                    else
                        # Extra fish go behind
                        x_pos = 0.45 + ((i - 4) ÷ 2) * x_spacing
                        z_pos = 0.5 + ((i - 4) % 2 == 0 ? -1 : 1) * spacing
                    end
                else
                    # Fall back to staggered for < 4 fish
                    x_pos = 0.2 + (i - 1) * x_spacing
                    z_pos = 0.5 + (i - (n_fish + 1) / 2) * spacing
                end
            elseif formation == :side_by_side
                # Side-by-side formation
                x_pos = 0.25
                z_pos = 0.5 + (i - (n_fish + 1) / 2) * spacing
            else
                # Default to staggered
                x_pos = 0.2 + (i - 1) * x_spacing
                z_pos = 0.5 + (i - (n_fish + 1) / 2) * spacing
            end

            wave_phase = (i - 1) * phase_offset
            h_phase = (i - 1) * heave_phase_offset
            p_phase = (i - 1) * pitch_phase_offset + π/2

            push!(configs, FishConfig(x_pos, z_pos, wave_phase, h_phase, p_phase))
        end
        configs
    end

    # Amplitude envelope function
    function get_amplitude(s)
        if amplitude_envelope == :carangiform
            return carangiform_envelope(s, L, A_tail)
        elseif amplitude_envelope == :anguilliform
            return anguilliform_envelope(s, L, A_head, A_tail)
        elseif amplitude_envelope == :subcarangiform
            return subcarangiform_envelope(s, L, A_tail)
        elseif amplitude_envelope == :uniform
            return uniform_envelope(s, L, A_tail)
        else
            return carangiform_envelope(s, L, A_tail)
        end
    end

    function thickness_envelope(s)
        s_norm = clamp(s / L, 0, 1)
        return h * 4 * s_norm * (1 - s_norm) + 0.001
    end

    # Centerline displacement for a single fish
    function centerline_displacement(s, t, φ_wave, φ_heave, φ_pitch)
        # Leading edge heave motion
        y_heave = h_heave * sin(ω * t + φ_heave)

        # Leading edge pitch motion
        θ_t = θ_max * sin(ω * t + φ_pitch)
        y_pitch = s * sin(θ_t)

        # Traveling wave deformation
        A_local = get_amplitude(s)
        y_wave = A_local * sin(k * s - ω * t + φ_wave)

        return y_heave + y_pitch + y_wave
    end

    """
    Combined SDF for multiple fish (union of all fish bodies).
    """
    function school_sdf(x, t)
        min_dist = Inf

        for cfg in fish_configs
            x_head = cfg.x_pos * Lx
            z_center = cfg.z_pos * Lz
            s = x[1] - x_head

            # Get head displacement for this fish
            y_head_current = centerline_displacement(0, t, cfg.phase, cfg.heave_phase, cfg.pitch_phase)

            local d
            if s < 0
                z_head = z_center + y_head_current
                d = sqrt(s^2 + (x[2] - z_head)^2) - thickness_envelope(0)
            elseif s > L
                z_tail = z_center + centerline_displacement(L, t, cfg.phase, cfg.heave_phase, cfg.pitch_phase)
                d = sqrt((s - L)^2 + (x[2] - z_tail)^2) - thickness_envelope(L)
            else
                z_body = z_center + centerline_displacement(s, t, cfg.phase, cfg.heave_phase, cfg.pitch_phase)
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

    return sim, fish_configs
end

# =============================================================================
# RUNNER FUNCTIONS
# =============================================================================

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
    sim, fish_configs = fish_school_sim(; kwargs...)
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

    return sim, history, fish_configs
end

# =============================================================================
# CONVENIENCE EXAMPLE FUNCTIONS
# =============================================================================

"""
    anguilliform_fish_sim(; kwargs...)

Create a simulation of an anguilliform (eel-like) swimmer with motion along the entire body.
The amplitude envelope is linear from head to tail with non-zero head amplitude.
"""
function anguilliform_fish_sim(; head_amplitude::Real=0.05, amplitude::Real=0.12, kwargs...)
    swimming_fish_sim(;
        amplitude_envelope=:anguilliform,
        head_amplitude=head_amplitude,
        amplitude=amplitude,
        wavelength=0.8,  # Shorter wavelength typical of anguilliform
        kwargs...
    )
end

"""
    heaving_fish_sim(; heave_amplitude=0.05, kwargs...)

Create a simulation of a fish with sinusoidal heaving (vertical oscillation) at the leading edge.
This models fish that use head oscillation combined with body undulation.
"""
function heaving_fish_sim(; heave_amplitude::Real=0.05, kwargs...)
    swimming_fish_sim(;
        heave_amplitude=heave_amplitude,
        heave_phase=0.0,  # In phase with body wave
        kwargs...
    )
end

"""
    pitching_fish_sim(; pitch_amplitude=0.15, kwargs...)

Create a simulation of a fish with sinusoidal pitching (angular oscillation) at the leading edge.
The pitch is 90° out of phase with the body wave for optimal thrust.
"""
function pitching_fish_sim(; pitch_amplitude::Real=0.15, kwargs...)
    swimming_fish_sim(;
        pitch_amplitude=pitch_amplitude,
        pitch_phase=π/2,  # 90° phase lead for thrust
        kwargs...
    )
end

"""
    combined_motion_fish_sim(; heave_amplitude=0.03, pitch_amplitude=0.1, kwargs...)

Create a simulation of a fish with combined heave + pitch motion at the leading edge
plus body undulation. This is the most realistic model for many fish species.
"""
function combined_motion_fish_sim(; heave_amplitude::Real=0.03, pitch_amplitude::Real=0.1, kwargs...)
    swimming_fish_sim(;
        heave_amplitude=heave_amplitude,
        heave_phase=0.0,
        pitch_amplitude=pitch_amplitude,
        pitch_phase=π/2,
        kwargs...
    )
end

"""
    synchronized_school_sim(; n_fish=3, kwargs...)

Create a fish school where all fish swim with synchronized motion (same phase).
This models tightly coordinated schooling behavior.
"""
function synchronized_school_sim(; n_fish::Int=3, kwargs...)
    fish_school_sim(;
        n_fish=n_fish,
        phase_offset=0.0,  # Same phase
        formation=:side_by_side,
        kwargs...
    )
end

"""
    wave_school_sim(; n_fish=5, phase_offset=π/4, kwargs...)

Create a fish school with wave-like phase progression (like a traveling wave through the school).
"""
function wave_school_sim(; n_fish::Int=5, phase_offset::Real=π/4, kwargs...)
    fish_school_sim(;
        n_fish=n_fish,
        phase_offset=phase_offset,
        formation=:inline,
        kwargs...
    )
end

# =============================================================================
# AMR (ADAPTIVE MESH REFINEMENT) SUPPORT
# =============================================================================

"""
    swimming_fish_amr_sim(; nx=256, nz=128, amr_max_level=2, kwargs...)

Create an AMR-enabled simulation of a swimming fish.
The mesh automatically refines around the fish body and follows its motion.

# AMR Arguments
- `amr_max_level`: Maximum refinement level (1=2x, 2=4x finer)
- `amr_body_threshold`: Distance in cells to refine around body

# Example
```julia
sim = swimming_fish_amr_sim(heave_amplitude=0.05, amplitude_envelope=:anguilliform)
for _ in 1:500
    sim_step!(sim; remeasure=true)  # Patches follow the fish
end
amr_info(sim)  # Print AMR status
```
"""
function swimming_fish_amr_sim(; nx::Int=256, nz::Int=128,
                                  Lx::Real=1.0, Lz::Real=0.5,
                                  ν::Real=0.001, U::Real=1.0, ρ::Real=1000.0,
                                  fish_length::Real=0.2,
                                  fish_thickness::Real=0.02,
                                  amplitude::Real=0.1,
                                  frequency::Real=1.0,
                                  wavelength::Real=1.0,
                                  center::Tuple{Real,Real}=(0.3, 0.5),
                                  heave_amplitude::Real=0.0,
                                  heave_phase::Real=0.0,
                                  pitch_amplitude::Real=0.0,
                                  pitch_phase::Real=π/2,
                                  amplitude_envelope::Symbol=:carangiform,
                                  head_amplitude::Real=0.05,
                                  # AMR options
                                  amr_max_level::Int=2,
                                  amr_body_threshold::Real=4.0)

    # Fish parameters (same as swimming_fish_sim)
    L = fish_length
    h = fish_thickness / 2
    A_tail = amplitude * L
    A_head = head_amplitude * L
    k = 2π / (wavelength * L)
    ω = 2π * frequency
    h_heave = heave_amplitude * L
    θ_max = pitch_amplitude
    x_head = center[1] * Lx
    z_center = center[2] * Lz

    function get_amplitude(s)
        if amplitude_envelope == :carangiform
            return carangiform_envelope(s, L, A_tail)
        elseif amplitude_envelope == :anguilliform
            return anguilliform_envelope(s, L, A_head, A_tail)
        elseif amplitude_envelope == :subcarangiform
            return subcarangiform_envelope(s, L, A_tail)
        elseif amplitude_envelope == :uniform
            return uniform_envelope(s, L, A_tail)
        else
            return carangiform_envelope(s, L, A_tail)
        end
    end

    function thickness_envelope(s)
        s_norm = clamp(s / L, 0, 1)
        return h * 4 * s_norm * (1 - s_norm) + 0.001
    end

    function centerline_displacement(s, t)
        y_heave = h_heave * sin(ω * t + heave_phase)
        θ_t = θ_max * sin(ω * t + pitch_phase)
        y_pitch = s * sin(θ_t)
        A_local = get_amplitude(s)
        y_wave = A_local * sin(k * s - ω * t)
        return y_heave + y_pitch + y_wave
    end

    function fish_sdf(x, t)
        s = x[1] - x_head
        y_head_current = centerline_displacement(0, t)

        if s < 0
            z_head = z_center + y_head_current
            return sqrt(s^2 + (x[2] - z_head)^2) - thickness_envelope(0)
        elseif s > L
            z_tail = z_center + centerline_displacement(L, t)
            return sqrt((s - L)^2 + (x[2] - z_tail)^2) - thickness_envelope(L)
        end

        z_body = z_center + centerline_displacement(s, t)
        h_local = thickness_envelope(s)
        return abs(x[2] - z_body) - h_local
    end

    fish = AutoBody(fish_sdf)

    # Create AMR config optimized for flexible bodies
    amr_config = FlexibleBodyAMRConfig(
        max_level=amr_max_level,
        body_distance_threshold=amr_body_threshold,
        indicator_change_threshold=0.05,  # 5% change triggers regrid
        min_regrid_interval=2
    )

    # Create AMR simulation
    sim = AMRSimulation((nx, nz), (Lx, Lz);
                        inletBC=(U, 0.0),
                        ν=ν,
                        ρ=ρ,
                        body=fish,
                        L_char=fish_length,
                        outletBC=true,
                        amr_config=amr_config)

    return sim
end

"""
    fish_school_amr_sim(; n_fish=3, amr_max_level=2, kwargs...)

Create an AMR-enabled simulation of a fish school.
The mesh automatically refines around all fish bodies and follows their motion.
"""
function fish_school_amr_sim(; nx::Int=512, nz::Int=256,
                                Lx::Real=2.0, Lz::Real=1.0,
                                n_fish::Int=3,
                                spacing::Real=0.15,
                                x_spacing::Real=0.2,
                                phase_offset::Real=π/3,
                                formation::Symbol=:staggered,
                                ν::Real=0.001, U::Real=1.0, ρ::Real=1000.0,
                                fish_length::Real=0.15,
                                fish_thickness::Real=0.015,
                                amplitude::Real=0.1,
                                frequency::Real=1.0,
                                wavelength::Real=1.0,
                                heave_amplitude::Real=0.0,
                                heave_phase_offset::Real=0.0,
                                pitch_amplitude::Real=0.0,
                                pitch_phase_offset::Real=0.0,
                                amplitude_envelope::Symbol=:carangiform,
                                head_amplitude::Real=0.05,
                                # AMR options
                                amr_max_level::Int=2,
                                amr_body_threshold::Real=4.0)

    # Get the fish school body (reuse the SDF construction from fish_school_sim)
    # but create an AMR simulation instead
    L = fish_length
    h = fish_thickness / 2
    A_tail = amplitude * L
    A_head = head_amplitude * L
    k = 2π / (wavelength * L)
    ω = 2π * frequency
    h_heave = heave_amplitude * L
    θ_max = pitch_amplitude

    # Generate fish configurations
    fish_configs = FishConfig[]
    for i in 1:n_fish
        if formation == :staggered
            x_pos = 0.2 + (i - 1) * x_spacing
            z_pos = 0.5 + (i - (n_fish + 1) / 2) * spacing
        elseif formation == :inline
            x_pos = 0.15 + (i - 1) * (L/Lx + 0.1)
            z_pos = 0.5
        elseif formation == :side_by_side
            x_pos = 0.25
            z_pos = 0.5 + (i - (n_fish + 1) / 2) * spacing
        else
            x_pos = 0.2 + (i - 1) * x_spacing
            z_pos = 0.5 + (i - (n_fish + 1) / 2) * spacing
        end
        wave_phase = (i - 1) * phase_offset
        h_phase = (i - 1) * heave_phase_offset
        p_phase = (i - 1) * pitch_phase_offset + π/2
        push!(fish_configs, FishConfig(x_pos, z_pos, wave_phase, h_phase, p_phase))
    end

    function get_amplitude(s)
        if amplitude_envelope == :carangiform
            return carangiform_envelope(s, L, A_tail)
        elseif amplitude_envelope == :anguilliform
            return anguilliform_envelope(s, L, A_head, A_tail)
        elseif amplitude_envelope == :subcarangiform
            return subcarangiform_envelope(s, L, A_tail)
        else
            return carangiform_envelope(s, L, A_tail)
        end
    end

    function thickness_envelope(s)
        s_norm = clamp(s / L, 0, 1)
        return h * 4 * s_norm * (1 - s_norm) + 0.001
    end

    function centerline_displacement(s, t, φ_wave, φ_heave, φ_pitch)
        y_heave = h_heave * sin(ω * t + φ_heave)
        θ_t = θ_max * sin(ω * t + φ_pitch)
        y_pitch = s * sin(θ_t)
        A_local = get_amplitude(s)
        y_wave = A_local * sin(k * s - ω * t + φ_wave)
        return y_heave + y_pitch + y_wave
    end

    function school_sdf(x, t)
        min_dist = Inf
        for cfg in fish_configs
            x_head = cfg.x_pos * Lx
            z_center = cfg.z_pos * Lz
            s = x[1] - x_head
            y_head_current = centerline_displacement(0, t, cfg.phase, cfg.heave_phase, cfg.pitch_phase)

            local d
            if s < 0
                z_head = z_center + y_head_current
                d = sqrt(s^2 + (x[2] - z_head)^2) - thickness_envelope(0)
            elseif s > L
                z_tail = z_center + centerline_displacement(L, t, cfg.phase, cfg.heave_phase, cfg.pitch_phase)
                d = sqrt((s - L)^2 + (x[2] - z_tail)^2) - thickness_envelope(L)
            else
                z_body = z_center + centerline_displacement(s, t, cfg.phase, cfg.heave_phase, cfg.pitch_phase)
                h_local = thickness_envelope(s)
                d = abs(x[2] - z_body) - h_local
            end
            min_dist = min(min_dist, d)
        end
        return min_dist
    end

    school = AutoBody(school_sdf)

    # Create AMR config optimized for flexible bodies
    amr_config = FlexibleBodyAMRConfig(
        max_level=amr_max_level,
        body_distance_threshold=amr_body_threshold,
        indicator_change_threshold=0.03,  # More sensitive for multiple fish
        min_regrid_interval=2,
        buffer_size=3  # Larger buffer for school
    )

    sim = AMRSimulation((nx, nz), (Lx, Lz);
                        inletBC=(U, 0.0),
                        ν=ν,
                        ρ=ρ,
                        body=school,
                        L_char=fish_length,
                        outletBC=true,
                        amr_config=amr_config)

    return sim, fish_configs
end

"""
    run_swimming_fish_amr(; steps=500, kwargs...)

Run a swimming fish simulation with AMR and record force history.
"""
function run_swimming_fish_amr(; steps::Int=500, verbose_amr::Bool=false, kwargs...)
    sim = swimming_fish_amr_sim(; kwargs...)
    history = NamedTuple[]

    for k in 1:steps
        sim_step!(sim; remeasure=true)
        record_force!(history, sim.sim)  # Access underlying simulation for forces

        if k % 100 == 0
            t = sim_time(sim)
            forces = history[end]
            @info "Step $k: t=$(round(t, digits=3)), Cd=$(round(forces.total_coeff[1], digits=4)), Cl=$(round(forces.total_coeff[2], digits=4))"
            if verbose_amr
                stats = get_body_motion_stats(sim)
                @info "  AMR: $(stats.n_refined_cells) refined cells, $(stats.n_patches) patches"
            end
        end
    end

    return sim, history
end

# =============================================================================
# MAIN SCRIPT
# =============================================================================

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 60)
    println("BioFlows.jl - Flexible Swimming Fish Examples")
    println("=" ^ 60)

    # Example 1: Carangiform swimming (default, tail-dominated)
    @info "Example 1: Carangiform swimming fish (tail-dominated motion)..."
    sim1, history1 = run_swimming_fish(steps=200, amplitude_envelope=:carangiform)
    stats1 = summarize_force_history(history1; discard=0.2)
    @info "Carangiform results:" mean_drag=round(stats1.drag_mean, digits=4) mean_lift=round(stats1.lift_mean, digits=4)

    # Example 2: Anguilliform swimming (whole-body motion)
    @info "\nExample 2: Anguilliform swimming fish (whole-body motion)..."
    sim2, history2 = run_swimming_fish(steps=200, amplitude_envelope=:anguilliform, head_amplitude=0.05)
    stats2 = summarize_force_history(history2; discard=0.2)
    @info "Anguilliform results:" mean_drag=round(stats2.drag_mean, digits=4) mean_lift=round(stats2.lift_mean, digits=4)

    # Example 3: Fish with leading edge heave motion
    @info "\nExample 3: Fish with leading edge heave motion..."
    sim3, history3 = run_swimming_fish(steps=200, heave_amplitude=0.05)
    stats3 = summarize_force_history(history3; discard=0.2)
    @info "Heaving fish results:" mean_drag=round(stats3.drag_mean, digits=4) mean_lift=round(stats3.lift_mean, digits=4)

    # Example 4: Fish with leading edge pitch motion
    @info "\nExample 4: Fish with leading edge pitch motion..."
    sim4, history4 = run_swimming_fish(steps=200, pitch_amplitude=0.15)
    stats4 = summarize_force_history(history4; discard=0.2)
    @info "Pitching fish results:" mean_drag=round(stats4.drag_mean, digits=4) mean_lift=round(stats4.lift_mean, digits=4)

    # Example 5: Fish school in staggered formation
    @info "\nExample 5: Fish school (3 fish, staggered formation)..."
    sim5, history5, configs5 = run_fish_school(steps=200, n_fish=3, formation=:staggered)
    stats5 = summarize_force_history(history5; discard=0.2)
    @info "Fish school results:" mean_drag=round(stats5.drag_mean, digits=4) mean_lift=round(stats5.lift_mean, digits=4)

    # Example 6: Fish school with anguilliform swimmers
    @info "\nExample 6: Anguilliform fish school with leading edge motion..."
    sim6, history6, configs6 = run_fish_school(
        steps=200,
        n_fish=3,
        amplitude_envelope=:anguilliform,
        head_amplitude=0.04,
        heave_amplitude=0.02,
        formation=:side_by_side
    )
    stats6 = summarize_force_history(history6; discard=0.2)
    @info "Anguilliform school results:" mean_drag=round(stats6.drag_mean, digits=4) mean_lift=round(stats6.lift_mean, digits=4)

    println("\n" * "=" ^ 60)
    println("All examples completed successfully!")
    println("=" ^ 60)
end
