# =============================================================================
# ADAPTIVE MESH REFINEMENT FOR FLEXIBLE BODIES (EULER-BERNOULLI BEAM)
# =============================================================================
# This module provides AMR support for flexible bodies that deform during
# simulation. It creates refinement indicators based on beam position and
# integrates with the existing BioFlows AMR infrastructure.
#
# Key features:
# - Beam-based refinement indicator following the deformed shape
# - Automatic regridding when beam moves significantly
# - Integration with EulerBernoulliBeam solver
# =============================================================================

"""
    FlexibleBodySDF{T}

A time-dependent signed distance function that follows a deforming beam.
The beam centerline position is updated from the EulerBernoulliBeam state.

# Fields
- `beam`: Reference to the EulerBernoulliBeam
- `x_head`: x-coordinate of beam head (clamped end)
- `z_center`: z-coordinate of undeformed centerline
- `thickness_func`: Function h(s) returning local half-thickness
- `width`: Beam width (for 3D, or out-of-plane thickness for 2D)
"""
mutable struct FlexibleBodySDF{T<:AbstractFloat, B<:EulerBernoulliBeam}
    beam::B
    x_head::T
    z_center::T
    thickness_func::Function
    width::T

    # Cache for interpolation
    s_coords::Vector{T}
    w_current::Vector{T}

    function FlexibleBodySDF(beam::EulerBernoulliBeam{T}, x_head::Real, z_center::Real;
                             thickness_func::Function=s->beam.geometry.thickness(s),
                             width::Real=0.01) where T
        s_coords = collect(beam.s)
        w_current = zeros(T, beam.n_nodes)
        new{T, typeof(beam)}(beam, T(x_head), T(z_center), thickness_func, T(width),
                              s_coords, w_current)
    end
end

"""
    update!(sdf::FlexibleBodySDF)

Update the cached beam displacement from the current beam state.
Call this after each beam time step.
"""
function update!(sdf::FlexibleBodySDF)
    copyto!(sdf.w_current, sdf.beam.w)
    return sdf
end

@inline function _beam_sdf_with_displacement(sdf::FlexibleBodySDF{T},
                                             x,
                                             w_vals::AbstractVector) where T
    # x[1] is streamwise, x[2] is transverse (z)
    x_rel = x[1] - sdf.x_head
    z_pos = x[2]
    L = sdf.beam.geometry.L

    # Check if outside beam extent
    if x_rel < 0
        # Before beam head - distance to leading edge
        h_head = sdf.thickness_func(zero(T))
        z_beam = sdf.z_center + T(w_vals[1])
        dz = z_pos - z_beam
        return sqrt(x_rel^2 + max(zero(T), abs(dz) - h_head)^2)
    elseif x_rel > L
        # After beam tail - distance to trailing edge
        h_tail = sdf.thickness_func(L)
        z_beam = sdf.z_center + T(w_vals[end])
        dz = z_pos - z_beam
        return sqrt((x_rel - L)^2 + max(zero(T), abs(dz) - h_tail)^2)
    end

    # Inside beam extent - interpolate displacement
    s = x_rel
    n = length(w_vals)
    ds = L / (n - 1)
    idx = clamp(floor(Int, s / ds) + 1, 1, n - 1)

    # Linear interpolation of displacement
    s1, s2 = sdf.s_coords[idx], sdf.s_coords[idx + 1]
    w1, w2 = T(w_vals[idx]), T(w_vals[idx + 1])
    α = (s - s1) / (s2 - s1)
    w_interp = (one(T) - α) * w1 + α * w2

    # Beam centerline position
    z_beam = sdf.z_center + w_interp

    # Local half-thickness
    h = sdf.thickness_func(s)

    # Signed distance (negative inside)
    dz = z_pos - z_beam
    return abs(dz) - h
end

"""
    (sdf::FlexibleBodySDF)(x, t)

Compute signed distance from point x to the deformed beam surface.
Negative inside, positive outside.
"""
function (sdf::FlexibleBodySDF{T})(x, t) where T
    return _beam_sdf_with_displacement(sdf, x, sdf.w_current)
end

"""
    compute_beam_refinement_indicator(flow::Flow, beam_sdf::FlexibleBodySDF;
                                       distance_threshold=3.0)

Compute refinement indicator based on proximity to the deformed beam.
Returns array with 1.0 where refinement is needed (near beam), 0.0 elsewhere.

# Arguments
- `flow`: The BioFlows Flow struct
- `beam_sdf`: FlexibleBodySDF representing the deformed beam
- `distance_threshold`: Distance in grid cells within which to refine

# Returns
- Array of same size as `flow.p` with indicator values in [0, 1]
"""
function compute_beam_refinement_indicator(flow::Flow{N,T}, beam_sdf::FlexibleBodySDF;
                                           distance_threshold::Real=3.0) where {N,T}
    indicator = similar(flow.p)
    fill!(indicator, zero(T))
    threshold = T(distance_threshold)

    # Update beam SDF with current state
    update!(beam_sdf)

    # Compute indicator
    for I in inside(flow.p)
        x = loc(0, I, T)
        d = beam_sdf(x, zero(T))
        indicator[I] = abs(d) < threshold ? one(T) : zero(T)
    end

    return indicator
end

"""
    compute_beam_motion_indicator(flow::Flow, beam_sdf::FlexibleBodySDF,
                                  last_displacement::Vector;
                                  motion_threshold=0.5)

Compute refinement indicator based on beam motion since last check.
Marks regions where the beam has moved significantly.

# Arguments
- `flow`: The BioFlows Flow struct
- `beam_sdf`: FlexibleBodySDF representing the deformed beam
- `last_displacement`: Previous displacement field w(s)
- `motion_threshold`: Minimum displacement change to trigger refinement

# Returns
- Array with 1.0 where beam has moved, 0.0 elsewhere
"""
function compute_beam_motion_indicator(flow::Flow{N,T}, beam_sdf::FlexibleBodySDF,
                                       last_displacement::Vector;
                                       motion_threshold::Real=0.5,
                                       distance_threshold::Real=5.0) where {N,T}
    indicator = similar(flow.p)
    fill!(indicator, zero(T))
    threshold = T(distance_threshold)
    motion_thresh = T(motion_threshold)

    # Update beam SDF
    update!(beam_sdf)

    # Compute maximum displacement change
    Δw_max = maximum(abs.(beam_sdf.w_current .- last_displacement))

    if Δw_max < motion_thresh
        return indicator  # No significant motion
    end

    # Mark region around both old and new positions
    for I in inside(flow.p)
        x = loc(0, I, T)

        # Distance to current and previous beam positions
        d_new = abs(_beam_sdf_with_displacement(beam_sdf, x, beam_sdf.w_current))
        d_old = abs(_beam_sdf_with_displacement(beam_sdf, x, last_displacement))

        if d_new < threshold || d_old < threshold
            indicator[I] = one(T)
        end
    end

    return indicator
end

"""
    BeamAMRTracker{T}

Tracks beam state for AMR regridding decisions.
"""
mutable struct BeamAMRTracker{T<:AbstractFloat}
    beam_sdf::FlexibleBodySDF{T}
    last_displacement::Vector{T}
    last_regrid_step::Int
    motion_history::Vector{T}  # Track motion magnitude

    function BeamAMRTracker(beam_sdf::FlexibleBodySDF{T}) where T
        n = beam_sdf.beam.n_nodes
        new{T}(beam_sdf, zeros(T, n), 0, T[])
    end
end

"""
    should_regrid(tracker::BeamAMRTracker, step::Int;
                  min_interval=5, motion_threshold=0.5)

Determine if AMR regridding is needed based on beam motion.

# Returns
- `true` if regridding should occur, `false` otherwise
"""
function should_regrid(tracker::BeamAMRTracker{T}, step::Int;
                       min_interval::Int=5,
                       motion_threshold::Real=0.5) where T
    # Enforce minimum interval
    if step - tracker.last_regrid_step < min_interval
        return false
    end

    # Update beam SDF
    update!(tracker.beam_sdf)

    # Compute motion since last regrid
    Δw = tracker.beam_sdf.w_current .- tracker.last_displacement
    motion = maximum(abs.(Δw))

    push!(tracker.motion_history, motion)

    return motion > T(motion_threshold)
end

"""
    mark_regrid!(tracker::BeamAMRTracker, step::Int)

Mark that regridding occurred at this step and update state.
"""
function mark_regrid!(tracker::BeamAMRTracker, step::Int)
    tracker.last_regrid_step = step
    copyto!(tracker.last_displacement, tracker.beam_sdf.w_current)
    return tracker
end

# =============================================================================
# COMBINED INDICATOR FOR BEAM + FLOW FEATURES
# =============================================================================

"""
    compute_beam_combined_indicator(flow::Flow, beam_sdf::FlexibleBodySDF;
                                     beam_threshold=3.0,
                                     gradient_threshold=1.0,
                                     vorticity_threshold=1.0,
                                     beam_weight=0.6,
                                     gradient_weight=0.25,
                                     vorticity_weight=0.15)

Compute combined refinement indicator for flexible body simulation.
Combines beam proximity with flow gradient and vorticity indicators.

Higher `beam_weight` ensures refinement follows the deforming body.
"""
function compute_beam_combined_indicator(flow::Flow{N,T}, beam_sdf::FlexibleBodySDF;
                                         beam_threshold::Real=3.0,
                                         gradient_threshold::Real=1.0,
                                         vorticity_threshold::Real=1.0,
                                         beam_weight::Real=0.6,
                                         gradient_weight::Real=0.25,
                                         vorticity_weight::Real=0.15) where {N,T}
    # Normalize weights
    total_weight = beam_weight + gradient_weight + vorticity_weight
    bw = T(beam_weight / total_weight)
    gw = T(gradient_weight / total_weight)
    vw = T(vorticity_weight / total_weight)

    # Compute individual indicators
    beam_ind = compute_beam_refinement_indicator(flow, beam_sdf;
                                                  distance_threshold=beam_threshold)
    gradient_ind = compute_velocity_gradient_indicator(flow)
    vorticity_ind = compute_vorticity_indicator(flow)

    # Threshold-based activation
    gt = T(gradient_threshold)
    vt = T(vorticity_threshold)

    # Combine with weights
    combined = similar(flow.p)
    for I in inside(flow.p)
        g_active = gradient_ind[I] > gt ? one(T) : zero(T)
        v_active = vorticity_ind[I] > vt ? one(T) : zero(T)
        combined[I] = bw * beam_ind[I] + gw * g_active + vw * v_active
    end

    # Set boundary values to zero
    if N == 2
        fill!(view(combined, 1, :), zero(T))
        fill!(view(combined, size(combined, 1), :), zero(T))
        fill!(view(combined, :, 1), zero(T))
        fill!(view(combined, :, size(combined, 2)), zero(T))
    else  # N == 3
        fill!(view(combined, 1, :, :), zero(T))
        fill!(view(combined, size(combined, 1), :, :), zero(T))
        fill!(view(combined, :, 1, :), zero(T))
        fill!(view(combined, :, size(combined, 2), :), zero(T))
        fill!(view(combined, :, :, 1), zero(T))
        fill!(view(combined, :, :, size(combined, 3)), zero(T))
    end

    return combined
end

# =============================================================================
# AMR SIMULATION WRAPPER FOR FLEXIBLE BODIES
# =============================================================================

"""
    BeamAMRConfig

Configuration for AMR with Euler-Bernoulli beam flexible bodies.
"""
Base.@kwdef struct BeamAMRConfig
    max_level::Int = 2
    beam_distance_threshold::Float64 = 3.0
    gradient_threshold::Float64 = 1.0
    vorticity_threshold::Float64 = 1.0
    beam_weight::Float64 = 0.6
    gradient_weight::Float64 = 0.25
    vorticity_weight::Float64 = 0.15
    buffer_size::Int = 2
    min_regrid_interval::Int = 5
    motion_threshold::Float64 = 0.5
    regrid_interval::Int = 10  # Force regrid every N steps regardless of motion
end

"""
    regrid_for_beam!(amr_sim, beam_sdf::FlexibleBodySDF, tracker::BeamAMRTracker,
                     step::Int, config::BeamAMRConfig)

Perform AMR regridding based on beam position and motion.
"""
function regrid_for_beam!(amr_sim, beam_sdf::FlexibleBodySDF,
                          tracker::BeamAMRTracker, step::Int,
                          config::BeamAMRConfig)
    flow = amr_sim.sim.flow

    # Check if regridding is needed
    needs_regrid = should_regrid(tracker, step;
                                  min_interval=config.min_regrid_interval,
                                  motion_threshold=config.motion_threshold)

    # Also force regrid at intervals
    if step % config.regrid_interval == 0
        needs_regrid = true
    end

    if !needs_regrid
        return false
    end

    # Compute combined indicator
    indicator = compute_beam_combined_indicator(flow, beam_sdf;
        beam_threshold=config.beam_distance_threshold,
        gradient_threshold=config.gradient_threshold,
        vorticity_threshold=config.vorticity_threshold,
        beam_weight=config.beam_weight,
        gradient_weight=config.gradient_weight,
        vorticity_weight=config.vorticity_weight)

    # Apply buffer zone
    apply_buffer_zone!(indicator; buffer_size=config.buffer_size)

    # Mark cells for refinement
    cells_to_refine = mark_cells_for_refinement(indicator; threshold=0.5)

    # Update refined grid
    update_refined_cells!(amr_sim.refined_grid, cells_to_refine, config.max_level)

    # Create patches
    create_patches!(amr_sim.composite_pois, amr_sim.refined_grid, flow.μ₀)

    # Synchronize data
    synchronize_base_and_patches!(flow, amr_sim.composite_pois)

    # Mark that regrid occurred
    mark_regrid!(tracker, step)

    return true
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
    create_beam_body(beam::EulerBernoulliBeam, x_head, z_center;
                     thickness_func=nothing, width=0.01)

Create an AutoBody from an EulerBernoulliBeam for use with standard simulation.
The body's SDF automatically updates as the beam deforms.

# Returns
- `(body, beam_sdf)` tuple - the AutoBody and its underlying FlexibleBodySDF
"""
function create_beam_body(beam::EulerBernoulliBeam{T}, x_head::Real, z_center::Real;
                          thickness_func::Union{Nothing,Function}=nothing,
                          width::Real=0.01) where T
    # Default thickness function from geometry
    h_func = thickness_func === nothing ? s -> beam.geometry.thickness(s) : thickness_func

    beam_sdf = FlexibleBodySDF(beam, x_head, z_center;
                                thickness_func=h_func, width=width)

    # Create AutoBody with time-dependent SDF
    x_head_T = T(x_head)
    velocity_func = (x, t) -> SVector{2,T}(zero(T), interpolate_velocity(beam, x[1] - x_head_T))
    body = AutoBody((x, t) -> beam_sdf(x, t); velocity=velocity_func)

    return body, beam_sdf
end

"""
    get_beam_bounding_box(beam_sdf::FlexibleBodySDF; margin=0.0)

Get the bounding box of the current beam configuration.

# Returns
- `(x_min, x_max, z_min, z_max)` tuple
"""
function get_beam_bounding_box(beam_sdf::FlexibleBodySDF{T}; margin::Real=0.0) where T
    update!(beam_sdf)

    L = beam_sdf.beam.geometry.L
    x_min = beam_sdf.x_head - T(margin)
    x_max = beam_sdf.x_head + L + T(margin)

    # Find z extent
    w_min, w_max = extrema(beam_sdf.w_current)

    # Add thickness
    h_max = maximum(beam_sdf.thickness_func(s) for s in beam_sdf.s_coords)

    z_min = beam_sdf.z_center + w_min - h_max - T(margin)
    z_max = beam_sdf.z_center + w_max + h_max + T(margin)

    return (x_min, x_max, z_min, z_max)
end

"""
    count_refined_cells_near_beam(refined_grid, beam_sdf::FlexibleBodySDF;
                                   distance=3.0)

Count how many refined cells are within a given distance of the beam.
Useful for monitoring AMR performance.
"""
function count_refined_cells_near_beam(refined_grid, beam_sdf::FlexibleBodySDF{T};
                                       distance::Real=3.0) where T
    update!(beam_sdf)
    count = 0

    if hasfield(typeof(refined_grid), :refined_cells_2d)
        for (cell, level) in refined_grid.refined_cells_2d
            I = CartesianIndex(cell)
            x = loc(0, I, T)
            d = abs(beam_sdf(x, zero(T)))
            if d < T(distance)
                count += 1
            end
        end
    end

    return count
end

# =============================================================================
# BEAM AMR SIMULATION WRAPPER
# =============================================================================

"""
    BeamAMRSimulation{T}

Complete simulation wrapper for flexible bodies with adaptive mesh refinement.
Integrates EulerBernoulliBeam dynamics with AMRSimulation.

# Fields
- `amr_sim`: Underlying AMRSimulation
- `beam`: EulerBernoulliBeam solver
- `beam_sdf`: FlexibleBodySDF for body representation
- `tracker`: BeamAMRTracker for motion-based regridding
- `config`: BeamAMRConfig settings
- `forcing_func`: Optional active forcing function f(s, t)
- `dt_beam`: Time step for beam solver
- `couple_interval`: Couple beam/fluid every N flow steps

# Example
```julia
# Create beam
material = BeamMaterial(ρ=1050.0, E=5e5)
geometry = BeamGeometry(0.2, 51; thickness=fish_thickness_profile(0.2, 0.02))
beam = EulerBernoulliBeam(geometry, material; bc_left=CLAMPED, bc_right=FREE)

# Create simulation
sim = BeamAMRSimulation((256, 128), (2.0, 1.0), beam, 0.3, 0.5;
                         ν=0.001, U=1.0)

# Set forcing
f_wave = traveling_wave_forcing(amplitude=100.0, frequency=2.0)
set_forcing!(sim, f_wave)

# Run
for step in 1:1000
    sim_step!(sim)
end
```
"""
mutable struct BeamAMRSimulation{T<:AbstractFloat}
    amr_sim::AMRSimulation
    beam::EulerBernoulliBeam{T}
    beam_sdf::FlexibleBodySDF{T}
    tracker::BeamAMRTracker{T}
    config::BeamAMRConfig
    forcing_func::Union{Nothing, Function}
    dt_beam::T
    couple_interval::Int
    step_count::Int

    function BeamAMRSimulation(dims::NTuple{N}, L::NTuple{N},
                                beam::EulerBernoulliBeam{T},
                                x_head::Real, z_center::Real;
                                config::BeamAMRConfig=BeamAMRConfig(),
                                thickness_func::Union{Nothing,Function}=nothing,
                                dt_beam::Real=1e-4,
                                couple_interval::Int=1,
                                kwargs...) where {N, T}
        # Create beam body and SDF
        body, beam_sdf = create_beam_body(beam, x_head, z_center;
                                          thickness_func=thickness_func)

        # Create AMR config from beam config
        amr_config = AMRConfig(
            max_level=config.max_level,
            body_distance_threshold=config.beam_distance_threshold,
            velocity_gradient_threshold=config.gradient_threshold,
            vorticity_threshold=config.vorticity_threshold,
            regrid_interval=config.regrid_interval,
            buffer_size=config.buffer_size,
            body_weight=config.beam_weight,
            gradient_weight=config.gradient_weight,
            vorticity_weight=config.vorticity_weight,
            flexible_body=true,
            indicator_change_threshold=0.05,
            min_regrid_interval=config.min_regrid_interval
        )

        # Create AMR simulation with beam body
        amr_sim = AMRSimulation(dims, L; body=body, amr_config=amr_config, kwargs...)

        # Create tracker
        tracker = BeamAMRTracker(beam_sdf)
        mark_regrid!(tracker, 0)

        new{T}(amr_sim, beam, beam_sdf, tracker, config, nothing, T(dt_beam),
               couple_interval, 0)
    end
end

"""
    set_forcing!(sim::BeamAMRSimulation, f::Function)

Set the active forcing function for the beam.
The function should have signature f(s, t) returning force per unit length.
"""
function set_forcing!(sim::BeamAMRSimulation, f::Function)
    sim.forcing_func = f
    return sim
end

"""
    sim_step!(sim::BeamAMRSimulation; kwargs...)

Advance the beam-AMR simulation by one time step.

This performs:
1. Apply active forcing to beam (if set)
2. Advance beam dynamics
3. Update beam SDF
4. Check for AMR regridding based on beam motion
5. Advance fluid with updated body position
"""
function sim_step!(sim::BeamAMRSimulation; λ=quick, kwargs...)
    sim.step_count += 1
    t = sim.step_count * sim.dt_beam

    # Apply forcing if set
    if !isnothing(sim.forcing_func)
        set_active_forcing!(sim.beam, sim.forcing_func, t)
    end

    # Advance beam
    step!(sim.beam, sim.dt_beam)

    # Update SDF with new beam position
    update!(sim.beam_sdf)

    # Check if regridding needed based on beam motion
    if should_regrid(sim.tracker, sim.step_count;
                     min_interval=sim.config.min_regrid_interval,
                     motion_threshold=sim.config.motion_threshold)
        # Perform beam-aware regridding
        perform_beam_regrid!(sim)
    end

    # Advance fluid (remeasure body position)
    if sim.step_count % sim.couple_interval == 0
        sim_step!(sim.amr_sim; remeasure=true, λ=λ, kwargs...)
    end

    return sim
end

"""
    sim_step!(sim::BeamAMRSimulation, t_end; kwargs...)

Advance simulation to target dimensionless time.
"""
function sim_step!(sim::BeamAMRSimulation, t_end::Real; max_steps=typemax(Int), kwargs...)
    n_steps = 0
    while sim_time(sim) < t_end && n_steps < max_steps
        sim_step!(sim; kwargs...)
        n_steps += 1
    end
    return sim
end

"""
    perform_beam_regrid!(sim::BeamAMRSimulation)

Perform AMR regridding based on current beam position.
"""
function perform_beam_regrid!(sim::BeamAMRSimulation{T}) where T
    flow = sim.amr_sim.sim.flow
    config = sim.config

    # Compute combined indicator
    indicator = compute_beam_combined_indicator(flow, sim.beam_sdf;
        beam_threshold=config.beam_distance_threshold,
        gradient_threshold=config.gradient_threshold,
        vorticity_threshold=config.vorticity_threshold,
        beam_weight=config.beam_weight,
        gradient_weight=config.gradient_weight,
        vorticity_weight=config.vorticity_weight)

    # Apply buffer zone
    apply_buffer_zone!(indicator; buffer_size=config.buffer_size)

    # Mark cells for refinement
    cells_to_refine = mark_cells_for_refinement(indicator; threshold=0.5)

    # Update refined grid
    update_refined_cells!(sim.amr_sim.refined_grid, cells_to_refine, config.max_level)

    # Create patches
    create_patches!(sim.amr_sim.composite_pois, sim.amr_sim.refined_grid, flow.μ₀)

    # Synchronize data
    synchronize_base_and_patches!(flow, sim.amr_sim.composite_pois)

    # Mark regrid occurred
    mark_regrid!(sim.tracker, sim.step_count)

    return sim
end

# Forward common functions to underlying simulation
sim_time(sim::BeamAMRSimulation) = sim_time(sim.amr_sim)
time(sim::BeamAMRSimulation) = time(sim.amr_sim)

"""
    beam_info(sim::BeamAMRSimulation)

Print beam and simulation status.
"""
function beam_info(sim::BeamAMRSimulation)
    println("=" ^ 60)
    println("BEAM-AMR SIMULATION STATUS")
    println("=" ^ 60)
    println("  Step: $(sim.step_count)")
    println("  Time (tU/L): $(round(sim_time(sim), digits=4))")

    # Beam state
    w_max = maximum(abs.(sim.beam.w))
    θ_max = maximum(abs.(sim.beam.θ))
    E_total = total_energy(sim.beam)
    println("\n  Beam:")
    println("    Max displacement: $(round(w_max * 1000, digits=2)) mm")
    println("    Max rotation: $(round(rad2deg(θ_max), digits=2))°")
    println("    Total energy: $(round(E_total, digits=4)) J")

    # AMR status
    n_refined = num_refined_cells(sim.amr_sim.refined_grid)
    n_patches = num_patches(sim.amr_sim.composite_pois)
    println("\n  AMR:")
    println("    Refined cells: $n_refined")
    println("    Active patches: $n_patches")
    println("    Last regrid: step $(sim.tracker.last_regrid_step)")

    # Bounding box
    bbox = get_beam_bounding_box(sim.beam_sdf; margin=0.01)
    println("\n  Beam bounding box:")
    println("    x: [$(round(bbox[1], digits=3)), $(round(bbox[2], digits=3))]")
    println("    z: [$(round(bbox[3], digits=3)), $(round(bbox[4], digits=3))]")
    println("=" ^ 60)
end

"""
    get_flow(sim::BeamAMRSimulation)

Get the underlying Flow struct.
"""
get_flow(sim::BeamAMRSimulation) = sim.amr_sim.sim.flow

"""
    get_beam(sim::BeamAMRSimulation)

Get the EulerBernoulliBeam.
"""
get_beam(sim::BeamAMRSimulation) = sim.beam

# =============================================================================
# CONVENIENCE CONSTRUCTORS
# =============================================================================

"""
    swimming_fish_simulation(; L_fish=0.2, Re=1000, St=0.3, n_nodes=51,
                              grid_size=(256, 128), domain=(2.0, 1.0),
                              amr_config=BeamAMRConfig(), kwargs...)

Create a ready-to-run swimming fish simulation with AMR.

# Arguments
- `L_fish`: Fish body length (default: 0.2)
- `Re`: Reynolds number based on body length (default: 1000)
- `St`: Strouhal number for tail beat (default: 0.3)
- `n_nodes`: Number of beam nodes (default: 51)
- `grid_size`: Grid dimensions (default: (256, 128))
- `domain`: Physical domain size (default: (2.0, 1.0))
- `amr_config`: AMR configuration (default: BeamAMRConfig())

# Returns
- `BeamAMRSimulation` ready for time stepping
"""
function swimming_fish_simulation(;
        L_fish::Real=0.2,
        Re::Real=1000.0,
        St::Real=0.3,
        n_nodes::Int=51,
        grid_size::NTuple{2,Int}=(256, 128),
        domain::NTuple{2,Real}=(2.0, 1.0),
        amr_config::BeamAMRConfig=BeamAMRConfig(),
        E::Real=5e5,
        ρ_fish::Real=1050.0,
        h_max::Real=0.02,
        kwargs...)

    T = Float64

    # Material properties
    material = BeamMaterial(ρ=ρ_fish, E=E)

    # Fish body geometry with NACA-like profile
    h_func = fish_thickness_profile(L_fish, h_max)
    geometry = BeamGeometry(L_fish, n_nodes; thickness=h_func, width=h_max)

    # Create beam
    beam = EulerBernoulliBeam(geometry, material;
                              bc_left=CLAMPED, bc_right=FREE,
                              damping=0.5)

    # Compute viscosity from Reynolds number
    U = one(T)  # Reference velocity
    ν = U * L_fish / Re

    # Position fish in domain (head at 1/5 of domain, centered vertically)
    x_head = domain[1] / 5
    z_center = domain[2] / 2

    # Create simulation
    sim = BeamAMRSimulation(grid_size, domain, beam, x_head, z_center;
                            config=amr_config,
                            ν=ν, U=U, L_char=L_fish,
                            kwargs...)

    # Set traveling wave forcing based on Strouhal number
    # St = f * A / U, frequency f = St * U / A
    # Typical amplitude A ~ 0.1 * L for fish
    A_tail = 0.1 * L_fish
    freq = St * U / A_tail

    f_wave = traveling_wave_forcing(
        amplitude=100.0,  # Will be adjusted by Strouhal
        frequency=freq,
        wavelength=L_fish,
        envelope=:carangiform,
        L=L_fish
    )

    set_forcing!(sim, f_wave)

    return sim
end
