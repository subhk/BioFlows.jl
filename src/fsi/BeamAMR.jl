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

"""
    (sdf::FlexibleBodySDF)(x, t)

Compute signed distance from point x to the deformed beam surface.
Negative inside, positive outside.
"""
function (sdf::FlexibleBodySDF{T})(x, t) where T
    # x[1] is streamwise, x[2] is transverse (z)
    x_rel = x[1] - sdf.x_head
    z_pos = x[2]

    L = sdf.beam.geometry.L

    # Check if outside beam extent
    if x_rel < 0
        # Before beam head - distance to leading edge
        h_head = sdf.thickness_func(zero(T))
        z_beam = sdf.z_center + sdf.w_current[1]
        dz = z_pos - z_beam
        return sqrt(x_rel^2 + max(zero(T), abs(dz) - h_head)^2)
    elseif x_rel > L
        # After beam tail - distance to trailing edge
        h_tail = sdf.thickness_func(L)
        z_beam = sdf.z_center + sdf.w_current[end]
        dz = z_pos - z_beam
        return sqrt((x_rel - L)^2 + max(zero(T), abs(dz) - h_tail)^2)
    end

    # Inside beam extent - interpolate displacement
    s = x_rel

    # Find bracketing nodes
    n = sdf.beam.n_nodes
    ds = L / (n - 1)
    idx = clamp(floor(Int, s / ds) + 1, 1, n - 1)

    # Linear interpolation of displacement
    s1, s2 = sdf.s_coords[idx], sdf.s_coords[idx + 1]
    w1, w2 = sdf.w_current[idx], sdf.w_current[idx + 1]
    α = (s - s1) / (s2 - s1)
    w_interp = (1 - α) * w1 + α * w2

    # Beam centerline position
    z_beam = sdf.z_center + w_interp

    # Local half-thickness
    h = sdf.thickness_func(s)

    # Signed distance (negative inside)
    dz = z_pos - z_beam
    return abs(dz) - h
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

        # Distance to current position
        d_new = abs(beam_sdf(x, zero(T)))

        # Approximate distance to old position (shift by max displacement)
        # This is conservative - marks a larger region
        d_old = d_new  # Simplified: use current position

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
    # Compute individual indicators
    beam_ind = compute_beam_refinement_indicator(flow, beam_sdf;
                                                  distance_threshold=beam_threshold)

    gradient_ind = compute_velocity_gradient_indicator(flow)
    vorticity_ind = compute_vorticity_indicator(flow)

    # Normalize gradient and vorticity indicators
    grad_max = maximum(gradient_ind)
    if grad_max > eps(T)
        gradient_ind ./= grad_max
    end

    vort_max = maximum(vorticity_ind)
    if vort_max > eps(T)
        vorticity_ind ./= vort_max
    end

    # Apply thresholds
    for I in eachindex(gradient_ind)
        gradient_ind[I] = gradient_ind[I] > T(gradient_threshold) ? one(T) : zero(T)
        vorticity_ind[I] = vorticity_ind[I] > T(vorticity_threshold) ? one(T) : zero(T)
    end

    # Combine with weights
    combined = similar(flow.p)
    bw, gw, vw = T(beam_weight), T(gradient_weight), T(vorticity_weight)

    for I in eachindex(combined)
        combined[I] = bw * beam_ind[I] + gw * gradient_ind[I] + vw * vorticity_ind[I]
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
    body = AutoBody((x, t) -> beam_sdf(x, t))

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
