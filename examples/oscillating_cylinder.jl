#!/usr/bin/env julia

using BioFlows
using StaticArrays: SVector
using LinearAlgebra: norm

"""
    oscillating_cylinder_sim(; n=3*2^5, m=2^6, ν=0.01f0, U=1f0, St=0.2f0, amplitude=0.25f0)

Oscillating cylinder benchmark. The cylinder translates sinusoidally in the
cross-flow direction with Strouhal number `St` and peak-to-peak amplitude
`2*amplitude*radius`.

# Arguments
- `n`, `m`: Grid dimensions
- `ν`: Kinematic viscosity (m²/s)
- `U`: Inflow velocity (m/s)
- `St`: Strouhal number for oscillation frequency
- `amplitude`: Oscillation amplitude relative to radius
"""
function oscillating_cylinder_sim(; n::Int=3*2^5, m::Int=2^6,
                                     ν::Real=0.01f0, U::Real=1f0,
                                     St::Real=0.2f0, amplitude::Real=0.25f0)
    radius = m / 8
    center = SVector(m / 2 - 1, m / 2 - 1)
    sdf(x, t) = norm(x .- center) - radius
    displacement(t) = amplitude * radius * sin(2π * St * t)
    move(x, t) = x - SVector(zero(t), displacement(t))
    diameter = 2radius
    # Domain size = grid cells (Δx = 1), L_char = diameter for force coefficients
    Simulation((n, m), (Float32(n), Float32(m));
               inletBC=(U, 0),
               ν=ν,
               body=AutoBody(sdf, move),
               L_char=diameter)
end

"""
    run_oscillating_cylinder(; steps=400, St=0.2f0, amplitude=0.25f0, kwargs...)

Advance the oscillating-cylinder case for `steps` solver iterations while
recording the instantaneous displacement and total force coefficients.
Returns `(sim, history)` where each history entry stores `(step, time, y_disp,
drag, lift)`.
"""
function run_oscillating_cylinder(; steps::Int=400, St::Real=0.2f0, amplitude::Real=0.25f0, kwargs...)
    sim = oscillating_cylinder_sim(; St, amplitude, kwargs...)
    history = Vector{NamedTuple}(undef, steps)
    radius = sim.L / 2
    for k in 1:steps
        sim_step!(sim; remeasure=true)
        t = BioFlows.time(sim) # physical time used by the body motion
        disp = amplitude * radius * sin(2π * St * t)
        coeff = total_force(sim) ./ (0.5f0 * sim.L * sim.U^2)
        history[k] = (step=k,
                      time=sim_time(sim),
                      y_disp=disp,
                      drag=coeff[1],
                      lift=coeff[2])
    end
    sim, history
end

# =============================================================================
# AMR (ADAPTIVE MESH REFINEMENT) SUPPORT FOR OSCILLATING CYLINDER
# =============================================================================
# These functions demonstrate how to use AMR with rigid moving bodies.
# The RigidBodyAMRConfig optimizes settings for objects that translate or rotate
# without deforming, allowing efficient mesh refinement that tracks the body.
# =============================================================================

"""
    oscillating_cylinder_amr_sim(; n=3*2^5, m=2^6, ν=0.01f0, U=1f0, St=0.2f0,
                                   amplitude=0.25f0, max_level=2, kwargs...)

Create an oscillating cylinder simulation with Adaptive Mesh Refinement.
Uses RigidBodyAMRConfig for optimal settings with moving rigid bodies.

# Arguments
- `n`, `m`: Grid dimensions
- `ν`: Kinematic viscosity (m²/s)
- `U`: Inflow velocity (m/s)
- `St`: Strouhal number for oscillation frequency
- `amplitude`: Oscillation amplitude relative to radius
- `max_level`: Maximum AMR refinement level (1=2x, 2=4x, 3=8x)
- `kwargs...`: Additional AMR configuration options

# Example
```julia
sim = oscillating_cylinder_amr_sim(max_level=2)
for i in 1:100
    sim_step!(sim; remeasure=true)
end
```
"""
function oscillating_cylinder_amr_sim(; n::Int=3*2^5, m::Int=2^6,
                                        ν::Real=0.01f0, U::Real=1f0,
                                        St::Real=0.2f0, amplitude::Real=0.25f0,
                                        max_level::Int=2, kwargs...)
    radius = m / 8
    center = SVector(m / 2 - 1, m / 2 - 1)
    sdf(x, t) = norm(x .- center) - radius
    displacement(t) = amplitude * radius * sin(2π * St * t)
    move(x, t) = x - SVector(zero(t), displacement(t))
    diameter = 2radius

    # Use RigidBodyAMRConfig for rigid moving bodies
    amr_config = BioFlows.RigidBodyAMRConfig(;
        max_level=max_level,
        body_distance_threshold=2.5f0,  # Refinement distance in grid cells
        regrid_interval=10,           # Regrid every 10 steps
        indicator_change_threshold=0.15f0,  # Motion sensitivity
        kwargs...
    )

    BioFlows.AMRSimulation((n, m), (Float32(n), Float32(m));
                           inletBC=(U, 0),
                           ν=ν,
                           body=AutoBody(sdf, move),
                           L_char=diameter,
                           amr_config=amr_config)
end

"""
    run_oscillating_cylinder_amr(; steps=400, St=0.2f0, amplitude=0.25f0, kwargs...)

Run oscillating cylinder simulation with AMR enabled.
Records displacement, forces, and AMR statistics.

# Returns
- `(sim, history)` where history contains per-step data including AMR info
"""
function run_oscillating_cylinder_amr(; steps::Int=400, St::Real=0.2f0,
                                        amplitude::Real=0.25f0, verbose::Bool=false,
                                        kwargs...)
    sim = oscillating_cylinder_amr_sim(; St, amplitude, kwargs...)
    history = Vector{NamedTuple}(undef, steps)
    radius = sim.L / 2

    for k in 1:steps
        sim_step!(sim; remeasure=true)
        t = BioFlows.time(sim)
        disp = amplitude * radius * sin(2π * St * t)
        coeff = total_force(sim) ./ (0.5f0 * sim.L * sim.U^2)

        # Get AMR statistics
        amr_stats = BioFlows.amr_info(sim)

        history[k] = (step=k,
                      time=sim_time(sim),
                      y_disp=disp,
                      drag=coeff[1],
                      lift=coeff[2],
                      num_patches=amr_stats.num_patches,
                      refined_cells=amr_stats.refined_cells)

        if verbose && k % 50 == 0
            @info "Step $k" time=sim_time(sim) drag=coeff[1] lift=coeff[2] patches=amr_stats.num_patches
        end
    end
    sim, history
end

"""
    rotating_cylinder_amr_sim(; n=3*2^5, m=2^6, ν=0.01f0, U=1f0, ω=0.5f0, max_level=2)

Create a rotating cylinder simulation with AMR.
The cylinder rotates in place at angular velocity ω.

# Arguments
- `n`, `m`: Grid dimensions
- `ν`: Kinematic viscosity
- `U`: Inflow velocity
- `ω`: Angular velocity (rad/time)
- `max_level`: AMR refinement level

# Example
```julia
sim = rotating_cylinder_amr_sim(ω=1f0, max_level=2)
```
"""
function rotating_cylinder_amr_sim(; n::Int=3*2^5, m::Int=2^6,
                                     ν::Real=0.01f0, U::Real=1f0,
                                     ω::Real=0.5f0, max_level::Int=2)
    radius = m / 8
    center = SVector(m / 2 - 1, m / 2 - 1)
    sdf(x, t) = norm(x .- center) - radius

    # For a rotating cylinder, the map rotates coordinates around center
    # The SDF is rotationally symmetric, so motion is purely kinematic
    function rotate(x, t)
        θ = -ω * t  # Negative because we're mapping back
        dx = x .- center
        c, s = cos(θ), sin(θ)
        rotated = SVector(c * dx[1] - s * dx[2], s * dx[1] + c * dx[2])
        return rotated .+ center
    end

    diameter = 2radius

    # Rotating bodies use same RigidBodyAMRConfig
    amr_config = BioFlows.RigidBodyAMRConfig(;
        max_level=max_level,
        body_distance_threshold=2f0,
        regrid_interval=15,  # Rotation is smooth, less frequent regridding
        indicator_change_threshold=0.2f0
    )

    BioFlows.AMRSimulation((n, m), (Float32(n), Float32(m));
                           inletBC=(U, 0),
                           ν=ν,
                           body=AutoBody(sdf, rotate),
                           L_char=diameter,
                           amr_config=amr_config)
end

"""
    orbiting_cylinder_amr_sim(; n=4*2^5, m=4*2^5, ν=0.01f0, orbit_radius=16f0,
                                orbit_period=100f0, max_level=2)

Create a cylinder orbiting in a circular path with AMR.
Demonstrates large-amplitude rigid body motion tracking.

# Arguments
- `n`, `m`: Grid dimensions
- `ν`: Kinematic viscosity
- `orbit_radius`: Orbital radius in grid cells
- `orbit_period`: Period of one full orbit (in time units)
- `max_level`: AMR refinement level
"""
function orbiting_cylinder_amr_sim(; n::Int=4*2^5, m::Int=4*2^5,
                                     ν::Real=0.01f0,
                                     orbit_radius::Real=16f0,
                                     orbit_period::Real=100f0,
                                     max_level::Int=2)
    cylinder_radius = m / 12
    domain_center = SVector(n / 2, m / 2)

    # Initial center position (at t=0)
    center_at_0 = domain_center .+ orbit_radius .* SVector(1f0, 0f0)

    # Center of cylinder orbits around domain center
    function orbit_center(t)
        θ = 2π * t / orbit_period
        domain_center .+ orbit_radius .* SVector(cos(θ), sin(θ))
    end

    # SDF in reference frame (time-independent)
    sdf(x, t) = norm(x .- center_at_0) - cylinder_radius

    # Map: translate point back to reference frame
    # For AutoBody with compose=true: sdf'(x,t) = sdf(map(x,t), t)
    # = norm((x - displacement) - center_at_0) - radius
    # = norm(x - center_at_t) - radius (correct distance to current position)
    function move(x, t)
        displacement = orbit_center(t) .- center_at_0
        return x .- displacement
    end

    diameter = 2cylinder_radius

    # For orbiting motion, use more aggressive motion tracking
    amr_config = BioFlows.RigidBodyAMRConfig(;
        max_level=max_level,
        body_distance_threshold=3f0,
        regrid_interval=8,  # More frequent for larger motion
        indicator_change_threshold=0.12f0,
        min_regrid_interval=3
    )

    BioFlows.AMRSimulation((n, m), (Float32(n), Float32(m));
                           inletBC=(0, 0),  # Quiescent fluid
                           ν=ν,
                           body=AutoBody(sdf, move),
                           L_char=diameter,
                           amr_config=amr_config)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running oscillating cylinder simulation...")
    println("Choose simulation type:")
    println("  1. Standard (no AMR)")
    println("  2. With AMR")
    println("  3. Rotating cylinder with AMR")

    # Default to standard simulation
    sim, history = run_oscillating_cylinder()
    final = history[end]
    @info "BioFlows oscillating-cylinder example complete" steps=final.step time=final.time displacement=final.y_disp drag=final.drag lift=final.lift
end
