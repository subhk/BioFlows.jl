#!/usr/bin/env julia

using BioFlows
using StaticArrays: SVector
using LinearAlgebra: norm

"""
    oscillating_cylinder_sim(; n=3*2^5, m=2^6, ν=0.01, U=1, St=0.2, amplitude=0.25)

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
                                     ν::Real=0.01, U::Real=1,
                                     St::Real=0.2, amplitude::Real=0.25)
    radius = m / 8
    center = SVector(m / 2 - 1, m / 2 - 1)
    sdf(x, t) = norm(x .- center) - radius
    displacement(t) = amplitude * radius * sin(2π * St * t)
    move(x, t) = x - SVector(zero(t), displacement(t))
    diameter = 2radius
    # Domain size = grid cells (Δx = 1), L_char = diameter for force coefficients
    Simulation((n, m), (Float64(n), Float64(m));
               inletBC=(U, 0),
               ν=ν,
               body=AutoBody(sdf, move),
               L_char=diameter)
end

"""
    run_oscillating_cylinder(; steps=400, St=0.2, amplitude=0.25, kwargs...)

Advance the oscillating-cylinder case for `steps` solver iterations while
recording the instantaneous displacement and total force coefficients.
Returns `(sim, history)` where each history entry stores `(step, time, y_disp,
drag, lift)`.
"""
function run_oscillating_cylinder(; steps::Int=400, St::Real=0.2, amplitude::Real=0.25, kwargs...)
    sim = oscillating_cylinder_sim(; St, amplitude, kwargs...)
    history = Vector{NamedTuple}(undef, steps)
    radius = sim.L / 2
    for k in 1:steps
        sim_step!(sim; remeasure=true)
        t = BioFlows.time(sim) # physical time used by the body motion
        disp = amplitude * radius * sin(2π * St * t)
        coeff = total_force(sim) ./ (0.5 * sim.L * sim.U^2)
        history[k] = (step=k,
                      time=sim_time(sim),
                      y_disp=disp,
                      drag=coeff[1],
                      lift=coeff[2])
    end
    sim, history
end

if abspath(PROGRAM_FILE) == @__FILE__
    sim, history = run_oscillating_cylinder()
    final = history[end]
    @info "BioFlows oscillating-cylinder example complete" steps=final.step time=final.time displacement=final.y_disp drag=final.drag lift=final.lift
end
