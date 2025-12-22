#!/usr/bin/env julia

using BioFlows
using StaticArrays: SVector
using LinearAlgebra: norm

"""
    circle_sim(; n=3*2^5, m=2^6, Re=100, U=1)

Construct the classic circle simulation using BioFlows.
The setup matches the official WaterLily example with a
uniform inflow, stationary cylinder, and Re based on the diameter.
"""
function circle_sim(; n::Int=3*2^5, m::Int=2^6, Re::Real=100, U::Real=1)
    radius = m / 8
    center = SVector(m / 2 - 1, m / 2 - 1)
    sdf(x, t) = norm(x .- center) - radius
    Simulation((n, m), (U, 0), 2radius; ν=U * 2radius / Re, body=AutoBody(sdf))
end

"""
    run_circle(; steps=250, remeasure=false)

Advance the circle simulation for `steps` convective time steps,
recording the pressure-force coefficients after each call to `sim_step!`.
Returns both the simulation object and a vector of `(step, time, drag, lift)`
measurements.
"""
function run_circle(; steps::Int=250, remeasure::Bool=false)
    sim = circle_sim()
    history = Vector{NamedTuple}(undef, steps)
    for k in 1:steps
        sim_step!(sim; remeasure)
        coeff = pressure_force(sim) ./ (0.5 * sim.L * sim.U^2)
        history[k] = (step=k,
                      time=sim_time(sim),
                      drag=coeff[1],
                      lift=coeff[2])
    end
    sim, history
end

if Base.program_file() == @__FILE__
    sim, history = run_circle()
    final = history[end]
    @info "BioFlows circle example complete" steps=final.step time=final.time drag=final.drag lift=final.lift
end
