#!/usr/bin/env julia

using BioFlows
using StaticArrays: SVector
using LinearAlgebra: norm

"""
    donut_sim(; n=2^6, Re=1600, U=1, major_ratio=0.35, minor_ratio=0.12)

Construct the 3D torus (donut) benchmark. `major_ratio` and
`minor_ratio` scale the major and minor radii relative to the grid extent.
"""
function donut_sim(; n::Int=2^6, Re::Real=1600, U::Real=1,
                     major_ratio::Real=0.35, minor_ratio::Real=0.12)
    dims = (n, n, n)
    center = SVector(map(d -> d / 2 - 1, dims)...)
    L = maximum(dims)
    major = major_ratio * L
    minor = minor_ratio * L
    sdf(x, t) = begin
        xp = x .- center
        radial = norm(SVector(xp[1], xp[2])) - major
        q = SVector(radial, xp[3])
        norm(q) - minor
    end
    diameter = 2major
    # Domain size = grid cells (Δx = 1), L_char = diameter for force coefficients
    Simulation(dims, (U, 0, 0), (Float64(n), Float64(n), Float64(n));
               ν = U * diameter / Re,
               body = AutoBody(sdf),
               perdir = (1,),
               L_char = diameter)
end

"""
    run_donut(; steps=200, kwargs...)

Advance the 3D torus simulation for `steps` iterations. Returns `(sim, history)`
where each history entry records `(step, time, drag, side, spanwise)`.
"""
function run_donut(; steps::Int=200, kwargs...)
    sim = donut_sim(; kwargs...)
    history = Vector{NamedTuple}(undef, steps)
    for k in 1:steps
        sim_step!(sim; remeasure=false)
        coeff = total_force(sim) ./ (0.5 * sim.L * sim.U^2)
        history[k] = (step=k,
                      time=sim_time(sim),
                      drag=coeff[1],
                      side=coeff[2],
                      spanwise=coeff[3])
    end
    sim, history
end

if abspath(PROGRAM_FILE) == @__FILE__
    sim, history = run_donut()
    final = history[end]
    @info "BioFlows 3D donut example complete" steps=final.step time=final.time drag=final.drag side=final.side spanwise=final.spanwise
end
