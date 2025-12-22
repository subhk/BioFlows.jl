#!/usr/bin/env julia

using BioFlows
using StaticArrays: SVector
using LinearAlgebra: norm

"""
    sphere_sim(; n=3*2^5, m=2^6, ℓ=2^6, Re=300, U=1, T=Float32, mem=Array)

Construct the 3D sphere example. The flow travels in the x-direction
past a stationary sphere of diameter `2*radius`.
"""
function sphere_sim(; n::Int=3*2^5, m::Int=2^6, ℓ::Int=2^6,
                       Re::Real=300, U::Real=1,
                       T::Type{<:AbstractFloat}=Float32,
                       mem=Array)
    dims = (n, m, ℓ)
    radius = m / 8
    center = SVector(map(d -> d / 2 - 1, dims)...)
    sdf(x, t) = norm(x .- center) - radius
    Simulation(dims, (U, 0, 0), 2radius;
               ν=U * 2radius / Re,
               body=AutoBody(sdf),
               T, mem)
end

"""
    run_sphere(; steps=150, kwargs...)

Advance the 3D sphere simulation for `steps` iterations collecting pressure and
viscous force coefficients. Returns `(sim, history)` where entries store
`(step, time, drag, side, spanwise)`.
"""
function run_sphere(; steps::Int=150, kwargs...)
    sim = sphere_sim(; kwargs...)
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
    sim, history = run_sphere()
    final = history[end]
    @info "BioFlows 3D sphere example complete" steps=final.step time=final.time drag=final.drag side=final.side spanwise=final.spanwise
end
