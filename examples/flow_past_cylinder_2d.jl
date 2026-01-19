using BioFlows
using Statistics
using Random

"""
    flow_past_cylinder_2d_sim(; nx=240, nz=240, Lx=4f0, Lz=4f0, ν=0.001f0, ...)

Construct the classic 2D cylinder benchmark.
Returns `(sim, meta)`.
"""
function flow_past_cylinder_2d_sim(; nx::Int=240, nz::Int=240,
                                      Lx::Real=4f0, Lz::Real=4f0,
                                      ν::Real=0.001f0, U::Real=1f0,
                                      radius::Union{Nothing,Real}=nothing,
                                      dt=nothing, inletBC=nothing,
                                      perdir=(2,), outletBC::Bool=true)
    dx = Lx / nx
    dz = Lz / nz
    @assert isapprox(dx, dz; atol=1e-8, rtol=1e-6) "Non-uniform cell spacing (Δx ≠ Δz) is not supported"

    radius_phys = isnothing(radius) ? 0.2f0 : radius
    center_x_cells = nx / 12 - 1
    center_z_cells = nz / 2 - 1
    radius_cells = radius_phys / dx

    sdf(x, t) = √((x[1] - center_x_cells)^2 + (x[2] - center_z_cells)^2) - radius_cells
    boundary = isnothing(inletBC) ? (U, 0) : inletBC

    diameter = 2radius_phys
    base_kwargs = (; ν = ν, perdir = perdir, outletBC = outletBC,
                    body = AutoBody(sdf), L_char = diameter)

    sim = dt === nothing ?
        Simulation((nx, nz), (Lx, Lz); inletBC=boundary, base_kwargs...) :
        Simulation((nx, nz), (Lx, Lz); inletBC=boundary, Δt = dt, base_kwargs...)

    meta = (domain = (Lx, Lz), grid = (nx, nz), cell_size = (dx, dz),
            radius = radius_phys, Δt = sim.flow.Δt[end], fixed_dt = dt,
            inletBC = boundary, perdir = perdir, outletBC = outletBC, ν = ν, U = U)
    return sim, meta
end

"""
    summarize_force_history(history; discard=200)

Compute drag/lift statistics from recorded force history.
"""
function summarize_force_history(history; discard::Int=200)
    length(history) ≤ discard && return (drag_mean=NaN, drag_std=NaN, lift_rms=NaN, samples=0)
    trimmed = view(history, discard+1:length(history))
    drag_coeffs = [sample.total_coeff[1] for sample in trimmed]
    lift_coeffs = [sample.total_coeff[2] for sample in trimmed]
    return (drag_mean = mean(drag_coeffs), drag_std = std(drag_coeffs; corrected=false),
            lift_rms = sqrt(mean(abs2, lift_coeffs)), samples = length(trimmed))
end

if abspath(PROGRAM_FILE) == @__FILE__
    @info "Running flow past cylinder 2D example" threads=Threads.nthreads() backend=BioFlows.backend
    sim, meta = flow_past_cylinder_2d_sim(; nx=64, nz=64)
    for _ in 1:100
        sim_step!(sim; remeasure=false)
    end
    drag, lift = force_coefficients(sim)
    @info "Flow past cylinder complete" Cd=drag Cl=lift
end
