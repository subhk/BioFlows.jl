using BioFlows
using Statistics
using Random

"""
    flow_past_cylinder_2d_sim(; nx=240, nz=240,
                                 Lx=4.0, Lz=4.0,
                                 ν=0.001, U=1,
                                 radius=nothing,
                                 dt=nothing,
                                 inletBC=nothing,
                                 perdir=(2,),
                                 outletBC=true)

Construct the classic 2D cylinder benchmark with explicit control over grid
resolution `(nx,nz)` and physical domain `(Lx,Lz)`. The cylinder radius defaults
to `0.2`, but can be overridden with the `radius` keyword (specified in
physical units). Provide a fixed time step via `dt` (set to `nothing` to keep
adaptive CFL stepping) and customise boundary conditions with `inletBC`,
`perdir`, and `outletBC` (e.g. `perdir=(2,)` makes the z-direction periodic).

Returns `(sim, meta)` where `meta` records the domain, grid, cell size, radius,
time step (or `nothing` when adaptive), and boundary configuration used.
"""
function flow_past_cylinder_2d_sim(; nx::Int=240,
                                      nz::Int=240,
                                      Lx::Real=4.0,
                                      Lz::Real=4.0,
                                      ν::Real=0.001,
                                      U::Real=1.0,
                                      radius::Union{Nothing,Real}=nothing,
                                      dt=nothing,
                                      inletBC=nothing,
                                      perdir=(2,),
                                      outletBC::Bool=true)
    dx = Lx / nx
    dz = Lz / nz
    @assert isapprox(dx, dz; atol=1e-8, rtol=1e-6) "Non-uniform cell spacing (Δx ≠ Δz) is not supported"

    radius_phys = isnothing(radius) ? 0.2 : radius

    center_x_cells = nx / 12 - 1
    center_z_cells = nz / 2 - 1

    radius_cells = radius_phys / dx

    sdf(x, t) = √((x[1] - center_x_cells)^2 + (x[2] - center_z_cells)^2) - radius_cells
    boundary = isnothing(inletBC) ? (U, 0) : inletBC

    diameter = 2radius_phys
    base_kwargs = (; ν = ν,
                    perdir = perdir,
                    outletBC = outletBC,
                    body = AutoBody(sdf),
                    L_char = diameter)

    sim = dt === nothing ?
        Simulation((nx, nz), boundary, (Lx, Lz); base_kwargs...) :
        Simulation((nx, nz), boundary, (Lx, Lz); Δt = dt, base_kwargs...)

    meta = (
        domain = (Lx, Lz),
        grid = (nx, nz),
        cell_size = (dx, dz),
        radius = radius_phys,
        Δt = sim.flow.Δt[end],
        fixed_dt = dt,
        inletBC = boundary,
        perdir = perdir,
        outletBC = outletBC,
        ν = ν,
        U = U,
    )
    return sim, meta
end

"""
    run_flow_past_cylinder(; steps=nothing, discard=200,
                              final_time=50.0,
                              save_center_fields=false,
                              center_interval_time=nothing,
                              center_filename="center_fields.jld2",
                              diagnostic_interval=100,
                              kwargs...)

Advance the simulation for `steps` calls (or until `final_time`, whichever comes
first), recording the force coefficients and, optionally, saving cell-centred
velocity/vorticity snapshots. Returns `(sim, history, stats, writer, diagnostics)`.
"""
function run_flow_past_cylinder(; steps::Union{Nothing,Int}=nothing,
                                discard::Int=200,
                                final_time=50.0,
                                save_center_fields::Bool=true,
                                center_interval_time=1.0,
                                center_filename::AbstractString="center_fields.jld2",
                                diagnostic_interval::Int=100,
                                kwargs...)

    sim, meta_tuple = flow_past_cylinder_2d_sim(; kwargs...)

    Random.seed!(42)
    perturb!(sim; noise=3e-2)

    meta = NamedTuple(meta_tuple)

    history = NamedTuple[]
    diag_interval = diagnostic_interval > 0 ? diagnostic_interval : typemax(Int)
    fixed_dt = meta.fixed_dt

    fixed_dt !== nothing && (sim.flow.Δt[end] = fixed_dt)

    conv_interval = nothing
    sim_time_interval = nothing
    writer = nothing
    if save_center_fields
        sim_time_interval = isnothing(center_interval_time) ? 0.1 * (sim.L / sim.U) : center_interval_time
        conv_interval = sim_time_interval * (sim.U / sim.L)
        writer = CenterFieldWriter(center_filename; interval=conv_interval)
    end

    if isnothing(final_time) && isnothing(steps)
        error("Provide at least one of `final_time` or `steps`.")
    end

    step_limit = isnothing(steps) ? typemax(Int) : steps
    target_time = final_time
    total_steps = 0

    while true
        if target_time !== nothing && sim_time(sim) >= target_time
            break
        end
        total_steps >= step_limit && break

        total_steps += 1
        sim_step!(sim; remeasure=false)
        record_force!(history, sim)
        writer !== nothing && file_save!(writer, sim)
        fixed_dt !== nothing && (sim.flow.Δt[end] = fixed_dt)

        if total_steps % diag_interval == 0
            stats_line = summarize_force_history(history; discard=min(discard, length(history)))
            diag_line = compute_diagnostics(sim)

            t        = round(sim_time(sim); digits=3)
            drag_val = round(stats_line.drag_mean; digits=3)
            lift_val = round(stats_line.lift_rms; digits=3)
            max_u    = round(diag_line.max_u; digits=3)
            max_w    = round(diag_line.max_w; digits=3)
            cfl_val  = round(diag_line.CFL; digits=3)
            dt_val   = round(diag_line.Δt; digits=3)

            println("[iter $(total_steps)] t=$(t) drag=$(drag_val), lift_rms=$(lift_val), max_u=$(max_u), max_w=$(max_w), CFL=$(cfl_val), Δt=$(dt_val)")
        end
    end

    stats = summarize_force_history(history; discard)
    diagnostics = merge(compute_diagnostics(sim),
                        meta,
                        (steps=total_steps,
                         final_time=sim_time(sim),
                         target_time=target_time,
                         center_interval_conv=conv_interval,
                         center_interval_time=sim_time_interval))

    return sim, history, stats, writer, diagnostics
end

"""
    summarize_force_history(history; discard=200)

Compute drag/lift statistics from the recorded force history.
"""
function summarize_force_history(history; discard::Int=200)
    length(history) ≤ discard && return (drag_mean=NaN, drag_std=NaN,
                                         lift_rms=NaN, samples=0)

    trimmed = view(history, discard+1:length(history))
    drag_coeffs = [sample.total_coeff[1] for sample in trimmed]
    lift_coeffs = [sample.total_coeff[2] for sample in trimmed]

    return (
        drag_mean = mean(drag_coeffs),
        drag_std = std(drag_coeffs; corrected=false),
        lift_rms = sqrt(mean(abs2, lift_coeffs)),
        samples = length(trimmed)
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    sim, history, stats, writer, diagnostics = run_flow_past_cylinder(;
                                                save_center_fields=true,
                                                center_filename="cylinder_center_fields.jld2")

    println("Flow past cylinder 2D complete")
    println("  steps         = ", diagnostics.steps)
    println("  final time    = ", diagnostics.final_time)
    println("  grid (nx,nz)  = ", diagnostics.grid)
    println("  domain (Lx,Lz)= ", diagnostics.domain)
    println("  ν             = ", diagnostics.ν)
    for (name, value) in pairs(stats)
        println("  ", name, " = ", value)
    end
end
