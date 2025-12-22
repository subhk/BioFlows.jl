using Random
using MPI

abstract type AbstractBDIMSimulation end

mutable struct Simulation <: AbstractBDIMSimulation
    U::Float64
    L::Float64
    ν::Float64
    ε::Float64
    dt::Float64
    config::SimulationConfig
    solver::Any
    state_old::Any
    state_new::Any
    use_mpi::Bool
    grid::StaggeredGrid
end

Simulation(sim::Simulation) = sim

function Simulation(dims::NTuple{2,Int}, uBC::Tuple{<:Real,<:Real}, L::Real;
                    domain::Tuple{<:Real,<:Real} = (L, L * dims[2] / dims[1]),
                    ν::Float64,
                    Δt::Float64 = 0.01,
                    body::RigidBody,
                    density::Float64 = 1.0,
                    smoothing::Float64 = 1.2,
                    use_mpi::Bool = false,
                    output_file::AbstractString = "bdim_output")
    nx, nz = dims
    domain_val = (Float64(domain[1]), Float64(domain[2]))
    u_tuple = (Float64(uBC[1]), Float64(uBC[2]))
    config = create_2d_simulation_config(
        nx = nx, nz = nz,
        Lx = domain_val[1], Lz = domain_val[2],
        density_value = density,
        nu = ν,
        inlet_velocity = u_tuple[1],
        outlet_type = :pressure,
        wall_type = :free_slip,
        dt = Δt,
        final_time = Δt,
        immersed_boundary_method = BDIM,
        use_mpi = use_mpi,
        output_interval = 1.0,
        output_file = output_file,
        output_save_flow_field = true,
        output_save_body_positions = true,
        output_save_force_coefficients = true)

    if !(body.shape isa Circle)
        error("Current BDIM wrapper supports Circle bodies only")
    end
    config = add_rigid_circle!(config, copy(body.center), (body.shape::Circle).radius)

    solver = create_solver(config)

    if use_mpi
        local_old = MPISolutionState2D(solver.decomp)
        local_new = MPISolutionState2D(solver.decomp)
        local_old.u .= u_tuple[1]
        local_old.w .= u_tuple[2]
        local_old.p .= 0.0
        local_old.t = 0.0
        local_old.step = 0
        local_new.u .= local_old.u
        local_new.w .= local_old.w
        local_new.p .= 0.0
        state_old, state_new = local_old, local_new
    else
        state_old = initialize_simulation(config)
        state_old.u .= u_tuple[1]
        state_old.w .= u_tuple[2]
        state_new = deepcopy(state_old)
    end

    grid = StaggeredGrid2D(config.nx, config.nz, config.Lx, config.Lz)

    ENV["BIOFLOWS_MASKS_EPS_MUL"] = string(smoothing)

    Simulation(Float64(hypot(u_tuple[1], u_tuple[2])), Float64(L), ν, smoothing * max(grid.dx, grid.dz), Δt,
               config, solver, state_old, state_new, use_mpi, grid)
end

sim_time(sim::Simulation) = sim.state_old.t
grid(sim::Simulation) = sim.grid
state(sim::Simulation) = sim.state_old
bodies(sim::Simulation) = sim.config.rigid_bodies
fluid(sim::Simulation) = sim.config.fluid

function measure!(sim::Simulation)
    return nothing
end

function sim_info(sim::Simulation)
    println("t = $(round(sim_time(sim), digits = 4)), Δt = $(round(sim.dt, digits = 4))")
end

function perturb!(sim::Simulation; noise::Float64 = 0.02,
                  seed::Union{Nothing,Integer}=nothing,
                  span::Tuple{Float64,Float64} = (0.0, Inf))
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    if sim.use_mpi
        grid_local = sim.solver.local_grid
        decomp = sim.solver.decomp
        n_ghost = decomp.n_ghost
        u = sim.state_old.u
        w = sim.state_old.w
        u_scale = max(sim.U, 1e-6)
        body_x = sim.config.rigid_bodies.bodies[1].center[1]
        for i in (n_ghost+1):(size(u, 1) - n_ghost)
            x = grid_local.xu[i]
            if span[1] <= x <= span[2]
                for j in (n_ghost+1):(size(u, 2) - n_ghost)
                    weight = exp(-((x - body_x) / (0.4 * sim.L))^2)
                    u[i, j] += noise * u_scale * randn(rng)
                end
            end
        end
        for i in (n_ghost+1):(size(w, 1) - n_ghost)
            x = grid_local.x[i]
            if span[1] <= x <= span[2]
                for j in (n_ghost+1):(size(w, 2) - n_ghost)
                    z = grid_local.zw[j]
                    envelope = sinpi(z / sim.grid.Lz)
                    w[i, j] += 0.5 * noise * u_scale * envelope * randn(rng)
                end
            end
        end
        sim.state_new.u .= sim.state_old.u
        sim.state_new.w .= sim.state_old.w
    else
        u = sim.state_old.u
        w = sim.state_old.w
        grid_global = sim.grid
        u_scale = max(sim.U, 1e-6)
        body_x = sim.config.rigid_bodies.bodies[1].center[1]
        for (i, x) in enumerate(grid_global.xu)
            if span[1] <= x <= span[2]
                for j in axes(u, 2)
                    weight = exp(-((x - body_x) / (0.4 * sim.L))^2)
                    u[i, j] += noise * u_scale * randn(rng)
                end
            end
        end
        for (i, x) in enumerate(grid_global.x)
            if span[1] <= x <= span[2]
                distance = x - body_x
                envelope_x = exp(-(distance / (0.3 * sim.L))^2)
                for (j, z) in enumerate(grid_global.zw)
                    base = sinpi(z / grid_global.Lz)
                    w[i, j] += 0.25 * noise * sim.U * envelope_x * base * randn(rng)
                end
            end
        end
        w .+= 1e-3 * sim.U * randn(rng, size(w))
        sim.state_new.u .= sim.state_old.u
        sim.state_new.w .= sim.state_old.w
    end
end

function _step_impl!(sim::Simulation)
    if sim.use_mpi
        mpi_solve_step_2d!(sim.solver, sim.state_new, sim.state_old, sim.dt, sim.config.rigid_bodies)
    else
        solve_step_2d!(sim.solver, sim.state_new, sim.state_old, sim.dt, sim.config.rigid_bodies)
    end
    sim.state_old, sim.state_new = sim.state_new, sim.state_old
end

function sim_step!(sim::Simulation; verbose::Bool=false)
    _step_impl!(sim)
    sanitize!(sim.state_old)
    if verbose
        if sim.use_mpi
            comm = sim.solver.decomp.comm
            max_u_local = maximum(abs, sim.state_old.u)
            max_w_local = maximum(abs, sim.state_old.w)
            max_u = MPI.Allreduce(max_u_local, MPI.MAX, comm)
            max_w = MPI.Allreduce(max_w_local, MPI.MAX, comm)
            if MPI.Comm_rank(comm) == 0
                println("t = $(round(sim_time(sim), digits = 4)), max|u| = $(round(max_u, digits = 3)), max|w| = $(round(max_w, digits = 3))")
            end
        else
            max_u = maximum(abs, sim.state_old.u)
            max_w = maximum(abs, sim.state_old.w)
            println("t = $(round(sim_time(sim), digits = 4)), max|u| = $(round(max_u, digits = 3)), max|w| = $(round(max_w, digits = 3))")
        end
    end
    sim
end

function sim_step!(sim::Simulation, t_end::Real; verbose::Bool=false, report_interval::Real=0.0)
    next_report = sim_time(sim) + report_interval
    while sim_time(sim) < t_end - 1e-12
        _step_impl!(sim)
        sanitize!(sim.state_old)
        if verbose && report_interval > 0
            if sim_time(sim) >= next_report - 1e-10
                if sim.use_mpi
                    comm = sim.solver.decomp.comm
                    max_u = MPI.Allreduce(maximum(abs, sim.state_old.u), MPI.MAX, comm)
                    max_w = MPI.Allreduce(maximum(abs, sim.state_old.w), MPI.MAX, comm)
                    if MPI.Comm_rank(comm) == 0
                        println("t = $(round(sim_time(sim), digits = 4)), max|u| = $(round(max_u, digits = 3)), max|w| = $(round(max_w, digits = 3))")
                    end
                else
                    max_u = maximum(abs, sim.state_old.u)
                    max_w = maximum(abs, sim.state_old.w)
                    println("t = $(round(sim_time(sim), digits = 4)), max|u| = $(round(max_u, digits = 3)), max|w| = $(round(max_w, digits = 3))")
                end
                next_report += report_interval
            end
        end
        if sim_time(sim) >= t_end - 1e-12
            break
        end
    end
    sim
end

function save_snapshot(sim::Simulation; filename::AbstractString = "cylinder_shedding.jld2")
    sanitize!(sim.state_old)
    if isfile(filename)
        if sim.use_mpi
            comm = sim.solver.decomp.comm
            if MPI.Comm_rank(comm) == 0
                rm(filename; force=true)
            end
            MPI.Barrier(comm)
        else
            rm(filename; force=true)
        end
    end
    cfg = NetCDFConfig(filename;
        save_flow_field = true,
        save_body_positions = true,
        save_force_coefficients = true,
        time_interval = Inf,
        iteration_interval = typemax(Int))
    writer = JLD2Output.JLD2Writer(filename, sim.grid, cfg)
    write_solution!(writer, sim.state_old, sim.config.rigid_bodies, sim.grid,
                    sim.config.fluid, sim_time(sim), sim.state_old.step; dt = sim.dt)
    close!(writer)
    filename
end

function sanitize!(st)
    @inbounds for idx in eachindex(st.u)
        v = st.u[idx]
        if !isfinite(v)
            st.u[idx] = 0.0
        end
    end
    @inbounds for idx in eachindex(st.w)
        v = st.w[idx]
        if !isfinite(v)
            st.w[idx] = 0.0
        end
    end
    if st.p !== nothing
        @inbounds for idx in eachindex(st.p)
            v = st.p[idx]
            if !isfinite(v)
                st.p[idx] = 0.0
            end
        end
    end
    return st
end

export Simulation, sim_step!, sim_time, sim_info, perturb!, save_snapshot,
       grid, state, bodies, fluid
