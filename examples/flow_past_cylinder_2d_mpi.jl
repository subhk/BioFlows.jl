# Flow past a 2D cylinder using MPI domain decomposition (XZ plane)
# Run with, e.g.: mpirun -np 4 julia --project examples/flow_past_cylinder_2d_mpi.jl

using BioFlows
using MPI
using NetCDF

# Simple CLI parser for core parameters
function parse_args()
    params = Dict{String,Any}(
        "nx"=>192, "nz"=>96, 
        "Lx"=>6.0, "Lz"=>2.0, 
        "uin"=>1.0,
        "D"=>0.2, 
        "rho"=>1000.0, 
        "nu"=>0.001, 
        "dt"=>0.002,
        "tfinal"=>2.0, 
        "save"=>0.05, 
        "outfile"=>"cylinder2d_mpi.nc",
        "xc"=>1.2, 
        "zc"=>1.0
    )
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if startswith(arg, "--") && i < length(ARGS)
            key = lowercase(arg[3:end])
            val = ARGS[i+1]
            if key in ("nx","nz")
                params[key] = parse(Int, val)
            elseif key in ("outfile")
                params[key] = val
            elseif key in ("xc", "zc", "Lx", "Lz", "uin", "D", "rho", "nu", "dt", "tfinal", "save")
                params[key] = parse(Float64, val)
            end
            i += 2
        else
            i += 1
        end
    end
    return params
end

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Domain and flow parameters (configurable via CLI)
    p = parse_args()
    nx = p["nx"]; nz = p["nz"]
    Lx = p["Lx"]; Lz = p["Lz"]
    Uin = p["uin"]
    D = p["D"]; R = D/2
    ρ = p["rho"]; ν = p["nu"]
    dt = p["dt"]
    Tfinal = p["tfinal"]
    save_interval = p["save"]
    outfile = String(p["outfile"])

    # Build config for convenience (uses our API to set BCs and fluid)
    config = BioFlows.create_2d_simulation_config(
        nx=nx, nz=nz,
        Lx=Lx, Lz=Lz,
        density_value=ρ,
        nu=ν,
        inlet_velocity=Uin,
        outlet_type=:pressure,
        wall_type=:no_slip,
        dt=dt, final_time=Tfinal,
        use_mpi=true,
        adaptive_refinement=false,
        output_interval=save_interval,
        output_file="cylinder2d_mpi"
    )

    # Add cylinder (global coordinates)
    xc = haskey(p, "xc") ? p["xc"] : 1.2
    zc = haskey(p, "zc") ? p["zc"] : Lz/2
    config = BioFlows.add_rigid_circle!(config, [xc, zc], R)

    # Create MPI solver directly from config (returns MPINavierStokesSolver2D)
    solver = BioFlows.create_solver(config)

    # Local solution states (with ghost cells)
    local_old = BioFlows.MPISolutionState2D(solver.decomp)
    local_new = BioFlows.MPISolutionState2D(solver.decomp)

    # Simple uniform-flow initial condition for u
    local_old.u .= Uin
    local_old.v .= 0.0
    local_old.p .= 0.0
    local_old.t = 0.0
    local_old.step = 0

    t = 0.0
    step = 0
    next_print = save_interval

    # Global grid spacings for CFL estimate
    dx = Lx / nx
    dz = Lz / nz

    # Initialize a global NetCDF writer on all ranks (file is created on root)
    if rank == 0
        println("MPI run with $nprocs ranks: 2D cylinder, $(nx)x$(nz), dt=$dt, T=$Tfinal")
    end
    global_grid = BioFlows.StaggeredGrid2D(nx, nz, Lx, Lz)
    writer = BioFlows.NetCDFWriter(outfile, global_grid,
        BioFlows.NetCDFConfig("cylinder2d_mpi";
            max_snapshots_per_file=50,
            save_mode=:time_interval,
            time_interval=save_interval,
            iteration_interval=999999,
            save_flow_field=true,
            save_body_positions=false,
            save_force_coefficients=false))
    if rank == 0
        BioFlows.initialize_netcdf_file!(writer)
        NetCDF.putatt(writer.ncfile, "global", Dict(
            "cylinder_x"=>xc, "cylinder_z"=>zc, "cylinder_radius"=>R,
            "domain_Lx"=>Lx, "domain_Lz"=>Lz,
            "inlet_velocity"=>Uin, "rho"=>ρ, "nu"=>ν
        ))
    end

    # (MPI writer now auto-gathers inside write_solution!)

    while t < Tfinal - 1e-12
        step += 1
        dt_step = min(dt, Tfinal - t)

        # One MPI time step
        BioFlows.mpi_solve_step_2d!(solver, local_new, local_old, dt_step)

        # Swap
        local_old, local_new = local_new, local_old
        t += dt_step

        # Periodic progress + CFL estimate (global max) and output
        if t + 1e-12 >= next_print || step % 100 == 0
            # Local maxima (exclude ghosts loosely by taking interior slice if available)
            maxu_loc = maximum(abs, local_old.u)
            maxv_loc = maximum(abs, local_old.v)
            maxu = MPI.Allreduce(maxu_loc, MPI.MAX, comm)
            maxv = MPI.Allreduce(maxv_loc, MPI.MAX, comm)
            cfl = max(maxu * dt_step / dx, maxv * dt_step / dz)
            if rank == 0
                @info "step=$step t=$(round(t, digits=3)) dt=$(round(dt_step, digits=4)) CFL=$(round(cfl, digits=3))"
            end
            # Global NetCDF output via writer (auto-detects MPI state)
            BioFlows.write_solution!(writer, local_old, nothing, global_grid, solver.fluid, t, step)
            next_print += save_interval
        end
    end

    if rank == 0
        println("MPI simulation complete. Global output written to $(outfile)")
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
