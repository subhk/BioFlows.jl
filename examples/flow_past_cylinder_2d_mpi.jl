# Flow past a 2D cylinder using MPI domain decomposition (XZ plane)
# Run with, e.g.: mpirun -np 4 julia --project examples/flow_past_cylinder_2d_mpi.jl

using BioFlows
using MPI

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Domain and flow parameters
    nx, nz = 192, 96          # global grid resolution
    Lx, Lz = 6.0, 2.0         # domain size
    Uin = 1.0                 # inlet velocity (m/s)
    D = 0.2                   # cylinder diameter
    R = D / 2
    ρ = 1000.0                # density (kg/m^3)
    ν = 0.001                 # kinematic viscosity (m^2/s)
    dt = 0.002                # time step
    Tfinal = 2.0              # shorter run for demo
    save_interval = 0.05      # output/print interval

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
    xc = 1.2
    zc = Lz / 2
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

    # Root initializes a global NetCDF writer on the actual global grid
    writer = nothing
    if rank == 0
        println("MPI run with $nprocs ranks: 2D cylinder, $(nx)x$(nz), dt=$dt, T=$Tfinal")
        global_grid = BioFlows.StaggeredGrid2D(nx, nz, Lx, Lz)
        writer = BioFlows.NetCDFWriter("cylinder2d_mpi.nc", global_grid,
            BioFlows.NetCDFConfig("cylinder2d_mpi";
                max_snapshots_per_file=50,
                save_mode=:time_interval,
                time_interval=save_interval,
                iteration_interval=999999,
                save_flow_field=true,
                save_body_positions=false,
                save_force_coefficients=false))
    end

    # Helper to gather distributed fields to rank 0 and write
    function gather_and_write!(writer, solver, local_state, t, step)
        decomp = solver.decomp
        comm = decomp.comm
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)

        # Local interior ranges
        i_start, i_end = decomp.i_start, decomp.i_end
        j_start, j_end = decomp.j_start, decomp.j_end
        ils, ile = decomp.i_local_start, decomp.i_local_end
        jls, jle = decomp.j_local_start, decomp.j_local_end

        # Local blocks (interior + one extra face on domain boundaries)
        u_i_hi = ile + (i_end == decomp.nx_global ? 1 : 0)
        v_j_hi = jle + (j_end == decomp.nz_global ? 1 : 0)
        u_blk = @view local_state.u[ils:u_i_hi, jls:jle]
        v_blk = @view local_state.v[ils:ile, jls:v_j_hi]
        p_blk = @view local_state.p[ils:ile, jls:jle]

        # Root allocates global arrays
        if rank == 0
            u_glob = zeros(size(writer.grid.xu, 1), size(writer.grid.z, 1))  # (nx+1, nz)
            v_glob = zeros(writer.grid.nx, writer.grid.nz + 1)               # (nx, nz+1)
            p_glob = zeros(writer.grid.nx, writer.grid.nz)                   # (nx, nz)

            # Place rank 0 block
            u_i_lo_gl = i_start
            u_i_hi_gl = i_end + (i_end == decomp.nx_global ? 1 : 0)
            v_j_lo_gl = j_start
            v_j_hi_gl = j_end + (j_end == decomp.nz_global ? 1 : 0)
            u_glob[u_i_lo_gl:u_i_hi_gl, j_start:j_end] .= u_blk
            v_glob[i_start:i_end, v_j_lo_gl:v_j_hi_gl] .= v_blk
            p_glob[i_start:i_end, j_start:j_end] .= p_blk

            # Receive from other ranks
            for src in 1:size-1
                hdr = Array{Int}(undef, 4)  # i_start, i_end, j_start, j_end
                MPI.Recv!(hdr, src, 100, comm)
                is, ie, js, je = hdr...
                # u dims and v dims
                u_count_i = ie - is + 1 + (ie == decomp.nx_global ? 1 : 0)
                v_count_j = je - js + 1 + (je == decomp.nz_global ? 1 : 0)
                u_recv = Array{Float64}(undef, u_count_i, je - js + 1)
                v_recv = Array{Float64}(undef, ie - is + 1, v_count_j)
                p_recv = Array{Float64}(undef, ie - is + 1, je - js + 1)
                MPI.Recv!(u_recv, src, 101, comm)
                MPI.Recv!(v_recv, src, 102, comm)
                MPI.Recv!(p_recv, src, 103, comm)

                u_i_lo = is
                u_i_hi = ie + (ie == decomp.nx_global ? 1 : 0)
                v_j_lo = js
                v_j_hi = je + (je == decomp.nz_global ? 1 : 0)
                u_glob[u_i_lo:u_i_hi, js:je] .= u_recv
                v_glob[is:ie, v_j_lo:v_j_hi] .= v_recv
                p_glob[is:ie, js:je] .= p_recv
            end

            # Write snapshot
            gstate = BioFlows.SolutionState2D(writer.grid.nx, writer.grid.nz)
            gstate.u .= u_glob
            gstate.v .= v_glob
            gstate.p .= p_glob
            gstate.t = t
            gstate.step = step
            BioFlows.save_snapshot!(writer, gstate, t, step)

        else
            # Non-root: send header and data
            hdr = Int[i_start, i_end, j_start, j_end]
            MPI.Send(hdr, 0, 100, comm)
            MPI.Send(Array(u_blk), 0, 101, comm)
            MPI.Send(Array(v_blk), 0, 102, comm)
            MPI.Send(Array(p_blk), 0, 103, comm)
        end
    end

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
            # Global NetCDF output (on time interval)
            gather_and_write!(writer, solver, local_old, t, step)
            next_print += save_interval
        end
    end

    if rank == 0
        println("MPI simulation complete.")
        println("Note: This demo does not write NetCDF. Each rank holds local arrays.")
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
