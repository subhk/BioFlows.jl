# MPI 2D conservation test: checks divergence reduction after projection
# Run with two ranks:
#   mpiexec -n 2 julia --project examples/mpi_conservation_test.jl

using MPI
using BioFlows

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    rank == 0 && println("MPI ranks: $size")

    # Global problem
    nxg, nzg = 64, 32
    Lx, Lz = 1.0, 0.5
    fluid = BioFlows.FluidProperties(0.01, BioFlows.ConstantDensity(1.0), 100.0)
    bc = BioFlows.BoundaryConditions2D()
    ts = BioFlows.RungeKutta3()

    # Build MPI solver
    solver = BioFlows.MPINavierStokesSolver2D(nxg, nzg, Lx, Lz, fluid, bc, ts; comm=comm)
    decomp = solver.decomp
    # Local states with ghost cells
    s_old = BioFlows.MPISolutionState2D(decomp)
    s_new = BioFlows.MPISolutionState2D(decomp)

    # Initialize a simple velocity field with a small perturbation
    s_old.u .= 0.0
    s_old.v .= 0.0
    ng = decomp.n_ghost
    nxg = decomp.nx_local_with_ghosts
    nzg = decomp.nz_local_with_ghosts
    if (nxg - 2ng) >= 4 && (nzg - 2ng) >= 4
        s_old.u[ng+2:nxg-ng-1, ng+2:nzg-ng-1] .= 0.1
    end

    # Compute global sum of divergence before
    div_before_local = similar(solver.local_rhs_p)
    BioFlows.exchange_ghost_cells_staggered_u_2d!(decomp, s_old.u)
    BioFlows.exchange_ghost_cells_staggered_v_2d!(decomp, s_old.v)
    BioFlows.divergence_2d!(div_before_local, s_old.u, s_old.v, solver.local_grid)
    sum_before = MPI.Allreduce(sum(div_before_local), MPI.SUM, comm)

    # One distributed step
    BioFlows.mpi_solve_step_2d!(solver, s_new, s_old, 1e-3)

    # Compute global sum of divergence after
    div_after_local = similar(solver.local_rhs_p)
    BioFlows.divergence_2d!(div_after_local, s_new.u, s_new.v, solver.local_grid)
    sum_after = MPI.Allreduce(sum(div_after_local), MPI.SUM, comm)

    if rank == 0
        println("Global divergence sum before: ", sum_before)
        println("Global divergence sum after:  ", sum_after)
        println("Reduction factor (|after|/|before|): ", abs(sum_after) / max(abs(sum_before), 1e-16))
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
