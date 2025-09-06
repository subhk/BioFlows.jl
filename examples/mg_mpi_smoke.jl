# Multigrid + MPI smoke test (runs with 1 or more ranks)
#
# Run (single rank):
#   julia --project examples/mg_mpi_smoke.jl
# Run (multiple ranks):
#   mpiexec -n 2 julia --project examples/mg_mpi_smoke.jl

using MPI
using BioFlows

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    if rank == 0
        println("MPI ranks: $size")
    end

    # Global problem and local grid via MPI decomposition
    nxg, nzg = 64, 32
    Lx, Lz = 1.0, 0.5
    decomp = BioFlows.MPI2DDecomposition(nxg, nzg, comm)
    # Create an interior-only local grid (no ghosts) for multigrid
    dx = Lx / nxg
    dz = Lz / nzg
    x_min = (decomp.i_start - 1) * dx
    z_min = (decomp.j_start - 1) * dz
    grid = BioFlows.StaggeredGrid2D(decomp.nx_local, decomp.nz_local,
                                    decomp.nx_local * dx, decomp.nz_local * dz;
                                    origin_x=x_min, origin_z=z_min)

    # Build a simple Neumann BC set for pressure
    bc = BioFlows.BoundaryConditions2D()

    # Allocate local phi and rhs and set a simple RHS pattern
    phi = zeros(Float64, decomp.nx_local, decomp.nz_local)
    rhs = zeros(Float64, decomp.nx_local, decomp.nz_local)

    # Create local multigrid solver and solve
    mg = BioFlows.MultigridPoissonSolver(grid)
    BioFlows.solve_poisson!(mg, phi, rhs, grid, bc)

    # Global norm check
    local_norm = sum(phi.^2)
    global_norm = MPI.Allreduce(local_norm, MPI.SUM, comm)
    if rank == 0
        println("Global pressure L2^2: ", global_norm)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
