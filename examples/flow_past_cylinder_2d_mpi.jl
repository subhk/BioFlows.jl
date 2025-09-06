# Flow past a 2D cylinder using MPI domain decomposition (XZ plane)
# Run with, e.g.: mpirun -np 4 julia --project examples/flow_past_cylinder_2d_mpi.jl

using BioFlows
using MPI
using NetCDF


function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Fixed domain and flow parameters
    nx, nz = 192, 96
    Lx, Lz = 6.0, 2.0
    Uin = 1.0
    D = 0.2; R = D/2
    ρ = 1000.0; ν = 0.001
    dt = 0.002
    Tfinal = 2.0
    save_interval = 0.05
    outfile = "cylinder2d_mpi.nc"

    # Build config (uses API to set BCs and fluid)
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
    zc = Lz/2
    config = BioFlows.add_rigid_circle!(config, [xc, zc], R)
    bodies = config.rigid_bodies  # keep a handle for output

    # Run using built-in MPI loop
    BioFlows.run_simulation_mpi_2d(config)

    if rank == 0
        # Read and print final Cd/Cl from the coefficients file (if created)
        try
            coeff_path = outfile |> x->replace(x, ".nc"=>"_coeffs.nc")
            if isfile(coeff_path)
                nc = NetCDF.open(coeff_path)
                time = NetCDF.readvar(nc, "time")
                Cd = NetCDF.readvar(nc, "Cd")
                Cl = NetCDF.readvar(nc, "Cl")
                NetCDF.close(nc)
                nt = length(time)
                Cd_last = Cd[1, nt]
                Cl_last = Cl[1, nt]
                println("Final coefficients (body 1): Cd=$(round(Cd_last, digits=4)), Cl=$(round(Cl_last, digits=4)) at t=$(round(time[end], digits=3))")
                println("Coefficient series saved to: $(coeff_path)")
            else
                @warn "Coefficient file not found: $(coeff_path)"
            end
        catch e
            @warn "Could not read Cd/Cl series: $e"
        end
    end
end

main()
