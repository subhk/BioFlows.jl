# Flow past a 2D cylinder using MPI domain decomposition (XZ plane)
# Run with, e.g.: mpirun -np 4 julia --project examples/flow_past_cylinder_2d_mpi.jl
#
# This example demonstrates:
# - MPI-parallel 2D flow simulation
# - Flow around a rigid circular cylinder
# - Enhanced diagnostic output showing iteration progress
# - Automatic calculation of drag/lift coefficients

using BioFlows
using MPI
using NetCDF


function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Print startup message from rank 0
    if rank == 0
        println("="^60)
        println("FLOW PAST CYLINDER - MPI 2D SIMULATION")
        println("Running on $nprocs MPI ranks")
        println("="^60)
        flush(stdout)
    end

    # Physical and numerical parameters
    # Domain geometry
    nx, nz = 144, 48           # Grid points (adjusted for good cylinder resolution)
    Lx, Lz = 6.0, 2.0         # Physical domain size [m]
    
    # Flow parameters
    Uin = 1.0                  # Inlet velocity [m/s]
    ρ = 1000.0                 # Fluid density [kg/m³]
    ν = 0.001                  # Kinematic viscosity [m²/s]
    
    # Cylinder geometry
    D = 0.2                    # Cylinder diameter [m]
    R = D/2                    # Cylinder radius [m]
    xc = 1.2                   # Cylinder center x-coordinate [m]
    zc = Lz/2                  # Cylinder center z-coordinate [m] (centerline)
    
    # Time integration
    dt = 0.002                 # Time step [s]
    Tfinal = 2.0               # Final simulation time [s]
    save_interval = 0.1        # Output saving interval [s]
    
    # Calculate Reynolds number for reference
    Re = Uin * D / ν
    if rank == 0
        println("Physical parameters:")
        println("  Reynolds number: Re = U*D/ν = $(round(Re, digits=1))")
        println("  Grid resolution: $(nx)×$(nz) = $(nx*nz) cells")
        println("  Domain size: $(Lx)×$(Lz) m")
        println("  Cylinder: D=$(D) m at ($(xc), $(zc))")
        println("  Time: dt=$(dt) s, T_final=$(Tfinal) s")
        println()
        flush(stdout)
    end

    # Build simulation configuration
    if rank == 0
        println("Setting up simulation configuration...")
        flush(stdout)
    end
    
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

    # Add rigid cylinder obstacle
    if rank == 0
        println("Adding rigid cylinder: center=($(xc), $(zc)), radius=$(R)")
        flush(stdout)
    end
    config = BioFlows.add_rigid_circle!(config, [xc, zc], R)
    bodies = config.rigid_bodies  # keep a handle for output

    # Start the MPI parallel simulation
    if rank == 0
        println()
        println("Starting MPI 2D simulation...")
        println("Expected output: cylinder2d_mpi.nc and cylinder2d_mpi_coeffs.nc")
        println("="^60)
        flush(stdout)
    end
    
    # Run the simulation (includes enhanced diagnostics)
    BioFlows.run_simulation_mpi_2d(config)

    # Simulation complete - analyze results
    if rank == 0
        println()
        println("="^60)
        println("SIMULATION COMPLETE")
        println("="^60)
        
        # Read and display final drag/lift coefficients
        try
            coeff_path = "cylinder2d_mpi_coeffs.nc"
            if isfile(coeff_path)
                nc = NetCDF.open(coeff_path)
                time = NetCDF.readvar(nc, "time")
                Cd = NetCDF.readvar(nc, "Cd")
                Cl = NetCDF.readvar(nc, "Cl")
                NetCDF.close(nc)
                
                nt = length(time)
                Cd_last = Cd[1, nt]
                Cl_last = Cl[1, nt]
                
                println("Results summary:")
                println("  Final time: t = $(round(time[end], digits=3)) s")
                println("  Total time steps: $(nt)")
                println("  Final drag coefficient: Cd = $(round(Cd_last, digits=4))")
                println("  Final lift coefficient: Cl = $(round(Cl_last, digits=4))")
                println()
                println("Output files created:")
                println("  Flow field: cylinder2d_mpi.nc")
                println("  Force coefficients: $(coeff_path)")
                
                # Basic flow regime assessment
                if Re < 40
                    flow_regime = "steady"
                elseif Re < 150
                    flow_regime = "periodic vortex shedding"  
                else
                    flow_regime = "turbulent"
                end
                println("  Expected flow regime (Re=$(round(Re, digits=1))): $(flow_regime)")
                
            else
                @warn "Coefficient file not found: $(coeff_path)"
                println("Flow field output: cylinder2d_mpi.nc")
            end
        catch e
            @warn "Could not read coefficient data: $e"
        end
        
        println("="^60)
        flush(stdout)
    end
end

main()
