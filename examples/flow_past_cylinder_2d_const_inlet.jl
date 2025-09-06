# Flow past a 2D cylinder with constant inlet velocity,
# pressure outlet, and no-slip top/bottom (XZ plane)
# Run:
#   julia --project examples/flow_past_cylinder_2d_const_inlet.jl

using BioFlows
using NetCDF

function main()
    # Fixed parameters for a simple, ready-to-run setup
    nx, nz = 240, 60
    Lx, Lz = 8.0, 2.0
    Uin = 1.0
    D = 0.2; R = D/2
    ρ = 1000.0; ν = 0.001
    dt = 0.002; Tfinal = 10.0
    saveint = 0.1
    outfile = "cylinder2d_const_inlet"
    xc, zc = 0.6, 1.0
    maxsnaps = 50
    use_amr = false

    config = create_2d_simulation_config(
        nx = nx, nz = nz,
        Lx = Lx, Lz = Lz,
        density_value = ρ,
        nu = ν,
        inlet_velocity = Uin,
        outlet_type = :pressure,
        wall_type = :no_slip,
        dt = dt,
        final_time = Tfinal,
        adaptive_refinement = use_amr,
        output_interval = saveint,
        output_file = outfile,
        output_max_snapshots = maxsnaps,
        output_save_mode = :time_interval
    )

    # Add a rigid circular cylinder centered vertically, upstream of mid-domain
    config = add_rigid_circle!(config, [xc, zc], R)

    # Create solver and initial state (uniform flow optional)
    solver = create_solver(config)
    state0 = initialize_simulation(config, initial_conditions = :uniform_flow)

    # Run simulation (NetCDF -> outfile.nc)
    run_simulation(config, solver, state0)

    # Annotate first NetCDF with cylinder metadata for plotting convenience
    try
        ncpath = string(outfile, ".nc")
        if isfile(ncpath)
            nc = NetCDF.open(ncpath, "c")
            NetCDF.putatt(nc, "global", Dict(
                "cylinder_x"=>xc, "cylinder_z"=>zc, "cylinder_radius"=>R,
                "domain_Lx"=>Lx, "domain_Lz"=>Lz,
                "inlet_velocity"=>Uin, "rho"=>ρ, "nu"=>ν
            ))
            NetCDF.close(nc)
        end
    catch e
        @warn "Could not write cylinder metadata to NetCDF: $e"
    end

    # Read and print final Cd/Cl from the coefficients file (if created)
    try
        coeff_path = string(outfile, "_coeffs.nc")
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

main()
