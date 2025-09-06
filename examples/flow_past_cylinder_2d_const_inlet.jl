# Flow past a 2D cylinder with constant inlet velocity,
# pressure outlet, and no-slip top/bottom (XZ plane)
# Run:
#   julia --project examples/flow_past_cylinder_2d_const_inlet.jl \
#     --nx 240 --nz 60 --Lx 8.0 --Lz 2.0 --uin 1.0 --D 0.2 \
#     --rho 1000.0 --nu 0.001 --dt 0.002 --tfinal 10.0 --save 0.1 \
#     --outfile cylinder2d_const_inlet --xc 0.6 --zc 1.0 --maxsnaps 50 --amr false

using BioFlows
using NetCDF

function parse_args()
    p = Dict{String,Any}(
        "nx"=>240, "nz"=>60, 
        "Lx"=>8.0, "Lz"=>2.0,
        "uin"=>1.0, "D"=>0.2, 
        "rho"=>1000.0, "nu"=>0.001,
        "dt"=>0.002, 
        "tfinal"=>10.0, "save"=>0.1,
        "outfile"=>"cylinder2d_const_inlet",
        "xc"=>0.6, "zc"=>1.0, "maxsnaps"=>50,
        "amr"=>false
    )
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if startswith(arg, "--") && i < length(ARGS)
            key = lowercase(arg[3:end])
            val = ARGS[i+1]
            if key in ("nx","nz","maxsnaps")
                p[key] = parse(Int, val)
            elseif key == "amr"
                p[key] = lowercase(val) in ("1","true","yes")
            elseif key in ("outfile")
                p[key] = val
            elseif key in ("Lx","Lz","uin","D","rho","nu","dt","tfinal","save","xc","zc")
                p[key] = parse(Float64, val)
            end
            i += 2
        else
            i += 1
        end
    end
    return p
end

function main()
    p = parse_args()
    nx = p["nx"]; nz = p["nz"]
    Lx = p["Lx"]; Lz = p["Lz"]
    Uin = p["uin"]
    D = p["D"]; R = D/2
    ρ = p["rho"]; ν = p["nu"]
    dt = p["dt"]; Tfinal = p["tfinal"]
    saveint = p["save"]
    outfile = String(p["outfile"])  # base name without .nc
    xc = p["xc"]; zc = p["zc"]
    maxsnaps = p["maxsnaps"]
    use_amr = p["amr"]

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
end

main()
