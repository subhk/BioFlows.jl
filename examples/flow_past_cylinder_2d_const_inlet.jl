# Flow past a 2D cylinder with constant inlet velocity,
# pressure outlet, and no-slip top/bottom (XZ plane)
# Run with: julia --project examples/flow_past_cylinder_2d_const_inlet.jl

using BioFlows

function main()
    # Domain and flow parameters
    nx, nz = 240, 60          # grid resolution (increase for accuracy)
    Lx, Lz = 8.0, 2.0         # domain size (x length, z height)
    Uin = 1.0                 # inlet velocity (m/s)
    D = 0.2                   # cylinder diameter (m)
    R = D / 2                 # cylinder radius
    ρ = 1000.0                # fluid density (kg/m^3)
    ν = 0.001                 # kinematic viscosity (m^2/s); yields Re = Uin*D/ν = 200
    dt = 0.002                # time step (s)
    Tfinal = 10.0             # final time (s)

    # Create 2D XZ-plane configuration
    config = create_2d_simulation_config(
        nx = nx, nz = nz,
        Lx = Lx, Lz = Lz,
        density_value = ρ,
        nu = ν,                       # specify kinematic viscosity
        inlet_velocity = Uin,
        outlet_type = :pressure,     # constant pressure outlet (p = 0)
        wall_type = :no_slip,        # top/bottom walls
        dt = dt,
        final_time = Tfinal,
        adaptive_refinement = false, # set true to enable AMR
        output_interval = 0.1,
        output_file = "cylinder2d_const_inlet",
        # NetCDF rollover: cap snapshots per file to trigger _1.nc, _2.nc
        output_max_snapshots = 50,
        output_save_mode = :time_interval
    )

    # Add a rigid circular cylinder centered vertically, upstream of mid-domain
    xc = 0.6
    zc = Lz / 2
    config = add_rigid_circle!(config, [xc, zc], R)

    # Create solver and initial state (uniform flow optional)
    solver = create_solver(config)
    state0 = initialize_simulation(config, initial_conditions = :uniform_flow)

    # Run simulation (NetCDF -> cylinder2d_const_inlet.nc)
    run_simulation(config, solver, state0)
end

main()
