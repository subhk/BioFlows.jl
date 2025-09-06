# Two Flexible Flags with Distance Control
#
# Run with:
#   julia --project examples/two_flags_controlled.jl

using BioFlows

# Create a simple 2D configuration (XZ plane)
config = create_2d_simulation_config(
    nx=64, nz=32,
    Lx=2.0, Lz=1.0,
    dt=0.002, final_time=0.1,
    output_interval=0.02,
    output_file="two_flags_controlled",
)

# Define two flag configurations
flag_configs = [
    (start_point=[0.4, 0.5], length=0.25, width=0.01,
     prescribed_motion=(type=:sinusoidal, amplitude=0.02, frequency=2.0)),
    (start_point=[0.9, 0.5], length=0.25, width=0.01,
     prescribed_motion=(type=:sinusoidal, amplitude=0.02, frequency=2.0)),
]

# Target distance (between trailing edges)
distances = [0.0 0.4; 0.4 0.0]

# Build flexible bodies and controller
bodies, controller = create_coordinated_flexible_system(
    flag_configs, distances;
    base_frequency=2.0,
    kp=0.5, ki=0.1, kd=0.05,
)

# Attach to config
config = add_flexible_bodies_with_controller!(config, bodies, controller)

# Create solver and initial state
solver = create_solver(config)
state0 = initialize_simulation(config; initial_conditions=:quiescent)

println("Running two-flags controlled example...")
final_state = run_simulation(config, solver, state0)
println("Done. Output written to NetCDF with base name: \"$(config.output_config.filename)\"")

