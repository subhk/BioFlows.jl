"""
Example: Simple 2D channel flow simulation

This example demonstrates the basic usage of BioFlow.jl for simulating
2D incompressible flow in a channel.
"""

using BioFlow

# Simple 2D channel flow - no bodies
function simple_channel_flow()
    println("Running simple 2D channel flow simulation...")
    
    # Run simulation with high-level API
    final_state = run_bioflow_2d(
        nx = 64,
        ny = 32,
        Lx = 4.0,
        Ly = 1.0,
        Reynolds = 100.0,
        inlet_velocity = 1.0,
        outlet_type = :pressure,
        wall_type = :no_slip,
        time_scheme = :rk3,
        dt = 0.01,
        final_time = 5.0,
        output_interval = 0.1,
        output_file = "channel_flow_2d"
    )
    
    println("Simulation completed!")
    return final_state
end

# 2D flow with a rigid circular cylinder
function flow_around_cylinder()
    println("Running 2D flow around circular cylinder...")
    
    # Create simulation configuration
    config = create_2d_simulation_config(
        nx = 128,
        ny = 64,
        Lx = 6.0,
        Ly = 2.0,
        Reynolds = 200.0,
        inlet_velocity = 1.0,
        dt = 0.005,
        final_time = 10.0,
        output_interval = 0.1,
        adaptive_refinement = true,
        output_file = "cylinder_flow_2d"
    )
    
    # Add circular cylinder
    add_rigid_circle!(config, [2.0, 1.0], 0.2)
    
    # Create solver and run simulation
    solver = create_solver(config)
    initial_state = initialize_simulation(config, initial_conditions=:uniform_flow)
    final_state = run_simulation(config, solver, initial_state)
    
    println("Simulation completed!")
    return final_state
end

# 2D flow with flexible body (fish-like swimming)
function flexible_body_flow()
    println("Running 2D flow with flexible body...")
    
    # Create simulation configuration
    config = create_2d_simulation_config(
        nx = 128,
        ny = 64,
        Lx = 6.0,
        Ly = 2.0,
        Reynolds = 100.0,
        inlet_velocity = 0.5,
        dt = 0.002,
        final_time = 5.0,
        output_interval = 0.05,
        output_file = "flexible_body_2d"
    )
    
    # Add flexible body with sinusoidal motion at the front
    add_flexible_body!(config, 
        [1.5, 1.0],          # front position
        1.0,                  # length
        20,                   # number of points
        thickness = 0.05,
        rigidity = 10.0,
        front_constraint = :sinusoidal,
        motion_amplitude = 0.2,
        motion_frequency = 2.0
    )
    
    # Create solver and run simulation
    solver = create_solver(config)
    initial_state = initialize_simulation(config)
    final_state = run_simulation(config, solver, initial_state)
    
    println("Simulation completed!")
    return final_state
end

# Run examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("BioFlow.jl 2D Flow Examples")
    println("="^50)
    
    # Example 1: Simple channel flow
    simple_channel_flow()
    
    # Example 2: Flow around cylinder
    flow_around_cylinder()
    
    # Example 3: Flexible body flow
    flexible_body_flow()
    
    println("\nAll examples completed successfully!")
end