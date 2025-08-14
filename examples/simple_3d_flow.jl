"""
Example: Simple 3D flow simulations

This example demonstrates the usage of BioFlows.jl for simulating
3D incompressible flows with various configurations.
"""

using BioFlows

# Simple 3D channel flow
function simple_3d_channel_flow()
    println("Running simple 3D channel flow simulation...")
    
    # Run simulation with high-level API
    final_state = run_bioflow_3d(
        nx = 32,
        ny = 16,
        nz = 16,
        Lx = 4.0,
        Ly = 1.0,
        Lz = 1.0,
        Reynolds = 100.0,
        inlet_velocity = 1.0,
        outlet_type = :pressure,
        wall_type = :no_slip,
        time_scheme = :rk3,
        dt = 0.01,
        final_time = 5.0,
        output_interval = 0.2,
        output_file = "channel_flow_3d"
    )
    
    println("3D channel flow simulation completed!")
    return final_state
end

# 3D flow around a sphere
function flow_around_sphere()
    println("Running 3D flow around sphere...")
    
    # Create simulation configuration
    config = create_3d_simulation_config(
        nx = 48,
        ny = 32,
        nz = 32,
        Lx = 6.0,
        Ly = 2.0,
        Lz = 2.0,
        Reynolds = 200.0,
        inlet_velocity = 1.0,
        dt = 0.005,
        final_time = 8.0,
        output_interval = 0.2,
        adaptive_refinement = true,
        output_file = "sphere_flow_3d"
    )
    
    # Add spherical body (using circle shape for 3D)
    add_rigid_circle!(config, [2.0, 1.0, 1.0], 0.3)
    
    # Create solver and run simulation
    solver = create_solver(config)
    initial_state = initialize_simulation(config, initial_conditions=:uniform_flow)
    final_state = run_simulation(config, solver, initial_state)
    
    println("3D sphere flow simulation completed!")
    return final_state
end

# 3D flow with multiple bodies
function multiple_bodies_3d()
    println("Running 3D flow with multiple rigid bodies...")
    
    # Create simulation configuration  
    config = create_3d_simulation_config(
        nx = 64,
        ny = 32,
        nz = 32,
        Lx = 8.0,
        Ly = 2.0,
        Lz = 2.0,
        Reynolds = 150.0,
        inlet_velocity = 1.0,
        dt = 0.008,
        final_time = 6.0,
        output_interval = 0.15,
        adaptive_refinement = true,
        output_file = "multi_body_3d"
    )
    
    # Add multiple bodies at different positions
    add_rigid_circle!(config, [2.0, 0.7, 1.0], 0.2)  # Sphere 1
    add_rigid_circle!(config, [3.5, 1.3, 1.0], 0.15) # Sphere 2
    add_rigid_square!(config, [5.0, 1.0, 0.6], 0.3)  # Cubic body
    
    # Create solver and run simulation
    solver = create_solver(config)
    initial_state = initialize_simulation(config, initial_conditions=:uniform_flow)
    final_state = run_simulation(config, solver, initial_state)
    
    println("3D multiple bodies simulation completed!")
    return final_state
end

# 3D flow with MPI parallelization (example)
function parallel_3d_flow()
    println("Running parallel 3D flow simulation...")
    
    # Create simulation configuration with MPI enabled
    config = create_3d_simulation_config(
        nx = 64,
        ny = 48,
        nz = 48,
        Lx = 6.0,
        Ly = 3.0,
        Lz = 3.0,
        Reynolds = 200.0,
        inlet_velocity = 1.0,
        dt = 0.005,
        final_time = 4.0,
        output_interval = 0.1,
        use_mpi = true,  # Enable MPI parallelization
        output_file = "parallel_3d"
    )
    
    # Add a body for more interesting flow
    add_rigid_circle!(config, [2.5, 1.5, 1.5], 0.4)
    
    # Create solver and run simulation
    solver = create_solver(config)
    initial_state = initialize_simulation(config, initial_conditions=:uniform_flow)
    final_state = run_simulation(config, solver, initial_state)
    
    println("Parallel 3D flow simulation completed!")
    return final_state
end

# Run examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("BioFlows.jl 3D Flow Examples")
    println("="^50)
    
    # Example 1: Simple 3D channel flow
    simple_3d_channel_flow()
    
    # Example 2: Flow around sphere
    flow_around_sphere()
    
    # Example 3: Multiple bodies
    multiple_bodies_3d()
    
    # Example 4: Parallel simulation (uncomment to run with MPI)
    # Note: Run with: mpirun -np 4 julia simple_3d_flow.jl
    # parallel_3d_flow()
    
    println("\n3D examples completed successfully!")
end