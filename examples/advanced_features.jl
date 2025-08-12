"""
Example: Advanced BioFlow.jl features

This example demonstrates advanced features including:
- Custom grid generation
- Adaptive mesh refinement
- Multiple flexible bodies
- Complex boundary conditions
"""

using BioFlow

# Example with custom stretched grid
function custom_grid_example()
    println("Running simulation with custom stretched grid...")
    
    # Create custom grid points with refinement near inlet and body region
    x_points = vcat(
        LinRange(0.0, 1.0, 20),     # Inlet region
        LinRange(1.0, 3.0, 60),     # Body region (fine)
        LinRange(3.0, 8.0, 40)      # Outlet region
    )
    
    y_points = vcat(
        LinRange(0.0, 0.5, 16),     # Bottom wall (fine)
        LinRange(0.5, 1.5, 32),     # Middle region
        LinRange(1.5, 2.0, 16)      # Top wall (fine)
    )
    
    # Create stretched grid
    grid = create_stretched_2d_grid(x_points, y_points)
    
    # Set up simulation with custom grid
    config = create_2d_simulation_config(
        nx = length(x_points) - 1,
        ny = length(y_points) - 1,
        Lx = 8.0,
        Ly = 2.0,
        Reynolds = 200.0,
        dt = 0.005,
        final_time = 5.0,
        output_file = "custom_grid_2d"
    )
    
    # Add bodies
    add_rigid_circle!(config, [2.0, 1.0], 0.25)
    
    # Use the custom grid approach (would need solver modification)
    solver = create_solver(config)  # This uses default grid, would need custom grid support
    initial_state = initialize_simulation(config)
    final_state = run_simulation(config, solver, initial_state)
    
    println("Custom grid simulation completed!")
    return final_state
end

# Example with aggressive adaptive refinement
function adaptive_refinement_example()
    println("Running simulation with aggressive adaptive refinement...")
    
    # Create configuration with custom refinement criteria
    config = create_2d_simulation_config(
        nx = 64,
        ny = 32,
        Lx = 6.0,
        Ly = 2.0,
        Reynolds = 300.0,
        dt = 0.002,
        final_time = 8.0,
        output_interval = 0.1,
        adaptive_refinement = true,
        output_file = "adaptive_refined_2d"
    )
    
    # Customize refinement criteria
    config.refinement_criteria = AdaptiveRefinementCriteria(
        velocity_gradient_threshold = 0.5,  # More sensitive
        pressure_gradient_threshold = 5.0,
        vorticity_threshold = 2.0,          # More sensitive
        body_distance_threshold = 0.3,      # Larger refinement zone
        max_refinement_level = 4,           # Deeper refinement
        min_grid_size = 0.005
    )
    
    # Add multiple bodies for complex flow
    add_rigid_circle!(config, [1.5, 1.0], 0.2)
    add_rigid_square!(config, [3.0, 0.7], 0.25)
    add_rigid_circle!(config, [4.5, 1.3], 0.15)
    
    solver = create_solver(config)
    initial_state = initialize_simulation(config, initial_conditions=:uniform_flow)
    final_state = run_simulation(config, solver, initial_state)
    
    println("Adaptive refinement simulation completed!")
    return final_state
end

# Example with multiple flexible bodies
function multiple_flexible_bodies()
    println("Running simulation with multiple flexible bodies...")
    
    config = create_2d_simulation_config(
        nx = 128,
        ny = 64,
        Lx = 8.0,
        Ly = 3.0,
        Reynolds = 150.0,
        dt = 0.001,
        final_time = 6.0,
        output_interval = 0.05,
        output_file = "multi_flexible_2d"
    )
    
    # Add multiple flexible bodies with different behaviors
    
    # Body 1: Sinusoidal motion (fish-like swimming)
    add_flexible_body!(config, 
        [1.0, 1.0], 1.2, 25,
        thickness = 0.04,
        rigidity = 8.0,
        front_constraint = :sinusoidal,
        motion_amplitude = 0.15,
        motion_frequency = 1.5
    )
    
    # Body 2: Fixed front end (flag-like)
    add_flexible_body!(config,
        [1.5, 2.2], 0.8, 20,
        thickness = 0.03,
        rigidity = 5.0,
        front_constraint = :fixed
    )
    
    # Body 3: Different swimming pattern
    add_flexible_body!(config,
        [2.5, 0.8], 1.0, 22,
        thickness = 0.045,
        rigidity = 12.0,
        front_constraint = :sinusoidal,
        motion_amplitude = 0.2,
        motion_frequency = 2.5
    )
    
    solver = create_solver(config)
    initial_state = initialize_simulation(config)
    final_state = run_simulation(config, solver, initial_state)
    
    println("Multiple flexible bodies simulation completed!")
    return final_state
end

# Example with complex 3D geometry
function complex_3d_geometry()
    println("Running 3D simulation with complex geometry...")
    
    config = create_3d_simulation_config(
        nx = 48,
        ny = 32,
        nz = 24,
        Lx = 6.0,
        Ly = 2.0,
        Lz = 1.5,
        Reynolds = 200.0,
        dt = 0.008,
        final_time = 5.0,
        output_interval = 0.2,
        adaptive_refinement = true,
        output_file = "complex_3d_geometry"
    )
    
    # Create a complex arrangement of bodies
    # Central large sphere
    add_rigid_circle!(config, [3.0, 1.0, 0.75], 0.3)
    
    # Surrounding smaller spheres
    add_rigid_circle!(config, [2.0, 0.6, 0.4], 0.12)
    add_rigid_circle!(config, [2.2, 1.4, 1.1], 0.1)
    add_rigid_circle!(config, [3.8, 0.7, 0.3], 0.15)
    add_rigid_circle!(config, [3.6, 1.3, 1.2], 0.11)
    
    # Cubic obstacles
    add_rigid_square!(config, [1.5, 1.0, 0.75], 0.2)
    add_rigid_square!(config, [4.5, 1.0, 0.75], 0.25)
    
    solver = create_solver(config)
    initial_state = initialize_simulation(config, initial_conditions=:uniform_flow)
    final_state = run_simulation(config, solver, initial_state)
    
    println("Complex 3D geometry simulation completed!")
    return final_state
end

# Demonstration of different boundary conditions
function boundary_conditions_demo()
    println("Demonstrating different boundary condition combinations...")
    
    # Example 1: Free-slip walls
    config1 = create_2d_simulation_config(
        nx = 64, ny = 32, Lx = 4.0, Ly = 2.0,
        Reynolds = 100.0,
        wall_type = :free_slip,  # Free-slip instead of no-slip
        final_time = 3.0,
        output_file = "free_slip_walls"
    )
    
    add_rigid_circle!(config1, [2.0, 1.0], 0.2)
    
    solver1 = create_solver(config1)
    state1 = initialize_simulation(config1, initial_conditions=:uniform_flow)
    run_simulation(config1, solver1, state1)
    
    # Example 2: Velocity outlet
    config2 = create_2d_simulation_config(
        nx = 64, ny = 32, Lx = 4.0, Ly = 2.0,
        Reynolds = 100.0,
        outlet_type = :velocity,  # Velocity outlet instead of pressure
        final_time = 3.0,
        output_file = "velocity_outlet"
    )
    
    add_rigid_circle!(config2, [2.0, 1.0], 0.2)
    
    solver2 = create_solver(config2)
    state2 = initialize_simulation(config2, initial_conditions=:uniform_flow)
    run_simulation(config2, solver2, state2)
    
    println("Boundary conditions demo completed!")
end

# Run advanced examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("BioFlow.jl Advanced Features Examples")
    println("="^50)
    
    # Example 1: Custom grid
    custom_grid_example()
    
    # Example 2: Adaptive refinement
    adaptive_refinement_example()
    
    # Example 3: Multiple flexible bodies
    multiple_flexible_bodies()
    
    # Example 4: Complex 3D geometry
    complex_3d_geometry()
    
    # Example 5: Boundary conditions
    boundary_conditions_demo()
    
    println("\nAll advanced examples completed successfully!")
end