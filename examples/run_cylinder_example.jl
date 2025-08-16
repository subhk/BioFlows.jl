"""
Simple Runner Script for 2D Cylinder Flow Example

This script provides a simplified interface to run the cylinder flow simulation
with different parameter sets and configurations.
"""

using BioFlows
include("flow_past_cylinder_2d.jl")

"""
Quick run with default parameters
"""
function quick_cylinder_run()
    println("Running cylinder flow simulation with default parameters...")
    return demonstrate_cylinder_flow()
end

"""
Run with custom Reynolds number
"""
function run_cylinder_custom_re(Re::Float64)
    println("Running cylinder flow simulation with Re = $Re")
    
    # Modify the create_cylinder_simulation function for custom Re
    U_inlet = 1.0
    D_cylinder = 0.4
    ρ = 1000.0
    μ = ρ * U_inlet * D_cylinder / Re
    
    config = create_2d_simulation_config(
        nx = 160, nz = 80,
        Lx = 8.0, Lz = 4.0,
        Reynolds = Re,
        density_value = ρ,
        viscosity = μ,
        inlet_velocity = U_inlet,
        outlet_type = :pressure,
        wall_type = :no_slip,
        time_scheme = :rk4,
        dt = 0.01,
        final_time = 20.0,
        adaptive_refinement = true,
        output_interval = 0.1,
        output_file = "cylinder_flow_Re$(Int(Re))"
    )
    
    cylinder_bodies = create_cylinder_body()
    amr_criteria = setup_adaptive_refinement()
    final_config = add_bodies_and_amr!(config, cylinder_bodies, amr_criteria)
    
    return run_simulation(final_config)
end

"""
Run with high resolution for detailed analysis
"""
function run_cylinder_high_res()
    println("Running high-resolution cylinder flow simulation...")
    
    config = create_2d_simulation_config(
        nx = 320, nz = 160,  # Double resolution
        Lx = 8.0, Lz = 4.0,
        Reynolds = 500.0,
        inlet_velocity = 1.0,
        outlet_type = :pressure,
        wall_type = :no_slip,
        time_scheme = :rk4,
        dt = 0.005,  # Smaller time step
        final_time = 20.0,
        adaptive_refinement = true,
        output_interval = 0.05,  # More frequent output
        output_file = "cylinder_flow_highres"
    )
    
    cylinder_bodies = create_cylinder_body()
    
    # More aggressive AMR criteria for high resolution
    amr_criteria = AdaptiveRefinementCriteria(
        velocity_gradient_threshold = 1.5,
        pressure_gradient_threshold = 3.0,
        vorticity_threshold = 2.0,
        body_proximity_distance = 0.3,
        max_refinement_level = 4,  # More refinement levels
        min_grid_size = 0.005
    )
    
    final_config = add_bodies_and_amr!(config, cylinder_bodies, amr_criteria)
    
    return run_simulation(final_config)
end

# """
# Parameter study: Run multiple Reynolds numbers
# """
# function reynolds_study()
#     println("Running Reynolds number study...")
    
#     Re_values = [40.0, 60.0, 80.0, 100.0, 150.0, 200.0]
#     results = Dict()
    
#     for Re in Re_values
#         println("  Running Re = $Re...")
#         try
#             success = run_cylinder_custom_re(Re)
#             results[Re] = success ? "SUCCESS" : "FAILED"
#         catch e
#             println("    Error: $e")
#             results[Re] = "ERROR"
#         end
#     end
    
#     println("\n Reynolds Study Results:")
#     for (Re, status) in results
#         println("  Re = $Re: $status")
#     end
    
#     return results
# end

# Interactive menu
function interactive_menu()
    println("\n" * "="^50)
    println("BioFlow.jl - 2D Cylinder Flow Examples")
    println("="^50)
    println("Choose an option:")
    println("1. Quick run (default parameters, Re=100)")
    println("2. Custom Reynolds number")
    println("3. High resolution simulation")
    println("4. Reynolds number study (Re=40-200)")
    println("5. Exit")
    println("="^50)
    
    while true
        print("Enter choice (1-5): ")
        choice = readline()
        
        if choice == "1"
            quick_cylinder_run()
            break
        elseif choice == "2"
            print("Enter Reynolds number (40-200): ")
            Re_str = readline()
            try
                Re = parse(Float64, Re_str)
                if 40 <= Re <= 200
                    run_cylinder_custom_re(Re)
                else
                    println("Reynolds number should be between 40 and 200")
                    continue
                end
            catch
                println("Invalid Reynolds number")
                continue
            end
            break
        elseif choice == "3"
            run_cylinder_high_res()
            break
        elseif choice == "4"
            reynolds_study()
            break
        elseif choice == "5"
            println("Goodbye!")
            break
        else
            println("Invalid choice. Please enter 1-5.")
        end
    end
end

# Run interactive menu if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    interactive_menu()
end