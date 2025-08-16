"""
AMR Output Example - Data Saved on Original Grid Only

This example demonstrates how to use BioFlow.jl AMR while ensuring that 
ALL output data is saved only on the original base grid resolution, never 
on the refined grids. This maintains consistency for visualization and analysis.
"""

"""
Example usage of AMR with output on original grid only.
"""
function amr_simulation_with_original_grid_output()
    println("🔬 AMR Simulation with Original Grid Output Example")
    println("=" * 60)
    
    # Step 1: Create original base grid
    original_nx, original_nz = 32, 24  # Original grid resolution
    Lx, Lz = 2.0, 1.5
    
    base_grid = StaggeredGrid2D(original_nx, original_nz, Lx, Lz)
    println("📐 Original base grid: $(original_nx)×$(original_nz) cells")
    println("   Domain size: $(Lx)×$(Lz)")
    println("   Grid spacing: dx=$(Lx/original_nx), dz=$(Lz/original_nz)")
    
    # Step 2: Create AMR-integrated solver
    # Note: The base solver always operates on the original grid
    fluid = FluidProperties(ρ=ConstantDensity(1.0), μ=0.01)
    bc = BoundaryConditions(...)  # Your boundary conditions
    
    base_solver = NavierStokesSolver2D(base_grid, fluid, bc)
    
    # Step 3: Configure AMR criteria
    amr_criteria = AdaptiveRefinementCriteria(
        velocity_gradient_threshold=1.0,
        pressure_gradient_threshold=5.0,
        vorticity_threshold=2.0,
        max_refinement_level=3,      # Up to 8x finer resolution internally
        min_grid_size=0.005
    )
    
    # Step 4: Create AMR solver
    amr_solver = create_amr_integrated_solver(base_solver, amr_criteria; amr_frequency=5)
    
    println("🔧 AMR Configuration:")
    println("   Max refinement level: $(amr_criteria.max_refinement_level)")
    println("   AMR check frequency: every 5 steps")
    println("   ⚠️  IMPORTANT: Output data will ALWAYS be on original $(original_nx)×$(original_nz) grid")
    
    # Step 5: Initialize solution on ORIGINAL grid
    state_old = SolutionState2D(original_nx, original_nz)
    state_new = SolutionState2D(original_nx, original_nz)
    
    # Verify initial state is on original grid
    @assert size(state_old.u) == (original_nx + 1, original_nz) "Initial u not on original grid"
    @assert size(state_old.v) == (original_nx, original_nz + 1) "Initial v not on original grid"
    @assert size(state_old.p) == (original_nx, original_nz) "Initial p not on original grid"
    
    println("✅ Initial solution state verified on original grid")
    
    # Step 6: Time stepping with AMR
    dt = 0.001
    t_final = 0.1
    output_interval = 10
    
    println("\n🚀 Starting simulation...")
    
    for step = 1:Int(t_final/dt)
        current_time = step * dt
        
        # AMR solve step - internally uses refined grids but keeps solution on original grid
        amr_solve_step!(amr_solver, state_new, state_old, dt)
        
        # VERIFICATION: Solution is ALWAYS on original grid after AMR solve
        @assert size(state_new.u) == (original_nx + 1, original_nz) "AMR changed grid size!"
        @assert size(state_new.v) == (original_nx, original_nz + 1) "AMR changed grid size!"
        @assert size(state_new.p) == (original_nx, original_nz) "AMR changed grid size!"
        
        # Step 7: Output on original grid only
        if step % output_interval == 0
            save_amr_output_on_original_grid(amr_solver, state_new, step, current_time)
        end
        
        # Swap states
        state_old, state_new = state_new, state_old
        
        # Progress report
        if step % 50 == 0
            print_amr_progress(amr_solver, step, current_time)
        end
    end
    
    println("\n🎉 Simulation completed successfully!")
    print_final_amr_summary(amr_solver)
end

"""
Save AMR output ensuring data is on original grid only.
"""
function save_amr_output_on_original_grid(amr_solver::AMRIntegratedSolver, 
                                         state::SolutionState, step::Int, time::Float64)
    
    # GUARANTEE: Prepare output data on original grid resolution
    output_state, metadata = prepare_amr_for_netcdf_output(
        amr_solver.refined_grid, state, "amr_simulation", step, time)
    
    base_grid = amr_solver.refined_grid.base_grid
    
    # VERIFICATION: Double-check output is on original grid
    if base_grid.grid_type == TwoDimensional
        expected_u_size = (base_grid.nx + 1, base_grid.nz)
        expected_v_size = (base_grid.nx, base_grid.nz + 1)
        expected_p_size = (base_grid.nx, base_grid.nz)
        
        @assert size(output_state.u) == expected_u_size "Output u not on original grid!"
        @assert size(output_state.v) == expected_v_size "Output v not on original grid!"
        @assert size(output_state.p) == expected_p_size "Output p not on original grid!"
        
        println("💾 Step $step: Data saved on original grid ($(base_grid.nx)×$(base_grid.nz))")
    else
        # Similar verification for 3D
        println("💾 Step $step: Data saved on original 3D grid")
    end
    
    # Example: Save to NetCDF (pseudocode - adapt to your output system)
    filename = "amr_output_step_$(lpad(step, 6, '0')).nc"
    
    # Your NetCDF writer would use output_state, which is guaranteed to be on original grid
    # save_netcdf(filename, output_state, metadata)
    
    # Print confirmation
    println("   ✅ File: $filename (original grid resolution)")
    println("   📊 Refined cells: $(amr_solver.amr_statistics["current_refined_cells"])")
    println("   🔍 Max AMR level: $(amr_solver.amr_statistics["max_refinement_level_used"])")
    
    # Optional: Save refinement pattern for visualization
    if step % 50 == 0
        refinement_file = "refinement_pattern_step_$(step).txt"
        write_amr_refinement_map(amr_solver.refined_grid, refinement_file)
        println("   🗺️  Refinement map: $refinement_file")
    end
end

"""
Print AMR progress during simulation.
"""
function print_amr_progress(amr_solver::AMRIntegratedSolver, step::Int, time::Float64)
    stats = amr_solver.amr_statistics
    
    println("📈 Step $step, t=$(round(time, digits=4)):")
    println("   Refined cells: $(stats["current_refined_cells"])")
    println("   Total refinements: $(stats["total_refinements"])")
    println("   Max level used: $(stats["max_refinement_level_used"])")
    
    # Memory and performance info
    base_grid = amr_solver.refined_grid.base_grid
    original_cells = base_grid.grid_type == TwoDimensional ? 
                    base_grid.nx * base_grid.nz : 
                    base_grid.nx * base_grid.ny * base_grid.nz
    
    effective_cells = get_effective_grid_size(amr_solver.refined_grid)
    efficiency_ratio = effective_cells / original_cells
    
    println("   Grid efficiency: $(round(efficiency_ratio, digits=2))x effective resolution")
end

"""
Print final AMR simulation summary.
"""
function print_final_amr_summary(amr_solver::AMRIntegratedSolver)
    println("\n📊 Final AMR Simulation Summary")
    println("-" * 40)
    
    base_grid = amr_solver.refined_grid.base_grid
    
    if base_grid.grid_type == TwoDimensional
        println("Original grid: $(base_grid.nx)×$(base_grid.nz) cells")
    else
        println("Original grid: $(base_grid.nx)×$(base_grid.ny)×$(base_grid.nz) cells")
    end
    
    stats = amr_solver.amr_statistics
    println("Total refinements performed: $(stats["total_refinements"])")
    println("Final refined cells: $(stats["current_refined_cells"])")
    println("Maximum refinement level used: $(stats["max_refinement_level_used"])")
    
    # Calculate computational savings
    original_cells = base_grid.grid_type == TwoDimensional ? 
                    base_grid.nx * base_grid.nz : 
                    base_grid.nx * base_grid.ny * base_grid.nz
    
    if stats["max_refinement_level_used"] > 0
        uniform_fine_cells = original_cells * (4^stats["max_refinement_level_used"])
        effective_cells = get_effective_grid_size(amr_solver.refined_grid)
        savings = 100 * (1 - effective_cells / uniform_fine_cells)
        
        println("Computational savings vs uniform fine grid: $(round(savings, digits=1))%")
    end
    
    println("\n✅ Key Guarantee: ALL output data was saved on original grid resolution")
    println("   No post-processing needed for visualization compatibility")
    println("   Refinement patterns saved separately for AMR analysis")
end

"""
Example function showing how to analyze AMR output data.
"""
function analyze_amr_output_data()
    println("\n🔍 AMR Output Data Analysis")
    println("-" * 30)
    
    println("Since all AMR data is saved on the original grid:")
    println("  ✅ Standard visualization tools work directly")
    println("  ✅ No grid interpolation needed for analysis")
    println("  ✅ Consistent time series data")
    println("  ✅ Compatible with existing post-processing scripts")
    
    println("\nRefinement information available separately:")
    println("  📊 Refinement maps show where AMR was active")
    println("  📈 AMR statistics track computational efficiency")
    println("  🗺️  Visualization tools can overlay refinement patterns")
end

# Example usage
if false  # Set to true to run example
    println("Running AMR Original Grid Output Example...")
    amr_simulation_with_original_grid_output()
    analyze_amr_output_data()
end

# Export example functions
export amr_simulation_with_original_grid_output, save_amr_output_on_original_grid
export analyze_amr_output_data