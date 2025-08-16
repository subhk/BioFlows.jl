"""
AMR-NetCDF Integration Example

This example demonstrates how to use the AMR system with the existing 
netcdf_writer.jl for output. The AMR system handles data transfer back 
to original grids, and netcdf_writer.jl handles the actual file writing.
"""

"""
    run_amr_simulation_with_netcdf_output()

Complete example of AMR simulation with NetCDF output using existing netcdf_writer.jl.
"""
function run_amr_simulation_with_netcdf_output()
    println("🚀 AMR Simulation with NetCDF Output Example")
    println("=" * 60)
    
    # Step 1: Create original base grid (this will be the output grid)
    original_nx, original_nz = 64, 48
    Lx, Lz = 2.0, 1.5
    base_grid = StaggeredGrid2D(original_nx, original_nz, Lx, Lz)
    
    println("📐 Base grid: $(original_nx)×$(original_nz) cells")
    println("   Domain: $(Lx)×$(Lz), dx=$(Lx/original_nx), dz=$(Lz/original_nz)")
    
    # Step 2: Create NetCDF writer for original grid
    netcdf_config = NetCDFConfig(
        "amr_flow_output";
        max_snapshots_per_file=50,
        save_mode=:both,
        time_interval=0.05,
        iteration_interval=10,
        save_flow_field=true,
        save_body_positions=false
    )
    
    netcdf_writer = NetCDFWriter(netcdf_config, base_grid)
    println("📁 NetCDF writer configured for original grid dimensions")
    
    # Step 3: Create AMR-integrated solver
    fluid = FluidProperties(ρ=ConstantDensity(1.0), μ=0.01)
    bc = BoundaryConditions(...)  # Your boundary conditions
    
    base_solver = NavierStokesSolver2D(base_grid, fluid, bc)
    
    amr_criteria = AdaptiveRefinementCriteria(
        velocity_gradient_threshold=1.0,
        pressure_gradient_threshold=5.0,
        vorticity_threshold=2.0,
        max_refinement_level=3,      # Up to 8x finer resolution internally
        min_grid_size=0.002
    )
    
    amr_solver = create_amr_integrated_solver(base_solver, amr_criteria; amr_frequency=5)
    println("🔧 AMR solver created (max level $(amr_criteria.max_refinement_level))")
    
    # Step 4: Initialize solution on original grid
    state_old = SolutionState2D(original_nx, original_nz)
    state_new = SolutionState2D(original_nx, original_nz)
    
    # Initialize with some flow pattern
    initialize_test_flow!(state_old, base_grid)
    
    println("✅ Initial solution on original grid: u$(size(state_old.u)), v$(size(state_old.v)), p$(size(state_old.p))")
    
    # Step 5: Main simulation loop with AMR + NetCDF output
    dt = 0.001
    t_final = 0.2
    n_steps = Int(t_final / dt)
    
    println("\\n🔄 Starting AMR simulation with NetCDF output...")
    println("   Steps: $n_steps, dt: $dt, t_final: $t_final")
    
    for step = 1:n_steps
        current_time = step * dt
        
        # AMR solve step - IMPORTANT: Solution ALWAYS lives on original grid!
        # Refined grids are used internally for flux/gradient accuracy only
        amr_solve_step!(amr_solver, state_new, state_old, dt)
        
        # VERIFICATION: Solution is ALWAYS on original grid (no projection needed)
        @assert size(state_new.u) == (original_nx + 1, original_nz) "AMR broke original grid consistency!"
        @assert size(state_new.v) == (original_nx, original_nz + 1) "AMR broke original grid consistency!"
        @assert size(state_new.p) == (original_nx, original_nz) "AMR broke original grid consistency!"
        
        # Step 6: AMR + NetCDF output (ONLY during file writing!)
        if step % 10 == 0  # Output every 10 steps
            # NOTE: AMR-to-original projection happens ONLY here, not every iteration!
            # This preserves performance while providing enhanced output accuracy
            output_state, success = save_amr_to_netcdf!(
                netcdf_writer, amr_solver.refined_grid, state_new, step, current_time)
            
            if success
                println("✅ Step $step: AMR data saved to NetCDF (original grid $(original_nx)×$(original_nz))")
            end
        end
        
        # Swap states
        state_old, state_new = state_new, state_old
        
        # Progress report
        if step % 50 == 0
            refined_cells = length(amr_solver.refined_grid.refined_cells_2d)
            println("🔍 Step $step: $refined_cells refined cells, t=$(round(current_time, digits=3))")
        end
    end
    
    # Step 7: Close NetCDF file
    close_netcdf!(netcdf_writer)
    
    println("\\n🎉 AMR-NetCDF simulation completed!")
    println("🎯 Key results:")
    println("   • ALL NetCDF data saved on original $(original_nx)×$(original_nz) grid")
    println("   • AMR used internally for accuracy (up to 8x finer)")
    println("   • Compatible with existing visualization tools")
    println("   • AMR refinement maps saved separately for analysis")
    
    return amr_solver, netcdf_writer
end

"""
    simple_amr_netcdf_workflow(netcdf_writer, amr_solver, state, step, time)

Simplified workflow for existing simulations to add AMR-NetCDF output.
"""
function simple_amr_netcdf_workflow(netcdf_writer, amr_solver, state, step, time)
    # One-line AMR to NetCDF output
    output_state, success = save_amr_to_netcdf!(
        netcdf_writer, amr_solver.refined_grid, state, step, time)
    
    if success
        println("✅ AMR data written to NetCDF file at step $step")
    else
        println("⏸️  NetCDF save skipped (save conditions not met)")
    end
    
    return output_state, success
end

"""
    initialize_test_flow!(state, grid)

Initialize a simple test flow pattern.
"""
function initialize_test_flow!(state::SolutionState, grid::StaggeredGrid)
    nx, nz = grid.nx, grid.nz
    Lx = grid.x[end] - grid.x[1] + grid.dx
    Lz = grid.z[end] - grid.z[1] + grid.dz
    
    # Create a simple flow pattern for testing
    for j = 1:nz, i = 1:nx+1
        x = grid.x[1] + (i-1) * grid.dx
        z = grid.z[1] + (j-0.5) * grid.dz
        state.u[i, j] = sin(π * x / Lx) * cos(π * z / Lz)
    end
    
    for j = 1:nz+1, i = 1:nx
        x = grid.x[1] + (i-0.5) * grid.dx
        z = grid.z[1] + (j-1) * grid.dz
        state.v[i, j] = -cos(π * x / Lx) * sin(π * z / Lz) * 0.5  # w-velocity in XZ plane
    end
    
    for j = 1:nz, i = 1:nx
        x = grid.x[1] + (i-0.5) * grid.dx
        z = grid.z[1] + (j-0.5) * grid.dz
        state.p[i, j] = 0.5 * (sin(2π * x / Lx) + cos(2π * z / Lz))
    end
end

# Example usage instructions
"""
## Usage in Your Code

### For New Simulations:
```julia
# Setup (once)
amr_solver, netcdf_writer = run_amr_simulation_with_netcdf_output()
```

### For Existing Simulations:
```julia
# In your time loop:
for step = 1:n_steps
    # Your AMR solve
    amr_solve_step!(amr_solver, state_new, state_old, dt)
    
    # AMR + NetCDF output (one line)
    if step % output_interval == 0
        save_amr_to_netcdf!(netcdf_writer, amr_solver.refined_grid, state_new, step, time)
    end
end
```

### Key Benefits:
- ✅ AMR handles transfer back to original grids automatically
- ✅ netcdf_writer.jl handles file writing unchanged
- ✅ Output compatible with existing tools (ParaView, matplotlib, etc.)
- ✅ AMR refinement maps saved separately for analysis

### Performance Design:
- 🚀 Solution ALWAYS lives on original grid (no projection overhead)
- 🚀 Refined grids used internally for flux/gradient accuracy only
- 🚀 AMR-to-original projection happens ONLY during file writing
- 🚀 Optimal performance: no unnecessary data copying per iteration
"""

# Export example functions
export run_amr_simulation_with_netcdf_output, simple_amr_netcdf_workflow
export initialize_test_flow!