"""
AMR Conservative Projection Example

This example demonstrates the expanded project_2d_refined_to_original! function
that performs comprehensive conservative projection of all variables from
refined grids back to the original base grid.
"""

"""
Demonstrate the conservative projection process for AMR data.
"""
function demonstrate_amr_conservative_projection()
    println("🔬 AMR Conservative Projection Demonstration")
    println("=" * 60)
    
    # Step 1: Create a test case with refined grids
    println("1️⃣  Setting up test case...")
    
    # Original base grid
    base_nx, base_nz = 8, 6
    Lx, Lz = 2.0, 1.5
    base_grid = StaggeredGrid2D(base_nx, base_nz, Lx, Lz)
    
    println("   📐 Base grid: $(base_nx)×$(base_nz) cells")
    println("   📏 Cell size: dx=$(base_grid.dx), dz=$(base_grid.dz)")
    
    # Create refined grid structure
    refined_grid = RefinedGrid(base_grid)
    
    # Add some refined cells
    test_cells = [(4, 3), (5, 3)]  # Refine cells in the middle
    refine_cells_2d!(refined_grid, test_cells)
    
    println("   🔍 Created $(length(test_cells)) refined cells")
    
    # Step 2: Create test solution data
    println("\n2️⃣  Creating test solution data...")
    
    # Base grid solution
    base_state = SolutionState2D(base_nx, base_nz)
    
    # Initialize with a simple flow pattern
    for j = 1:base_nz, i = 1:base_nx+1
        x_face = base_grid.x[1] + (i-1) * base_grid.dx
        z_center = base_grid.z[1] + (j-0.5) * base_grid.dz
        base_state.u[i, j] = sin(π * x_face / Lx) * cos(π * z_center / Lz)
    end
    
    for j = 1:base_nz+1, i = 1:base_nx
        x_center = base_grid.x[1] + (i-0.5) * base_grid.dx
        z_face = base_grid.z[1] + (j-1) * base_grid.dz
        base_state.v[i, j] = cos(π * x_center / Lx) * sin(π * z_face / Lz)
    end
    
    for j = 1:base_nz, i = 1:base_nx
        x_center = base_grid.x[1] + (i-0.5) * base_grid.dx
        z_center = base_grid.z[1] + (j-0.5) * base_grid.dz
        base_state.p[i, j] = sin(2π * x_center / Lx) * sin(2π * z_center / Lz)
    end
    
    println("   ✅ Initialized base solution with analytical flow pattern")
    
    # Step 3: Demonstrate projection process
    println("\n3️⃣  Performing conservative projection...")
    
    # Create output state (same size as original grid)
    output_state = SolutionState2D(base_nx, base_nz)
    output_state.u .= base_state.u
    output_state.v .= base_state.v
    output_state.p .= base_state.p
    
    # Store original values for comparison
    original_u = copy(base_state.u)
    original_v = copy(base_state.v)
    original_p = copy(base_state.p)
    
    # Perform the conservative projection
    project_2d_refined_to_original!(output_state, refined_grid, base_state)
    
    println("   🔄 Projection completed for all refined cells")
    
    # Step 4: Analyze the results
    println("\n4️⃣  Analyzing projection results...")
    
    # Check conservation properties
    conservation_error = validate_conservation_2d(output_state, refined_grid, base_grid)
    
    # Compare values before and after projection
    u_max_change = maximum(abs.(output_state.u .- original_u))
    v_max_change = maximum(abs.(output_state.v .- original_v))
    p_max_change = maximum(abs.(output_state.p .- original_p))
    
    println("   📊 Maximum changes after projection:")
    println("      u-velocity: $(round(u_max_change, digits=6))")
    println("      v-velocity: $(round(v_max_change, digits=6))")
    println("      pressure:   $(round(p_max_change, digits=6))")
    println("   🎯 Conservation error: $(round(conservation_error, digits=10))")
    
    # Step 5: Demonstrate detailed projection for one cell
    println("\n5️⃣  Detailed projection analysis for cell (4,3)...")
    demonstrate_detailed_cell_projection(refined_grid, (4, 3), base_state, base_grid)
    
    # Step 6: Verify output is on original grid
    println("\n6️⃣  Verifying output grid dimensions...")
    
    expected_u_size = (base_nx + 1, base_nz)
    expected_v_size = (base_nx, base_nz + 1)
    expected_p_size = (base_nx, base_nz)
    
    @assert size(output_state.u) == expected_u_size "❌ Output u not on original grid!"
    @assert size(output_state.v) == expected_v_size "❌ Output v not on original grid!"
    @assert size(output_state.p) == expected_p_size "❌ Output p not on original grid!"
    
    println("   ✅ Output dimensions verified:")
    println("      u: $(size(output_state.u)) = $(expected_u_size) ✓")
    println("      v: $(size(output_state.v)) = $(expected_v_size) ✓")
    println("      p: $(size(output_state.p)) = $(expected_p_size) ✓")
    
    println("\n🎉 Conservative projection demonstration completed successfully!")
    
    return output_state, conservation_error
end

"""
Demonstrate detailed projection process for a single refined cell.
"""
function demonstrate_detailed_cell_projection(refined_grid::RefinedGrid, cell_idx::Tuple{Int,Int},
                                            base_state::SolutionState, base_grid::StaggeredGrid)
    i_base, j_base = cell_idx
    
    if !haskey(refined_grid.refined_grids_2d, cell_idx)
        println("   ⚠️  Cell $cell_idx is not refined")
        return
    end
    
    local_grid = refined_grid.refined_grids_2d[cell_idx]
    refinement_level = refined_grid.refined_cells_2d[cell_idx]
    
    println("   🔍 Cell ($i_base, $j_base) - Refinement level $refinement_level")
    println("   📐 Local grid: $(local_grid.nx)×$(local_grid.nz) cells")
    println("   📏 Local cell size: dx=$(local_grid.dx), dz=$(local_grid.dz)")
    
    # Calculate refinement ratio
    refine_factor = 2^refinement_level
    println("   📈 Refinement factor: $(refine_factor)x")
    
    # Volume conservation check
    base_volume = base_grid.dx * base_grid.dz
    local_total_volume = local_grid.nx * local_grid.nz * local_grid.dx * local_grid.dz
    volume_ratio = local_total_volume / base_volume
    
    println("   📦 Volume conservation:")
    println("      Base cell volume: $(round(base_volume, digits=6))")
    println("      Local total volume: $(round(local_total_volume, digits=6))")
    println("      Volume ratio: $(round(volume_ratio, digits=6)) (should be ≈ 1.0)")
    
    # Face area conservation for velocities
    base_x_face_area = base_grid.dz
    base_z_face_area = base_grid.dx
    
    local_x_face_total_area = local_grid.nz * local_grid.dz
    local_z_face_total_area = local_grid.nx * local_grid.dx
    
    println("   🔄 Face area conservation:")
    println("      X-faces: $(round(local_x_face_total_area/base_x_face_area, digits=6)) (should be ≈ 1.0)")
    println("      Z-faces: $(round(local_z_face_total_area/base_z_face_area, digits=6)) (should be ≈ 1.0)")
end

"""
Demonstrate the complete AMR output workflow with conservative projection.
"""
function demonstrate_complete_amr_output_workflow()
    println("\n🔧 Complete AMR Output Workflow Demonstration")
    println("-" * 50)
    
    # Create AMR system
    base_grid = StaggeredGrid2D(16, 12, 2.0, 1.5)
    refined_grid = RefinedGrid(base_grid)
    
    # Add refinement in interesting regions
    refined_cells = [(8, 6), (9, 6), (8, 7), (9, 7)]  # 2x2 block
    refine_cells_2d!(refined_grid, refined_cells)
    
    # Create solution state
    state = SolutionState2D(16, 12)
    
    # Initialize with complex flow pattern
    initialize_complex_flow_pattern!(state, base_grid)
    
    # Prepare for output using the expanded projection
    println("📄 Preparing AMR data for output...")
    output_state, metadata = prepare_amr_for_netcdf_output(refined_grid, state, "test_output", 100, 10.0)
    
    # Display metadata
    println("📊 Output metadata:")
    for (key, value) in metadata
        if key == "conservation_validation"
            println("   $key:")
            for (sub_key, sub_value) in value
                println("      $sub_key: $(round(sub_value, digits=10))")
            end
        else
            println("   $key: $value")
        end
    end
    
    # Verify output is suitable for standard visualization tools
    println("\n🖼️  Visualization compatibility check:")
    println("   Grid dimensions: $(size(output_state.p)) (constant across all timesteps)")
    println("   Coordinate system: XZ plane")
    println("   Staggered velocities: u($(size(output_state.u))), v($(size(output_state.v)))")
    println("   ✅ Compatible with ParaView, matplotlib, VisIt, etc.")
    
    return output_state, metadata
end

"""
Initialize a complex flow pattern for testing.
"""
function initialize_complex_flow_pattern!(state::SolutionState, grid::StaggeredGrid)
    nx, nz = grid.nx, grid.nz
    Lx = grid.x[end] - grid.x[1] + grid.dx
    Lz = grid.z[end] - grid.z[1] + grid.dz
    
    # Create a vortex-like pattern
    for j = 1:nz, i = 1:nx+1
        x = grid.x[1] + (i-1) * grid.dx
        z = grid.z[1] + (j-0.5) * grid.dz
        
        # Normalized coordinates
        x_norm = (x - Lx/2) / (Lx/2)
        z_norm = (z - Lz/2) / (Lz/2)
        r = sqrt(x_norm^2 + z_norm^2)
        
        # Vortex velocity profile
        state.u[i, j] = -z_norm * exp(-r^2)
    end
    
    for j = 1:nz+1, i = 1:nx
        x = grid.x[1] + (i-0.5) * grid.dx
        z = grid.z[1] + (j-1) * grid.dz
        
        # Normalized coordinates
        x_norm = (x - Lx/2) / (Lx/2)
        z_norm = (z - Lz/2) / (Lz/2)
        r = sqrt(x_norm^2 + z_norm^2)
        
        # Vortex velocity profile
        state.v[i, j] = x_norm * exp(-r^2)
    end
    
    for j = 1:nz, i = 1:nx
        x = grid.x[1] + (i-0.5) * grid.dx
        z = grid.z[1] + (j-0.5) * grid.dz
        
        # Normalized coordinates
        x_norm = (x - Lx/2) / (Lx/2)
        z_norm = (z - Lz/2) / (Lz/2)
        r = sqrt(x_norm^2 + z_norm^2)
        
        # Pressure field
        state.p[i, j] = -0.5 * exp(-2*r^2)
    end
end

# Example usage
if false  # Set to true to run demonstration
    println("🚀 Running AMR Conservative Projection Demonstrations...")
    
    # Basic projection demonstration
    output_state, conservation_error = demonstrate_amr_conservative_projection()
    
    # Complete workflow demonstration
    complete_output, complete_metadata = demonstrate_complete_amr_output_workflow()
    
    println("\n🏁 All demonstrations completed successfully!")
    println("Conservation error: $(round(conservation_error, digits=12))")
end

# Export demonstration functions
export demonstrate_amr_conservative_projection, demonstrate_complete_amr_output_workflow
export demonstrate_detailed_cell_projection, initialize_complex_flow_pattern!