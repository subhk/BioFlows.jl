"""
Comprehensive AMR Integration for BioFlow.jl

This module provides the main integration points for adaptive mesh refinement
with the BioFlow.jl codebase, ensuring all components work together seamlessly.
"""

# Guarded includes to avoid double-loading when included from BioFlows.jl
if !isdefined(@__MODULE__, :AdaptiveRefinementCriteria)
    include("adaptive_refinement.jl")
end
if !isdefined(@__MODULE__, :distance_to_surface_2d)
    include("amr_helpers.jl")
end
if !isdefined(@__MODULE__, :validate_amr_setup)
    include("amr_validation.jl")
end
if !isdefined(@__MODULE__, :apply_boundary_conditions_amr!)
    include("amr_boundary_conditions.jl")
end
if !isdefined(@__MODULE__, :ensure_output_on_original_grid!)
    include("amr_output_integration.jl")
end

# Include advanced AMR if available
if !isdefined(@__MODULE__, :AMRHierarchy)
    try
        include("adaptive_refinement_v2.jl")
        global HAS_ADVANCED_AMR = true
    catch
        global HAS_ADVANCED_AMR = false
        println("Advanced AMR (v2) not available - using basic AMR only")
    end
else
    global HAS_ADVANCED_AMR = true
end

# Include MPI AMR if MPI is available
if !isdefined(@__MODULE__, :MPIAMRHierarchy)
    try
        include("adaptive_refinement_mpi.jl")
        global HAS_MPI_AMR = true
    catch
        global HAS_MPI_AMR = false
        println("MPI AMR not available - MPI support disabled for AMR")
    end
else
    global HAS_MPI_AMR = true
end

"""
    AMRIntegratedSolver

Main solver structure that integrates AMR with existing BioFlow.jl solvers.
"""
mutable struct AMRIntegratedSolver{T,N} <: AbstractSolver where {T<:AbstractFloat, N}
    base_solver::Union{NavierStokesSolver2D, NavierStokesSolver3D, MPINavierStokesSolver2D, MPINavierStokesSolver3D}
    refined_grid::RefinedGrid
    amr_criteria::AdaptiveRefinementCriteria
    amr_enabled::Bool
    
    # AMR timing and control
    last_amr_step::Int
    amr_frequency::Int  # How often to check for refinement
    
    # Advanced AMR (if available)
    amr_hierarchy::Union{Nothing, Any}  # AMRHierarchy from v2
    mpi_amr_hierarchy::Union{Nothing, Any}  # MPIAMRHierarchy if MPI available
    
    # Performance monitoring
    amr_timing::Dict{String, Float64}
    amr_statistics::Dict{String, Int}
end

"""
    create_amr_integrated_solver(solver, amr_criteria; amr_frequency=10)

Create an AMR-integrated solver from an existing BioFlow.jl solver.
"""
function create_amr_integrated_solver(solver, amr_criteria::AdaptiveRefinementCriteria; 
                                     amr_frequency::Int=10)
    # Get grid from existing solver
    if hasfield(typeof(solver), :grid)
        base_grid = solver.grid
    elseif hasfield(typeof(solver), :local_grid)
        base_grid = solver.local_grid
    else
        error("Cannot extract grid from solver type $(typeof(solver))")
    end
    
    # Create refined grid
    refined_grid = RefinedGrid(base_grid)
    
    # Initialize advanced AMR if available
    amr_hierarchy = if HAS_ADVANCED_AMR
        try
            AMRHierarchy(base_grid; 
                        max_level=amr_criteria.max_refinement_level,
                        velocity_gradient_threshold=amr_criteria.velocity_gradient_threshold,
                        pressure_gradient_threshold=amr_criteria.pressure_gradient_threshold,
                        vorticity_threshold=amr_criteria.vorticity_threshold,
                        body_distance_threshold=amr_criteria.body_distance_threshold)
        catch e
            println("Warning: Could not initialize advanced AMR: $e")
            nothing
        end
    else
        nothing
    end
    
    # Initialize MPI AMR if available and solver supports MPI
    mpi_amr_hierarchy = if HAS_MPI_AMR && (hasfield(typeof(solver), :decomp) || hasfield(typeof(solver), :mpi_comm))
        try
            if hasfield(typeof(solver), :decomp)
                MPIAMRHierarchy(base_grid, solver.decomp; 
                               max_level=amr_criteria.max_refinement_level)
            else
                nothing
            end
        catch e
            println("Warning: Could not initialize MPI AMR: $e")
            nothing
        end
    else
        nothing
    end
    
    # Initialize timing and statistics
    amr_timing = Dict{String, Float64}(
        "refinement" => 0.0,
        "coarsening" => 0.0,
        "interpolation" => 0.0,
        "boundary_conditions" => 0.0,
        "output_projection" => 0.0
    )
    
    amr_statistics = Dict{String, Int}(
        "total_refinements" => 0,
        "total_coarsenings" => 0,
        "current_refined_cells" => 0,
        "max_refinement_level_used" => 0
    )
    
    return AMRIntegratedSolver{Float64,2}(solver, refined_grid, amr_criteria, true,
                                         0, amr_frequency, amr_hierarchy, mpi_amr_hierarchy,
                                         amr_timing, amr_statistics)
end

"""
    amr_solve_step!(amr_solver, state_new, state_old, dt, bodies)

Main AMR-integrated solve step.
"""
function amr_solve_step!(amr_solver::AMRIntegratedSolver, 
                        state_new::SolutionState, state_old::SolutionState, 
                        dt::Float64, bodies=nothing)
    
    # Step 1: Check if AMR update is needed
    should_update_amr = (state_old.step - amr_solver.last_amr_step >= amr_solver.amr_frequency) || 
                       (state_old.step == 0)  # Always update on first step
    
    # Step 2: Perform AMR analysis and grid adaptation if needed
    if should_update_amr && amr_solver.amr_enabled
        update_amr_grid!(amr_solver, state_old, bodies)
    end
    
    # Step 3: Apply boundary conditions with AMR awareness
    amr_timing_start = time()
    apply_boundary_conditions_amr!(amr_solver.refined_grid, state_old, 
                                  amr_solver.base_solver.bc, state_old.t)
    amr_solver.amr_timing["boundary_conditions"] += time() - amr_timing_start
    
    # Step 4: Solve using the base solver (dispatch by grid dimensionality)
    base_grid = amr_solver.refined_grid.base_grid
    if base_grid.grid_type == TwoDimensional
        solve_step_2d!(amr_solver.base_solver, state_new, state_old, dt)
    else
        solve_step_3d!(amr_solver.base_solver, state_new, state_old, dt)
    end
    
    # Step 5: Enforce AMR boundary continuity
    enforce_boundary_continuity_amr!(amr_solver.refined_grid, state_new)
    
    # Step 6: Update AMR statistics
    update_amr_statistics!(amr_solver)
    
    # Step 7: Ensure state_new is on ORIGINAL grid for output consistency
    # The solver always works on the original grid, refined computation is internal
    ensure_output_on_original_grid!(amr_solver, state_new)
end

"""
    update_amr_grid!(amr_solver, state, bodies)

Update AMR grid based on solution indicators.
"""
function update_amr_grid!(amr_solver::AMRIntegratedSolver, state::SolutionState, bodies)
    amr_timing_start = time()
    
    if amr_solver.mpi_amr_hierarchy !== nothing && HAS_MPI_AMR
        # Use MPI-aware AMR
        try
            refined_count = coordinate_global_refinement!(amr_solver.mpi_amr_hierarchy, state, bodies)
            amr_solver.amr_statistics["total_refinements"] += refined_count
        catch e
            println("Warning: MPI AMR refinement failed: $e")
            # Fall back to basic AMR
            update_basic_amr!(amr_solver, state, bodies)
        end
    elseif amr_solver.amr_hierarchy !== nothing && HAS_ADVANCED_AMR
        # Use advanced AMR (v2): compute indicators and refine v2 hierarchy
        try
            level = amr_solver.amr_hierarchy.base_level
            inds = compute_refinement_indicators_amr(level, state, bodies, amr_solver.amr_hierarchy)
            # Mark cells above threshold (score > 0.5)
            marked = Tuple{Int,Int}[]
            for j = 1:size(inds, 2), i = 1:size(inds, 1)
                if inds[i, j] > 0.5
                    push!(marked, (i, j))
                end
            end
            # Use basic refine_cells_2d! on RefinedGrid for coupling
            if !isempty(marked)
                refine_cells_2d!(amr_solver.refined_grid, marked)
            end
            amr_solver.amr_statistics["total_refinements"] += length(marked)
        catch e
            println("Warning: Advanced AMR refinement failed: $e")
            # Fall back to basic AMR
            update_basic_amr!(amr_solver, state, bodies)
        end
    else
        # Use basic AMR
        update_basic_amr!(amr_solver, state, bodies)
    end
    
    amr_solver.amr_timing["refinement"] += time() - amr_timing_start
    amr_solver.last_amr_step = state.step
end

"""
    update_basic_amr!(amr_solver, state, bodies)

Update AMR using basic refinement algorithms.
"""
function update_basic_amr!(amr_solver::AMRIntegratedSolver, state::SolutionState, bodies)
    # Use basic AMR from adaptive_refinement.jl
    refined_count = adapt_grid!(amr_solver.refined_grid, state, bodies, amr_solver.amr_criteria)
    amr_solver.amr_statistics["total_refinements"] += refined_count
end

"""
    ensure_output_on_original_grid!(amr_solver, state)

Ensure that the solution state is on the original base grid for consistent output.
This is crucial - AMR computation is internal, but output must be on original grid.
"""
function ensure_output_on_original_grid!(amr_solver::AMRIntegratedSolver, state::SolutionState)
    base_grid = amr_solver.refined_grid.base_grid
    
    # Verify that state dimensions match original base grid
    if base_grid.grid_type == TwoDimensional
        expected_u_size = (base_grid.nx + 1, base_grid.nz)
        expected_w_size = (base_grid.nx, base_grid.nz + 1)  # For XZ-plane 2D, we use w not v
        expected_p_size = (base_grid.nx, base_grid.nz)
        
        if size(state.u) != expected_u_size || size(state.w) != expected_w_size || size(state.p) != expected_p_size
            error("AMR solver state is not on original grid! " *
                  "Expected u:$expected_u_size, w:$expected_w_size, p:$expected_p_size, " *
                  "got u:$(size(state.u)), w:$(size(state.w)), p:$(size(state.p))")
        end
    else
        expected_u_size = (base_grid.nx + 1, base_grid.ny, base_grid.nz)
        expected_v_size = (base_grid.nx, base_grid.ny + 1, base_grid.nz)
        expected_w_size = (base_grid.nx, base_grid.ny, base_grid.nz + 1)
        expected_p_size = (base_grid.nx, base_grid.ny, base_grid.nz)
        
        w_size = hasfield(typeof(state), :w) && state.w !== nothing ? size(state.w) : expected_w_size
        
        if size(state.u) != expected_u_size || size(state.v) != expected_v_size || 
           w_size != expected_w_size || size(state.p) != expected_p_size
            error("AMR solver state is not on original 3D grid! Check dimensions.")
        end
    end
    
    # State is verified to be on original grid - ready for output
end

"""
    update_amr_statistics!(amr_solver)

Update AMR performance and usage statistics.
"""
function update_amr_statistics!(amr_solver::AMRIntegratedSolver)
    # Count current refined cells
    base_grid = amr_solver.refined_grid.base_grid
    
    if base_grid.grid_type == TwoDimensional
        amr_solver.amr_statistics["current_refined_cells"] = length(amr_solver.refined_grid.refined_cells_2d)
        
        # Find maximum refinement level used
        if !isempty(amr_solver.refined_grid.refined_cells_2d)
            max_level = maximum(values(amr_solver.refined_grid.refined_cells_2d))
            amr_solver.amr_statistics["max_refinement_level_used"] = max(
                amr_solver.amr_statistics["max_refinement_level_used"], max_level)
        end
    else
        amr_solver.amr_statistics["current_refined_cells"] = length(amr_solver.refined_grid.refined_cells_3d)
        
        # Find maximum refinement level used
        if !isempty(amr_solver.refined_grid.refined_cells_3d)
            max_level = maximum(values(amr_solver.refined_grid.refined_cells_3d))
            amr_solver.amr_statistics["max_refinement_level_used"] = max(
                amr_solver.amr_statistics["max_refinement_level_used"], max_level)
        end
    end
end

"""
    validate_amr_integration(amr_solver)

Comprehensive validation of AMR integration.
"""
function validate_amr_integration(amr_solver::AMRIntegratedSolver)
    println("=== AMR Integration Validation ===")
    validation_passed = true
    
    # Test 1: Basic AMR functionality
    println("1. Testing basic AMR functionality...")
    if !validate_amr_refinement_algorithms(amr_solver.refined_grid, amr_solver.amr_criteria)
        println("   FAIL: Basic AMR validation failed")
        validation_passed = false
    else
        println("   PASS: Basic AMR validation passed")
    end
    
    # Test 2: Solver integration
    println("2. Testing solver integration...")
    if !test_solver_integration(amr_solver)
        println("   FAIL: Solver integration test failed")
        validation_passed = false
    else
        println("   PASS: Solver integration test passed")
    end
    
    # Test 3: Advanced AMR (if available)
    if HAS_ADVANCED_AMR && amr_solver.amr_hierarchy !== nothing
        println("3. Testing advanced AMR...")
        if !test_advanced_amr_integration(amr_solver)
            println("   FAIL: Advanced AMR test failed")
            validation_passed = false
        else
            println("   PASS: Advanced AMR test passed")
        end
    end
    
    # Test 4: MPI AMR (if available)
    if HAS_MPI_AMR && amr_solver.mpi_amr_hierarchy !== nothing
        println("4. Testing MPI AMR...")
        if !test_mpi_amr_integration(amr_solver)
            println("   FAIL: MPI AMR test failed")
            validation_passed = false
        else
            println("   PASS: MPI AMR test passed")
        end
    end
    
    # Test 5: Output integration
    println("5. Testing output integration...")
    if !test_amr_output_integration(amr_solver)
        println("   FAIL: Output integration test failed")
        validation_passed = false
    else
        println("   PASS: Output integration test passed")
    end
    
    if validation_passed
        println("SUCCESS: All AMR integration tests PASSED!")
        print_amr_summary(amr_solver)
    else
        println("ERROR: Some AMR integration tests FAILED")
    end
    
    return validation_passed
end

"""
    test_solver_integration(amr_solver)

Test integration with base solver.
"""
function test_solver_integration(amr_solver::AMRIntegratedSolver)
    # Check that solver has required fields
    base_solver = amr_solver.base_solver
    
    if !hasfield(typeof(base_solver), :bc)
        println("     Base solver missing boundary conditions")
        return false
    end
    
    # Check grid compatibility
    base_grid = if hasfield(typeof(base_solver), :grid)
        base_solver.grid
    elseif hasfield(typeof(base_solver), :local_grid)
        base_solver.local_grid
    else
        println("     Cannot extract grid from base solver")
        return false
    end
    
    if base_grid !== amr_solver.refined_grid.base_grid
        println("     Grid mismatch between solver and AMR system")
        return false
    end
    
    return true
end

"""
    test_advanced_amr_integration(amr_solver)

Test advanced AMR integration.
"""
function test_advanced_amr_integration(amr_solver::AMRIntegratedSolver)
    if amr_solver.amr_hierarchy === nothing
        return false
    end
    
    # Test that advanced AMR hierarchy is properly initialized
    hierarchy = amr_solver.amr_hierarchy
    if !hasfield(typeof(hierarchy), :base_level) || hierarchy.base_level === nothing
        println("     Advanced AMR hierarchy not properly initialized")
        return false
    end
    
    return true
end

"""
    test_mpi_amr_integration(amr_solver)

Test MPI AMR integration.
"""
function test_mpi_amr_integration(amr_solver::AMRIntegratedSolver)
    if amr_solver.mpi_amr_hierarchy === nothing
        return false
    end
    
    # Test that MPI AMR is properly set up
    mpi_hierarchy = amr_solver.mpi_amr_hierarchy
    if !hasfield(typeof(mpi_hierarchy), :mpi_comm) || mpi_hierarchy.mpi_comm === nothing
        println("     MPI AMR hierarchy not properly initialized")
        return false
    end
    
    return true
end

"""
    test_amr_output_integration(amr_solver)

Test AMR output integration.
"""
function test_amr_output_integration(amr_solver::AMRIntegratedSolver)
    # Test that output functions work
    try
        metadata = create_amr_output_metadata(amr_solver.refined_grid)
        if !isa(metadata, Dict) || isempty(metadata)
            println("     AMR output metadata creation failed")
            return false
        end
    catch e
        println("     AMR output integration error: $e")
        return false
    end
    
    return true
end

"""
    print_amr_summary(amr_solver)

Print summary of AMR configuration and performance.
"""
function print_amr_summary(amr_solver::AMRIntegratedSolver)
    println("\n=== AMR Configuration Summary ===")
    
    base_grid = amr_solver.refined_grid.base_grid
    println("Grid type: $(base_grid.grid_type)")
    
    if base_grid.grid_type == TwoDimensional
        println("Base grid size: $(base_grid.nx) × $(base_grid.nz) (XZ plane)")
    else
        println("Base grid size: $(base_grid.nx) × $(base_grid.ny) × $(base_grid.nz)")
    end
    
    println("AMR enabled: $(amr_solver.amr_enabled)")
    println("AMR frequency: every $(amr_solver.amr_frequency) steps")
    println("Max refinement level: $(amr_solver.amr_criteria.max_refinement_level)")
    println("Min grid size: $(amr_solver.amr_criteria.min_grid_size)")
    
    println("\n=== AMR Features Available ===")
    println("Advanced AMR (v2): $(HAS_ADVANCED_AMR ? "YES" : "NO")")
    println("MPI AMR: $(HAS_MPI_AMR ? "YES" : "NO")")
    println("Advanced AMR active: $(amr_solver.amr_hierarchy !== nothing ? "YES" : "NO")")
    println("MPI AMR active: $(amr_solver.mpi_amr_hierarchy !== nothing ? "YES" : "NO")")
    
    println("\n=== AMR Statistics ===")
    for (key, value) in amr_solver.amr_statistics
        println("$key: $value")
    end
    
    println("\n=== AMR Timing ===")
    total_amr_time = sum(values(amr_solver.amr_timing))
    for (key, value) in amr_solver.amr_timing
        percentage = total_amr_time > 0 ? 100 * value / total_amr_time : 0.0
        println("$key: $(round(value, digits=6))s ($(round(percentage, digits=1))%)")
    end
    
    println("Total AMR time: $(round(total_amr_time, digits=6))s")
    println("================================")
end

# Export main AMR integration functions
export AMRIntegratedSolver, create_amr_integrated_solver, amr_solve_step!
export validate_amr_integration, print_amr_summary
export HAS_ADVANCED_AMR, HAS_MPI_AMR
