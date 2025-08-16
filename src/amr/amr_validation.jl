"""
AMR Validation and Testing

This module provides comprehensive validation functions for the adaptive mesh 
refinement implementation to ensure correctness and consistency.
"""

"""
    validate_amr_refinement_algorithms(refined_grid, criteria)

Comprehensive validation of AMR refinement and coarsening algorithms.
"""
function validate_amr_refinement_algorithms(refined_grid::RefinedGrid, criteria::AdaptiveRefinementCriteria)
    println("=== AMR Refinement Algorithm Validation ===")
    
    base_grid = refined_grid.base_grid
    validation_passed = true
    
    # Test 1: Coordinate system consistency
    println("1. Testing coordinate system consistency...")
    if base_grid.grid_type == TwoDimensional
        # Check XZ plane consistency
        expected_dims = (base_grid.nx, base_grid.nz)
        if !check_2d_coordinate_consistency(refined_grid, expected_dims)
            println("   FAIL: 2D XZ plane coordinate system inconsistency detected")
            validation_passed = false
        else
            println("   PASS: 2D XZ plane coordinate system is consistent")
        end
    else
        # Check 3D consistency
        expected_dims = (base_grid.nx, base_grid.ny, base_grid.nz)
        if !check_3d_coordinate_consistency(refined_grid, expected_dims)
            println("   FAIL: 3D coordinate system inconsistency detected")
            validation_passed = false
        else
            println("   PASS: 3D coordinate system is consistent")
        end
    end
    
    # Test 2: Refinement level constraints
    println("2. Testing refinement level constraints...")
    if !validate_refinement_level_constraints(refined_grid, criteria)
        println("   FAIL: Refinement level constraints violated")
        validation_passed = false
    else
        println("   PASS: Refinement level constraints satisfied")
    end
    
    # Test 3: Grid size constraints
    println("3. Testing grid size constraints...")
    if !validate_grid_size_constraints(refined_grid, criteria)
        println("   FAIL: Grid size constraints violated")
        validation_passed = false
    else
        println("   PASS: Grid size constraints satisfied")
    end
    
    # Test 4: Data structure integrity
    println("4. Testing data structure integrity...")
    if !validate_data_structure_integrity(refined_grid)
        println("   FAIL: Data structure integrity issues found")
        validation_passed = false
    else
        println("   PASS: Data structure integrity verified")
    end
    
    # Test 5: Interpolation weight consistency
    println("5. Testing interpolation weight consistency...")
    if !validate_interpolation_weights(refined_grid)
        println("   FAIL: Interpolation weight inconsistencies found")
        validation_passed = false
    else
        println("   PASS: Interpolation weights are consistent")
    end
    
    if validation_passed
        println("SUCCESS: All AMR validation tests PASSED!")
    else
        println("ERROR: Some AMR validation tests FAILED - check implementation")
    end
    
    return validation_passed
end

"""
    check_2d_coordinate_consistency(refined_grid, expected_dims)

Check coordinate system consistency for 2D XZ plane.
"""
function check_2d_coordinate_consistency(refined_grid::RefinedGrid, expected_dims::Tuple{Int,Int})
    base_grid = refined_grid.base_grid
    nx_expected, nz_expected = expected_dims
    
    # Check base grid dimensions
    if base_grid.nx != nx_expected || base_grid.nz != nz_expected
        println("     Base grid dimensions mismatch: expected ($nx_expected, $nz_expected), got ($(base_grid.nx), $(base_grid.nz))")
        return false
    end
    
    # Check coordinate arrays
    if length(base_grid.x) != nx_expected || length(base_grid.z) != nz_expected
        println("     Coordinate array sizes mismatch")
        return false
    end
    
    # Check refined cell dictionaries use correct tuple types
    for (cell_idx, level) in refined_grid.refined_cells_2d
        if !isa(cell_idx, Tuple{Int,Int}) || length(cell_idx) != 2
            println("     Invalid 2D cell index type: $cell_idx")
            return false
        end
        
        i, j = cell_idx
        if i < 1 || i > nx_expected || j < 1 || j > nz_expected
            println("     Cell index out of bounds: ($i, $j)")
            return false
        end
    end
    
    return true
end

"""
    check_3d_coordinate_consistency(refined_grid, expected_dims)

Check coordinate system consistency for 3D.
"""
function check_3d_coordinate_consistency(refined_grid::RefinedGrid, expected_dims::Tuple{Int,Int,Int})
    base_grid = refined_grid.base_grid
    nx_expected, ny_expected, nz_expected = expected_dims
    
    # Check base grid dimensions
    if base_grid.nx != nx_expected || base_grid.ny != ny_expected || base_grid.nz != nz_expected
        println("     Base grid dimensions mismatch")
        return false
    end
    
    # Check coordinate arrays
    if length(base_grid.x) != nx_expected || length(base_grid.y) != ny_expected || length(base_grid.z) != nz_expected
        println("     Coordinate array sizes mismatch")
        return false
    end
    
    # Check refined cell dictionaries use correct tuple types
    for (cell_idx, level) in refined_grid.refined_cells_3d
        if !isa(cell_idx, Tuple{Int,Int,Int}) || length(cell_idx) != 3
            println("     Invalid 3D cell index type: $cell_idx")
            return false
        end
        
        i, j, k = cell_idx
        if i < 1 || i > nx_expected || j < 1 || j > ny_expected || k < 1 || k > nz_expected
            println("     Cell index out of bounds: ($i, $j, $k)")
            return false
        end
    end
    
    return true
end

"""
    validate_refinement_level_constraints(refined_grid, criteria)

Validate that refinement levels respect maximum constraints.
"""
function validate_refinement_level_constraints(refined_grid::RefinedGrid, criteria::AdaptiveRefinementCriteria)
    base_grid = refined_grid.base_grid
    
    if base_grid.grid_type == TwoDimensional
        # Check 2D refinement levels
        for (cell_idx, level) in refined_grid.refined_cells_2d
            if level < 0 || level > criteria.max_refinement_level
                println("     Invalid refinement level $level for cell $cell_idx (max: $(criteria.max_refinement_level))")
                return false
            end
        end
    else
        # Check 3D refinement levels
        for (cell_idx, level) in refined_grid.refined_cells_3d
            if level < 0 || level > criteria.max_refinement_level
                println("     Invalid refinement level $level for cell $cell_idx (max: $(criteria.max_refinement_level))")
                return false
            end
        end
    end
    
    return true
end

"""
    validate_grid_size_constraints(refined_grid, criteria)

Validate that refined grid sizes respect minimum constraints.
"""
function validate_grid_size_constraints(refined_grid::RefinedGrid, criteria::AdaptiveRefinementCriteria)
    base_grid = refined_grid.base_grid
    
    if base_grid.grid_type == TwoDimensional
        # Check 2D grid sizes
        for (cell_idx, level) in refined_grid.refined_cells_2d
            refined_dx = base_grid.dx / (2^level)
            refined_dz = base_grid.dz / (2^level)
            
            if refined_dx < criteria.min_grid_size || refined_dz < criteria.min_grid_size
                println("     Grid size too small for cell $cell_idx: dx=$refined_dx, dz=$refined_dz (min: $(criteria.min_grid_size))")
                return false
            end
        end
    else
        # Check 3D grid sizes
        for (cell_idx, level) in refined_grid.refined_cells_3d
            refined_dx = base_grid.dx / (2^level)
            refined_dy = base_grid.dy / (2^level)
            refined_dz = base_grid.dz / (2^level)
            
            if refined_dx < criteria.min_grid_size || refined_dy < criteria.min_grid_size || refined_dz < criteria.min_grid_size
                println("     Grid size too small for cell $cell_idx: dx=$refined_dx, dy=$refined_dy, dz=$refined_dz (min: $(criteria.min_grid_size))")
                return false
            end
        end
    end
    
    return true
end

"""
    validate_data_structure_integrity(refined_grid)

Validate that all data structures are consistent with each other.
"""
function validate_data_structure_integrity(refined_grid::RefinedGrid)
    base_grid = refined_grid.base_grid
    
    if base_grid.grid_type == TwoDimensional
        # Check that all 2D dictionaries have consistent keys
        cells_2d = Set(keys(refined_grid.refined_cells_2d))
        grids_2d = Set(keys(refined_grid.refined_grids_2d))
        weights_2d = Set(keys(refined_grid.interpolation_weights_2d))
        
        if cells_2d != grids_2d
            println("     Mismatch between refined_cells_2d and refined_grids_2d keys")
            return false
        end
        
        if cells_2d != weights_2d
            println("     Mismatch between refined_cells_2d and interpolation_weights_2d keys")
            return false
        end
        
        # Check that 3D dictionaries are empty for 2D case
        if !isempty(refined_grid.refined_cells_3d) || !isempty(refined_grid.refined_grids_3d) || !isempty(refined_grid.interpolation_weights_3d)
            println("     3D data structures should be empty for 2D grid")
            return false
        end
    else
        # Check that all 3D dictionaries have consistent keys
        cells_3d = Set(keys(refined_grid.refined_cells_3d))
        grids_3d = Set(keys(refined_grid.refined_grids_3d))
        weights_3d = Set(keys(refined_grid.interpolation_weights_3d))
        
        if cells_3d != grids_3d
            println("     Mismatch between refined_cells_3d and refined_grids_3d keys")
            return false
        end
        
        if cells_3d != weights_3d
            println("     Mismatch between refined_cells_3d and interpolation_weights_3d keys")
            return false
        end
        
        # Check that 2D dictionaries are empty for 3D case
        if !isempty(refined_grid.refined_cells_2d) || !isempty(refined_grid.refined_grids_2d) || !isempty(refined_grid.interpolation_weights_2d)
            println("     2D data structures should be empty for 3D grid")
            return false
        end
    end
    
    return true
end

"""
    validate_interpolation_weights(refined_grid)

Validate that interpolation weights sum to 1 and use valid indices.
"""
function validate_interpolation_weights(refined_grid::RefinedGrid)
    base_grid = refined_grid.base_grid
    tolerance = 1e-10
    
    if base_grid.grid_type == TwoDimensional
        # Check 2D interpolation weights
        for (cell_idx, weights) in refined_grid.interpolation_weights_2d
            total_weight = 0.0
            
            for ((i_base, j_base), weight) in weights
                total_weight += weight
                
                # Check that base indices are valid
                if i_base < 1 || i_base > base_grid.nx || j_base < 1 || j_base > base_grid.nz
                    println("     Invalid base grid indices in interpolation weights: ($i_base, $j_base)")
                    return false
                end
                
                # Check that weight is positive
                if weight <= 0.0
                    println("     Non-positive interpolation weight: $weight")
                    return false
                end
            end
            
            # Check that weights sum to approximately 1
            if abs(total_weight - 1.0) > tolerance
                println("     Interpolation weights don't sum to 1.0: $total_weight")
                return false
            end
        end
    else
        # Check 3D interpolation weights
        for (cell_idx, weights) in refined_grid.interpolation_weights_3d
            total_weight = 0.0
            
            for ((i_base, j_base, k_base), weight) in weights
                total_weight += weight
                
                # Check that base indices are valid
                if i_base < 1 || i_base > base_grid.nx || j_base < 1 || j_base > base_grid.ny || k_base < 1 || k_base > base_grid.nz
                    println("     Invalid base grid indices in interpolation weights: ($i_base, $j_base, $k_base)")
                    return false
                end
                
                # Check that weight is positive
                if weight <= 0.0
                    println("     Non-positive interpolation weight: $weight")
                    return false
                end
            end
            
            # Check that weights sum to approximately 1
            if abs(total_weight - 1.0) > tolerance
                println("     Interpolation weights don't sum to 1.0: $total_weight")
                return false
            end
        end
    end
    
    return true
end

"""
    test_amr_refinement_coarsening_cycle()

Test a complete refinement-coarsening cycle.
"""
function test_amr_refinement_coarsening_cycle()
    println("=== Testing AMR Refinement-Coarsening Cycle ===")
    
    # Create a simple test grid
    test_grid = StaggeredGrid2D(16, 16, 1.0, 1.0)  # 16x16 XZ plane grid
    refined_grid = RefinedGrid(test_grid)
    criteria = AdaptiveRefinementCriteria(max_refinement_level=2, min_grid_size=0.01)
    
    println("1. Initial state validation...")
    if !validate_amr_refinement_algorithms(refined_grid, criteria)
        println("FAIL: Initial state validation failed")
        return false
    end
    
    println("2. Testing refinement...")
    # Mark some cells for refinement
    test_cells_2d = [(8, 8), (9, 8), (8, 9), (9, 9)]  # 2x2 block in center
    refine_cells_2d!(refined_grid, test_cells_2d)
    
    if !validate_amr_refinement_algorithms(refined_grid, criteria)
        println("FAIL: Post-refinement validation failed")
        return false
    end
    
    println("3. Testing coarsening...")
    # Coarsen the same cells
    coarsen_cells_2d!(refined_grid, test_cells_2d)
    
    if !validate_amr_refinement_algorithms(refined_grid, criteria)
        println("FAIL: Post-coarsening validation failed")
        return false
    end
    
    println("SUCCESS: AMR refinement-coarsening cycle test PASSED!")
    return true
end

# Export validation functions
export validate_amr_refinement_algorithms, test_amr_refinement_coarsening_cycle