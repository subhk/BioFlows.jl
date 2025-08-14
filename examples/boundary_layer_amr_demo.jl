"""
Boundary Layer Adaptive Mesh Refinement Demonstration

This example demonstrates specialized AMR for resolving boundary layers around solid bodies:
1. Cylinder at different Reynolds numbers
2. Y⁺-based refinement for wall-bounded flows
3. Anisotropic refinement for thin boundary layers
4. Comparison with uniform grid approaches
"""

using BioFlows
using Printf

function demo_boundary_layer_amr()
    println("="^70)
    println("Boundary Layer Adaptive Mesh Refinement Demo")
    println("="^70)
    
    # Test different Reynolds numbers
    reynolds_numbers = [100.0, 1000.0, 10000.0, 100000.0]
    
    for (i, Re) in enumerate(reynolds_numbers)
        println("\n", "="^50)
        println("Test Case $i: Reynolds Number = $(@sprintf(\"%.0f\", Re))")
        println("="^50)
        
        test_cylinder_boundary_layer(Re)
        
        if i < length(reynolds_numbers)
            println("\nPress Enter to continue to next Reynolds number...")
            readline()
        end
    end
    
    println("\n" * "="^70)
    println("Boundary Layer AMR Demo Completed!")
    println("="^70)
end

function test_cylinder_boundary_layer(reynolds_number::Float64)
    # Problem setup: Flow around cylinder
    cylinder_diameter = 1.0
    cylinder_radius = cylinder_diameter / 2.0
    
    # Domain size based on Reynolds number (larger domains for higher Re)
    domain_factor = max(10.0, 5.0 * log10(reynolds_number))
    Lx = domain_factor * cylinder_diameter
    Ly = domain_factor * cylinder_diameter
    
    # Base grid resolution
    base_resolution = max(32, Int(round(16 * sqrt(reynolds_number / 100))))
    nx_base = base_resolution
    ny_base = base_resolution
    
    println("Problem Setup:")
    println("  Cylinder diameter: $cylinder_diameter")
    println("  Domain size: $(Lx) × $(Ly)")
    println("  Base grid: $(nx_base) × $(ny_base)")
    println("  Reynolds number: $(@sprintf(\"%.0f\", reynolds_number))")
    
    # Create base grid
    base_grid = StaggeredGrid2D(nx_base, ny_base, Lx, Ly)
    
    # Create cylinder
    circle = Circle(cylinder_radius)
    cylinder = RigidBody(circle, [Lx/3, Ly/2], 0.0, StationaryMotion())
    bodies = RigidBodyCollection()
    push!(bodies, cylinder)
    
    # Fluid properties
    ρ_ref = 1.0
    U_ref = 1.0  # Reference velocity
    μ = ρ_ref * U_ref * cylinder_diameter / reynolds_number
    
    fluid = FluidProperties(μ, ConstantDensity(ρ_ref), reynolds_number)
    
    println("  Fluid properties:")
    println("    Density: $ρ_ref")
    println("    Viscosity: $(@sprintf(\"%.6f\", μ))")
    println("    Reference velocity: $U_ref")
    
    # Create boundary layer AMR criteria
    bl_criteria = create_reynolds_appropriate_criteria(reynolds_number)
    
    println("\nBoundary Layer Criteria:")
    println("  Target y⁺: $(bl_criteria.target_y_plus)")
    println("  Max BL levels: $(bl_criteria.max_bl_levels)")
    println("  Est. BL thickness: $(@sprintf(\"%.6f\", bl_criteria.bl_thickness_factor / sqrt(reynolds_number)))")
    println("  Max wall distance: $(@sprintf(\"%.6f\", bl_criteria.max_wall_distance))")
    println("  Anisotropic refinement: $(bl_criteria.enable_anisotropic)")
    
    # Create initial solution (potential flow around cylinder)
    state = create_cylinder_initial_solution(base_grid, cylinder, U_ref)
    
    # Create AMR hierarchy
    max_levels = determine_max_levels_for_reynolds(reynolds_number)
    amr_hierarchy = AMRHierarchy(base_grid; 
                                max_level=max_levels,
                                regrid_interval=5)
    
    println("\nAMR Configuration:")
    println("  Maximum levels: $max_levels")
    println("  Initial cells: $(amr_hierarchy.total_cells)")
    
    # Perform boundary layer-aware refinement
    println("\nPerforming Boundary Layer-Aware Refinement:")
    println("-"^45)
    
    total_refined = 0
    refinement_iterations = 3
    
    for iteration = 1:refinement_iterations
        println("  Iteration $iteration:")
        
        # Compute boundary layer indicators
        bl_indicators = compute_boundary_layer_indicators(
            base_grid, state, bodies, bl_criteria, fluid
        )
        
        # Analyze indicators
        analyze_bl_indicators(bl_indicators, base_grid, bodies)
        
        # Perform refinement
        refined_count = refine_for_boundary_layers!(
            amr_hierarchy, state, bodies, fluid, bl_criteria
        )
        
        total_refined += refined_count
        
        println("    Refined cells: $refined_count")
        println("    Total cells: $(amr_hierarchy.total_cells + refined_count * 3)") # Approximate
        
        if refined_count == 0
            println("    No more refinement needed")
            break
        end
    end
    
    # Estimate boundary layer resolution
    println("\nBoundary Layer Resolution Analysis:")
    println("-"^40)
    
    analyze_boundary_layer_resolution(base_grid, bodies, bl_criteria, fluid, reynolds_number)
    
    # Compare with uniform grid requirements
    println("\nComparison with Uniform Grid:")
    println("-"^35)
    
    compare_with_uniform_grid(base_grid, bl_criteria, reynolds_number, total_refined)
    
    # Performance assessment
    println("\nPerformance Assessment:")
    println("-"^25)
    
    assess_bl_amr_performance(amr_hierarchy, reynolds_number, total_refined)
    
    return amr_hierarchy, state, bodies, bl_criteria
end

function create_reynolds_appropriate_criteria(reynolds_number::Float64)
    """Create boundary layer criteria appropriate for the Reynolds number."""
    
    if reynolds_number < 500
        # Low Reynolds - can resolve everything with modest refinement
        return BoundaryLayerAMRCriteria(reynolds_number;
            target_y_plus=0.5,
            max_bl_levels=2,
            enable_anisotropic=false,
            wall_normal_stretch_ratio=1.1)
            
    elseif reynolds_number < 5000
        # Moderate Reynolds - need careful boundary layer resolution
        return BoundaryLayerAMRCriteria(reynolds_number;
            target_y_plus=1.0,
            max_bl_levels=3,
            enable_anisotropic=true,
            wall_normal_stretch_ratio=1.2,
            aspect_ratio_limit=5.0)
            
    elseif reynolds_number < 50000
        # High Reynolds - requires aggressive BL refinement
        return BoundaryLayerAMRCriteria(reynolds_number;
            target_y_plus=1.0,
            max_bl_levels=4,
            enable_anisotropic=true,
            wall_normal_stretch_ratio=1.3,
            aspect_ratio_limit=10.0,
            velocity_boundary_threshold=20.0)
            
    else
        # Very high Reynolds - wall functions may be needed
        return BoundaryLayerAMRCriteria(reynolds_number;
            target_y_plus=30.0,  # Wall function range
            max_bl_levels=5,
            enable_anisotropic=true,
            wall_normal_stretch_ratio=1.5,
            aspect_ratio_limit=20.0,
            velocity_boundary_threshold=50.0)
    end
end

function determine_max_levels_for_reynolds(reynolds_number::Float64)
    """Determine appropriate maximum refinement levels based on Reynolds number."""
    
    if reynolds_number < 1000
        return 3
    elseif reynolds_number < 10000
        return 4
    elseif reynolds_number < 100000
        return 5
    else
        return 6
    end
end

function create_cylinder_initial_solution(grid::StaggeredGrid, cylinder::RigidBody, U_ref::Float64)
    """Create initial potential flow solution around cylinder."""
    
    nx, ny = grid.nx, grid.ny
    state = SolutionState2D(nx, ny)
    
    cylinder_center = cylinder.position
    cylinder_radius = cylinder.geometry.radius
    
    # Potential flow around cylinder: ψ = U∞(r - a²/r)sin(θ)
    # u = ∂ψ/∂y, v = -∂ψ/∂x
    
    # u-velocity (at x-faces)
    for j = 1:ny, i = 1:nx+1
        x = grid.xu[i]
        y_center = grid.y[j]
        
        # Relative to cylinder center
        dx = x - cylinder_center[1]
        dy = y_center - cylinder_center[2]
        
        r = sqrt(dx^2 + dy^2)
        
        if r > cylinder_radius * 1.01  # Outside cylinder
            theta = atan(dy, dx)
            
            # Potential flow u-component
            u_potential = U_ref * (1 - (cylinder_radius/r)^2 * (1 - 2*sin(theta)^2))
            state.u[i, j] = u_potential
        else
            state.u[i, j] = 0.0  # No-slip at wall
        end
    end
    
    # v-velocity (at y-faces)
    for j = 1:ny+1, i = 1:nx
        x_center = grid.x[i]
        y = grid.yv[j]
        
        dx = x_center - cylinder_center[1]
        dy = y - cylinder_center[2]
        
        r = sqrt(dx^2 + dy^2)
        
        if r > cylinder_radius * 1.01
            theta = atan(dy, dx)
            
            # Potential flow v-component
            v_potential = -U_ref * (cylinder_radius/r)^2 * sin(2*theta)
            state.v[i, j] = v_potential
        else
            state.v[i, j] = 0.0  # No-slip at wall
        end
    end
    
    # Pressure field (from Bernoulli equation)
    for j = 1:ny, i = 1:nx
        x = grid.x[i]
        y = grid.y[j]
        
        dx = x - cylinder_center[1]
        dy = y - cylinder_center[2]
        r = sqrt(dx^2 + dy^2)
        
        if r > cylinder_radius * 1.01
            theta = atan(dy, dx)
            
            # Pressure from potential flow theory
            u_local = U_ref * (1 - (cylinder_radius/r)^2 * (1 - 2*sin(theta)^2))
            v_local = -U_ref * (cylinder_radius/r)^2 * sin(2*theta)
            
            # Bernoulli: p + 0.5*ρ*|u|² = constant
            velocity_squared = u_local^2 + v_local^2
            state.p[i, j] = 0.5 * (U_ref^2 - velocity_squared)  # ρ = 1
        else
            state.p[i, j] = -0.5 * U_ref^2  # Stagnation pressure
        end
    end
    
    return state
end

function analyze_bl_indicators(indicators::Matrix{Float64}, grid::StaggeredGrid,
                              bodies::Union{RigidBodyCollection, FlexibleBodyCollection})
    """Analyze boundary layer refinement indicators."""
    
    wall_distance = compute_wall_distance_field(grid, bodies)
    
    # Statistics in boundary layer region
    bl_region_mask = wall_distance .< 0.1  # Within 10% of domain
    bl_indicators = indicators[bl_region_mask]
    
    if length(bl_indicators) > 0
        max_bl = maximum(bl_indicators)
        mean_bl = sum(bl_indicators) / length(bl_indicators)
        cells_flagged = count(bl_indicators .> 0.5)
        
        println("    BL indicator stats:")
        println("      Max in BL region: $(@sprintf(\"%.3f\", max_bl))")
        println("      Mean in BL region: $(@sprintf(\"%.3f\", mean_bl))")
        println("      BL cells flagged: $cells_flagged ($(@sprintf(\"%.1f\", 100*cells_flagged/length(bl_indicators)))%)")
    end
end

function analyze_boundary_layer_resolution(grid::StaggeredGrid,
                                         bodies::Union{RigidBodyCollection, FlexibleBodyCollection},
                                         bl_criteria::BoundaryLayerAMRCriteria,
                                         fluid::FluidProperties,
                                         reynolds_number::Float64)
    """Analyze how well the boundary layer is resolved."""
    
    # Estimate required resolution
    cylinder = bodies.bodies[1]  # Assume first body is cylinder
    cylinder_radius = cylinder.geometry.radius
    
    # Boundary layer thickness at 90° (side of cylinder)
    if reynolds_number > 1e5
        # Turbulent BL
        bl_thickness = 0.37 * cylinder_radius / (reynolds_number^0.2)
    else
        # Laminar BL  
        bl_thickness = 3.5 * cylinder_radius / sqrt(reynolds_number)
    end
    
    # Current grid resolution near wall
    min_grid_spacing = min(grid.dx, grid.dy)
    
    # Y⁺ estimation
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
    end
    ν = fluid.μ / ρ
    
    # Estimate friction velocity (rough approximation)
    u_tau_estimate = 0.05  # Typical value, would need actual calculation
    y_plus_current = min_grid_spacing * u_tau_estimate / ν
    
    println("  Estimated BL thickness: $(@sprintf(\"%.6f\", bl_thickness))")
    println("  Current grid spacing: $(@sprintf(\"%.6f\", min_grid_spacing))")
    println("  Points across BL: $(@sprintf(\"%.1f\", bl_thickness / min_grid_spacing))")
    println("  Current y⁺ estimate: $(@sprintf(\"%.2f\", y_plus_current))")
    println("  Target y⁺: $(bl_criteria.target_y_plus)")
    
    # Resolution assessment
    points_across_bl = bl_thickness / min_grid_spacing
    if points_across_bl < 5
        println("  ⚠  WARNING: Insufficient BL resolution (< 5 points)")
    elseif points_across_bl < 10
        println("  ⚠  Marginal BL resolution (< 10 points)")
    elseif points_across_bl < 20
        println("  ✓ Adequate BL resolution")
    else
        println("  ✓ Excellent BL resolution")
    end
    
    if y_plus_current > bl_criteria.target_y_plus * 2
        println("  ⚠  y⁺ too large - need finer near-wall resolution")
    elseif y_plus_current < bl_criteria.target_y_plus / 2
        println("  ✓ y⁺ well resolved (possibly over-resolved)")
    else
        println("  ✓ y⁺ appropriately resolved")
    end
end

function compare_with_uniform_grid(base_grid::StaggeredGrid,
                                  bl_criteria::BoundaryLayerAMRCriteria,
                                  reynolds_number::Float64,
                                  amr_refined_cells::Int)
    """Compare AMR approach with equivalent uniform grid."""
    
    # Estimate required uniform grid resolution
    target_y_plus = bl_criteria.target_y_plus
    
    # Required near-wall spacing
    cylinder_radius = 0.5  # From problem setup
    if reynolds_number > 1e5
        bl_thickness = 0.37 * cylinder_radius / (reynolds_number^0.2)
    else
        bl_thickness = 3.5 * cylinder_radius / sqrt(reynolds_number)
    end
    
    # Required wall-normal spacing for target y⁺
    required_wall_spacing = target_y_plus * 1e-5  # Rough estimate
    
    # Domain size
    domain_area = (base_grid.Lx) * (base_grid.Ly)
    current_cells = base_grid.nx * base_grid.ny
    
    # Uniform grid requirement
    uniform_spacing = min(required_wall_spacing, min(base_grid.dx, base_grid.dy) / 4)
    uniform_nx = Int(ceil(base_grid.Lx / uniform_spacing))
    uniform_ny = Int(ceil(base_grid.Ly / uniform_spacing))
    uniform_cells = uniform_nx * uniform_ny
    
    # AMR cells
    amr_total_cells = current_cells + amr_refined_cells * 3  # Approximate
    
    println("  Uniform grid requirement:")
    println("    Required spacing: $(@sprintf(\"%.7f\", uniform_spacing))")
    println("    Required cells: $(uniform_nx) × $(uniform_ny) = $uniform_cells")
    println("    Memory: $(@sprintf(\"%.1f\", uniform_cells * 3 * 8 / 1e6)) MB")
    
    println("  AMR approach:")
    println("    Total cells: ~$amr_total_cells")
    println("    Memory: ~$(@sprintf(\"%.1f\", amr_total_cells * 3 * 8 / 1e6)) MB")
    
    if uniform_cells > 0
        savings_factor = uniform_cells / amr_total_cells
        println("  AMR savings: $(@sprintf(\"%.1f\", savings_factor))× fewer cells")
        
        if savings_factor > 100
            println("  ✓ Excellent savings - AMR essential for this Re")
        elseif savings_factor > 10
            println("  ✓ Good savings - AMR recommended")
        elseif savings_factor > 2
            println("  ✓ Moderate savings - AMR beneficial")
        else
            println("  ⚠ Limited savings - consider uniform grid")
        end
    end
end

function assess_bl_amr_performance(amr_hierarchy::AMRHierarchy,
                                  reynolds_number::Float64,
                                  refined_cells::Int)
    """Assess boundary layer AMR performance characteristics."""
    
    base_cells = amr_hierarchy.total_cells
    effective_cells = base_cells + refined_cells * 3  # Approximation
    
    println("  Refinement efficiency:")
    println("    Base cells: $base_cells")
    println("    Refined cells: $refined_cells")
    println("    Effective cells: ~$effective_cells")
    println("    Refinement ratio: $(@sprintf(\"%.2f\", effective_cells / base_cells))×")
    
    # Computational cost estimate
    base_cost = base_cells
    amr_cost = effective_cells * 1.3  # 30% overhead for AMR operations
    cost_factor = amr_cost / base_cost
    
    println("  Computational cost:")
    println("    Base cost factor: 1.0")
    println("    AMR cost factor: $(@sprintf(\"%.2f\", cost_factor))")
    
    # Reynolds-specific assessment
    if reynolds_number > 1e5
        println("  High-Re assessment:")
        if refined_cells < base_cells / 10
            println("    ⚠ May need more aggressive refinement")
        else
            println("    ✓ Appropriate refinement level")
        end
    elseif reynolds_number > 1e3
        println("  Moderate-Re assessment:")
        if refined_cells < base_cells / 20
            println("    ⚠ Consider more refinement near walls")
        else
            println("    ✓ Good refinement strategy")
        end
    else
        println("  Low-Re assessment:")
        println("    ✓ Boundary layers naturally thick - less refinement needed")
    end
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_boundary_layer_amr()
end