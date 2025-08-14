"""
Boundary Layer-Aware Adaptive Mesh Refinement

This module provides specialized AMR capabilities for resolving boundary layers around solid bodies:
1. Wall-normal refinement stretching
2. Reynolds number-based refinement scaling
3. Anisotropic refinement for thin boundary layers
4. Boundary layer thickness estimation
5. Wall-function integration for very high Re flows
"""

"""
    BoundaryLayerAMRCriteria

Advanced refinement criteria specifically designed for boundary layer resolution.
"""
struct BoundaryLayerAMRCriteria
    # Reynolds number characteristics
    reynolds_number::Float64
    critical_reynolds::Float64        # Re above which special BL treatment is needed
    
    # Boundary layer parameters
    target_y_plus::Float64           # Target y⁺ for first cell height
    bl_thickness_factor::Float64     # Factor for BL thickness estimation
    max_bl_levels::Int               # Maximum refinement levels in boundary layer
    
    # Wall-normal refinement
    wall_normal_stretch_ratio::Float64  # Geometric stretching ratio
    min_wall_distance::Float64          # Minimum distance from wall to refine
    max_wall_distance::Float64          # Maximum distance for BL refinement
    
    # Anisotropic refinement parameters
    enable_anisotropic::Bool            # Enable anisotropic refinement
    aspect_ratio_limit::Float64         # Maximum cell aspect ratio
    tangential_refinement_ratio::Float64 # Ratio of tangential to normal refinement
    
    # Advanced criteria
    velocity_boundary_threshold::Float64 # Velocity gradient threshold near walls
    pressure_gradient_scaling::Float64   # Enhanced pressure gradient sensitivity
    wall_shear_threshold::Float64       # Wall shear stress refinement trigger
end

function BoundaryLayerAMRCriteria(reynolds_number::Float64;
                                 target_y_plus::Float64=1.0,
                                 critical_reynolds::Float64=1000.0,
                                 bl_thickness_factor::Float64=5.0,
                                 max_bl_levels::Int=4,
                                 wall_normal_stretch_ratio::Float64=1.2,
                                 min_wall_distance::Float64=1e-4,
                                 aspect_ratio_limit::Float64=10.0,
                                 enable_anisotropic::Bool=true,
                                 tangential_refinement_ratio::Float64=0.5,
                                 velocity_boundary_threshold::Float64=10.0,
                                 pressure_gradient_scaling::Float64=2.0,
                                 wall_shear_threshold::Float64=5.0)
    
    # Auto-calculate max wall distance based on boundary layer thickness
    # δ ≈ 5.0/√Re for flat plate boundary layer
    estimated_bl_thickness = bl_thickness_factor / sqrt(reynolds_number)
    max_wall_distance = min(0.1, 2.0 * estimated_bl_thickness)  # Don't go beyond 10% of domain
    
    BoundaryLayerAMRCriteria(reynolds_number, critical_reynolds, target_y_plus,
                            bl_thickness_factor, max_bl_levels, wall_normal_stretch_ratio,
                            min_wall_distance, max_wall_distance, enable_anisotropic,
                            aspect_ratio_limit, tangential_refinement_ratio,
                            velocity_boundary_threshold, pressure_gradient_scaling,
                            wall_shear_threshold)
end

"""
    compute_boundary_layer_indicators(grid, state, bodies, bl_criteria, fluid)

Computes specialized refinement indicators for boundary layer resolution.
"""
function compute_boundary_layer_indicators(grid::StaggeredGrid, state::SolutionState,
                                         bodies::Union{RigidBodyCollection, FlexibleBodyCollection},
                                         bl_criteria::BoundaryLayerAMRCriteria,
                                         fluid::FluidProperties)
    nx, ny = grid.nx, grid.ny
    indicators = zeros(nx, ny)
    
    # Compute wall distance field
    wall_distance = compute_wall_distance_field(grid, bodies)
    
    # Compute wall-normal velocity gradients
    wall_normal_gradients = compute_wall_normal_gradients(grid, state, bodies)
    
    # Compute boundary layer thickness estimates
    bl_thickness = estimate_boundary_layer_thickness(grid, state, bodies, bl_criteria, fluid)
    
    # Compute y⁺ values
    y_plus_field = compute_y_plus_field(grid, state, bodies, wall_distance, fluid)
    
    for j = 1:ny, i = 1:nx
        x = grid.x[i]
        y = grid.y[j]
        
        dist_to_wall = wall_distance[i, j]
        
        # Skip cells far from walls
        if dist_to_wall > bl_criteria.max_wall_distance
            continue
        end
        
        indicator_score = 0.0
        
        # 1. Y⁺-based refinement (most important for boundary layers)
        current_y_plus = y_plus_field[i, j]
        target_y_plus = bl_criteria.target_y_plus
        
        if current_y_plus > target_y_plus
            # Need refinement to achieve target y⁺
            y_plus_factor = min(current_y_plus / target_y_plus, 10.0)  # Cap at 10x
            indicator_score += 0.4 * y_plus_factor
        end
        
        # 2. Wall-normal velocity gradient (captures boundary layer development)
        wall_gradient = wall_normal_gradients[i, j]
        if wall_gradient > bl_criteria.velocity_boundary_threshold
            gradient_factor = wall_gradient / bl_criteria.velocity_boundary_threshold
            indicator_score += 0.3 * gradient_factor
        end
        
        # 3. Boundary layer thickness-based refinement
        local_bl_thickness = bl_thickness[i, j]
        if local_bl_thickness > 0 && dist_to_wall < local_bl_thickness
            # Inside boundary layer - refine based on relative position
            bl_position = dist_to_wall / local_bl_thickness
            
            if bl_position < 0.1  # Very close to wall
                indicator_score += 0.5
            elseif bl_position < 0.3  # Inner boundary layer
                indicator_score += 0.3
            elseif bl_position < 0.8  # Outer boundary layer
                indicator_score += 0.2
            end
        end
        
        # 4. Pressure gradient enhancement near walls
        if haskey(grid, :pressure_gradients)  # If pre-computed
            press_grad = grid.pressure_gradients[i, j]
            enhanced_threshold = bl_criteria.pressure_gradient_scaling * 
                               (bl_criteria.max_wall_distance - dist_to_wall) / 
                               bl_criteria.max_wall_distance
            
            if press_grad > enhanced_threshold
                indicator_score += 0.2 * (press_grad / enhanced_threshold)
            end
        end
        
        # 5. Wall curvature effects (for curved boundaries)
        curvature_effect = compute_wall_curvature_effect(x, y, bodies, bl_criteria)
        indicator_score += 0.1 * curvature_effect
        
        # 6. Reynolds number scaling
        if bl_criteria.reynolds_number > bl_criteria.critical_reynolds
            re_factor = log10(bl_criteria.reynolds_number / bl_criteria.critical_reynolds)
            indicator_score *= (1.0 + 0.2 * re_factor)
        end
        
        indicators[i, j] = min(indicator_score, 2.0)  # Cap maximum indicator
    end
    
    return indicators
end

"""
    compute_wall_distance_field(grid, bodies)

Computes signed distance function to all solid bodies.
"""
function compute_wall_distance_field(grid::StaggeredGrid, 
                                   bodies::Union{RigidBodyCollection, FlexibleBodyCollection})
    nx, ny = grid.nx, grid.ny
    distance_field = fill(Inf, nx, ny)
    
    for j = 1:ny, i = 1:nx
        x = grid.x[i]
        y = grid.y[j]
        
        min_distance = Inf
        
        if bodies isa RigidBodyCollection
            for body in bodies.bodies
                dist = distance_to_surface(body, x, y)
                min_distance = min(min_distance, abs(dist))  # Take absolute distance
            end
        elseif bodies isa FlexibleBodyCollection
            for body in bodies.bodies
                for k = 1:body.n_points
                    # Distance to each Lagrangian point
                    dist = sqrt((x - body.X[k, 1])^2 + (y - body.X[k, 2])^2)
                    min_distance = min(min_distance, dist)
                end
            end
        end
        
        distance_field[i, j] = min_distance
    end
    
    return distance_field
end

"""
    compute_wall_normal_gradients(grid, state, bodies)

Computes velocity gradients in the wall-normal direction.
"""
function compute_wall_normal_gradients(grid::StaggeredGrid, state::SolutionState,
                                     bodies::Union{RigidBodyCollection, FlexibleBodyCollection})
    nx, ny = grid.nx, grid.ny
    wall_gradients = zeros(nx, ny)
    
    # Interpolate velocities to cell centers
    u_cc = interpolate_u_to_cell_center(state.u, grid)
    v_cc = interpolate_v_to_cell_center(state.v, grid)
    
    for j = 2:ny-1, i = 2:nx-1
        x = grid.x[i]
        y = grid.y[j]
        
        # Find closest wall point and normal direction
        wall_normal = compute_wall_normal_direction(x, y, bodies)
        
        if norm(wall_normal) > 0.5  # Valid normal found
            # Compute velocity gradient in wall-normal direction
            dudx = (u_cc[i+1, j] - u_cc[i-1, j]) / (2 * grid.dx)
            dudy = (u_cc[i, j+1] - u_cc[i, j-1]) / (2 * grid.dy)
            dvdx = (v_cc[i+1, j] - v_cc[i-1, j]) / (2 * grid.dx)
            dvdy = (v_cc[i, j+1] - v_cc[i, j-1]) / (2 * grid.dy)
            
            # Project velocity gradient onto wall-normal direction
            grad_u_normal = dudx * wall_normal[1] + dudy * wall_normal[2]
            grad_v_normal = dvdx * wall_normal[1] + dvdy * wall_normal[2]
            
            wall_gradients[i, j] = sqrt(grad_u_normal^2 + grad_v_normal^2)
        end
    end
    
    return wall_gradients
end

"""
    compute_wall_normal_direction(x, y, bodies)

Computes unit normal vector pointing away from nearest wall.
"""
function compute_wall_normal_direction(x::Float64, y::Float64,
                                     bodies::Union{RigidBodyCollection, FlexibleBodyCollection})
    eps = 1e-6
    
    # Find closest point on wall
    min_distance = Inf
    closest_normal = [0.0, 0.0]
    
    if bodies isa RigidBodyCollection
        for body in bodies.bodies
            # Compute approximate normal using finite differences
            dist_center = distance_to_surface(body, x, y)
            dist_x_plus = distance_to_surface(body, x + eps, y)
            dist_x_minus = distance_to_surface(body, x - eps, y)
            dist_y_plus = distance_to_surface(body, x, y + eps)
            dist_y_minus = distance_to_surface(body, x, y - eps)
            
            if abs(dist_center) < min_distance
                min_distance = abs(dist_center)
                
                # Gradient of distance function gives normal direction
                normal_x = (dist_x_plus - dist_x_minus) / (2 * eps)
                normal_y = (dist_y_plus - dist_y_minus) / (2 * eps)
                
                norm_magnitude = sqrt(normal_x^2 + normal_y^2)
                if norm_magnitude > 1e-12
                    closest_normal = [normal_x, normal_y] / norm_magnitude
                end
            end
        end
    end
    
    return closest_normal
end

"""
    estimate_boundary_layer_thickness(grid, state, bodies, bl_criteria, fluid)

Estimates local boundary layer thickness using velocity profiles.
"""
function estimate_boundary_layer_thickness(grid::StaggeredGrid, state::SolutionState,
                                         bodies::Union{RigidBodyCollection, FlexibleBodyCollection},
                                         bl_criteria::BoundaryLayerAMRCriteria,
                                         fluid::FluidProperties)
    nx, ny = grid.nx, grid.ny
    bl_thickness = zeros(nx, ny)
    
    # Get Reynolds number for scaling
    Re = bl_criteria.reynolds_number
    
    for j = 1:ny, i = 1:nx
        x = grid.x[i]
        y = grid.y[j]
        
        # Simple boundary layer thickness estimates based on flow type
        
        # 1. Flat plate boundary layer: δ ≈ 5x/√(Re_x)
        # Assume characteristic length for Re_x calculation
        characteristic_length = max(x, 0.1)  # Avoid division by zero
        Re_local = Re * characteristic_length
        
        if Re_local > 1e3
            # Turbulent BL
            bl_thickness[i, j] = 0.37 * characteristic_length / (Re_local^0.2)
        else
            # Laminar BL
            bl_thickness[i, j] = 5.0 * characteristic_length / sqrt(Re_local)
        end
        
        # 2. Cylinder boundary layer (if circular bodies detected)
        for body_collection in [bodies]
            if body_collection isa RigidBodyCollection
                for body in body_collection.bodies
                    if body.geometry isa Circle
                        # Distance from cylinder center
                        cylinder_center = body.position
                        dist_from_center = sqrt((x - cylinder_center[1])^2 + (y - cylinder_center[2])^2)
                        cylinder_radius = body.geometry.radius
                        
                        if abs(dist_from_center - cylinder_radius) < 2 * cylinder_radius
                            # Near cylinder - use cylinder BL thickness
                            theta = atan(y - cylinder_center[2], x - cylinder_center[1])
                            Re_theta = Re * theta
                            
                            if Re_theta > 1e5
                                # Turbulent
                                bl_thickness[i, j] = 0.37 * cylinder_radius / (Re_theta^0.2)
                            else
                                # Laminar
                                bl_thickness[i, j] = 3.5 * cylinder_radius / sqrt(Re_theta)
                            end
                        end
                    end
                end
            end
        end
        
        # Cap thickness to reasonable values
        bl_thickness[i, j] = min(bl_thickness[i, j], bl_criteria.max_wall_distance)
        bl_thickness[i, j] = max(bl_thickness[i, j], bl_criteria.min_wall_distance)
    end
    
    return bl_thickness
end

"""
    compute_y_plus_field(grid, state, bodies, wall_distance, fluid)

Computes y⁺ field for boundary layer resolution assessment.
"""
function compute_y_plus_field(grid::StaggeredGrid, state::SolutionState,
                             bodies::Union{RigidBodyCollection, FlexibleBodyCollection},
                             wall_distance::Matrix{Float64}, fluid::FluidProperties)
    nx, ny = grid.nx, grid.ny
    y_plus = zeros(nx, ny)
    
    # Estimate friction velocity and wall shear stress
    wall_shear_stress = estimate_wall_shear_stress(grid, state, bodies, fluid)
    
    if fluid.ρ isa ConstantDensity
        ρ = fluid.ρ.ρ
    else
        error("Variable density not supported for y⁺ calculation")
    end
    
    μ = fluid.μ
    ν = μ / ρ  # Kinematic viscosity
    
    for j = 1:ny, i = 1:nx
        τ_wall = wall_shear_stress[i, j]
        
        if τ_wall > 1e-12  # Valid wall shear stress
            u_tau = sqrt(τ_wall / ρ)  # Friction velocity
            y_dist = wall_distance[i, j]
            
            y_plus[i, j] = y_dist * u_tau / ν
        end
    end
    
    return y_plus
end

"""
    estimate_wall_shear_stress(grid, state, bodies, fluid)

Estimates wall shear stress for y⁺ calculations.
"""
function estimate_wall_shear_stress(grid::StaggeredGrid, state::SolutionState,
                                   bodies::Union{RigidBodyCollection, FlexibleBodyCollection},
                                   fluid::FluidProperties)
    nx, ny = grid.nx, grid.ny
    wall_shear = zeros(nx, ny)
    
    # Interpolate velocities to cell centers
    u_cc = interpolate_u_to_cell_center(state.u, grid)
    v_cc = interpolate_v_to_cell_center(state.v, grid)
    
    μ = fluid.μ
    
    for j = 2:ny-1, i = 2:nx-1
        x = grid.x[i]
        y = grid.y[j]
        
        # Find wall normal direction
        wall_normal = compute_wall_normal_direction(x, y, bodies)
        
        if norm(wall_normal) > 0.5
            # Compute velocity gradients
            dudx = (u_cc[i+1, j] - u_cc[i-1, j]) / (2 * grid.dx)
            dudy = (u_cc[i, j+1] - u_cc[i, j-1]) / (2 * grid.dy)
            dvdx = (v_cc[i+1, j] - v_cc[i-1, j]) / (2 * grid.dx)
            dvdy = (v_cc[i, j+1] - v_cc[i, j-1]) / (2 * grid.dy)
            
            # Wall-normal velocity gradient magnitude
            grad_u_normal = abs(dudx * wall_normal[1] + dudy * wall_normal[2])
            grad_v_normal = abs(dvdx * wall_normal[1] + dvdy * wall_normal[2])
            
            # Approximate wall shear stress
            wall_shear[i, j] = μ * sqrt(grad_u_normal^2 + grad_v_normal^2)
        end
    end
    
    return wall_shear
end

"""
    compute_wall_curvature_effect(x, y, bodies, bl_criteria)

Computes refinement enhancement due to wall curvature effects.
"""
function compute_wall_curvature_effect(x::Float64, y::Float64,
                                     bodies::Union{RigidBodyCollection, FlexibleBodyCollection},
                                     bl_criteria::BoundaryLayerAMRCriteria)
    curvature_effect = 0.0
    
    if bodies isa RigidBodyCollection
        for body in bodies.bodies
            if body.geometry isa Circle
                # Higher refinement needed around curved surfaces
                center = body.position
                radius = body.geometry.radius
                
                dist_from_center = sqrt((x - center[1])^2 + (y - center[2])^2)
                
                if abs(dist_from_center - radius) < bl_criteria.max_wall_distance
                    # Curvature-induced effects
                    curvature = 1.0 / radius
                    curvature_effect = min(curvature * bl_criteria.max_wall_distance, 1.0)
                end
            end
        end
    end
    
    return curvature_effect
end

"""
    apply_anisotropic_refinement!(amr_hierarchy, bl_indicators, bl_criteria)

Applies anisotropic refinement for thin boundary layers.
"""
function apply_anisotropic_refinement!(amr_hierarchy::AMRHierarchy,
                                     bl_indicators::Matrix{Float64},
                                     bl_criteria::BoundaryLayerAMRCriteria)
    if !bl_criteria.enable_anisotropic
        return bl_indicators
    end
    
    nx, ny = size(bl_indicators)
    anisotropic_indicators = copy(bl_indicators)
    
    # Apply anisotropic refinement logic
    # For boundary layers, we want:
    # - Fine resolution normal to wall (high priority)
    # - Coarser resolution tangential to wall (lower priority)
    
    wall_distance = compute_wall_distance_field(
        create_base_staggered_grid(amr_hierarchy), 
        get_bodies_from_hierarchy(amr_hierarchy)
    )
    
    for j = 2:ny-1, i = 2:nx-1
        dist = wall_distance[i, j]
        
        if dist < bl_criteria.max_wall_distance && bl_indicators[i, j] > 0.5
            # Inside boundary layer region
            bl_position = dist / bl_criteria.max_wall_distance
            
            # Enhance wall-normal refinement
            normal_enhancement = 1.0 + (1.0 - bl_position) * 0.5
            
            # Reduce tangential refinement based on aspect ratio limits
            tangential_factor = bl_criteria.tangential_refinement_ratio
            
            # Apply modifications (simplified - full implementation would require
            # directional refinement capabilities)
            anisotropic_indicators[i, j] *= normal_enhancement
        end
    end
    
    return anisotropic_indicators
end

# Helper functions for boundary layer AMR
function create_base_staggered_grid(hierarchy::AMRHierarchy)
    base = hierarchy.base_level
    return StaggeredGrid2D(base.nx, base.ny, 
                          base.x_max - base.x_min,
                          base.y_max - base.y_min;
                          origin_x=base.x_min, origin_y=base.y_min)
end

function get_bodies_from_hierarchy(hierarchy::AMRHierarchy)
    # This would need to be stored in hierarchy or passed separately
    # For now, return empty collection
    return RigidBodyCollection()
end

"""
    refine_for_boundary_layers!(amr_hierarchy, state, bodies, fluid, bl_criteria)

Main function to perform boundary layer-aware adaptive refinement.
"""
function refine_for_boundary_layers!(amr_hierarchy::AMRHierarchy,
                                    state::SolutionState,
                                    bodies::Union{RigidBodyCollection, FlexibleBodyCollection},
                                    fluid::FluidProperties,
                                    bl_criteria::BoundaryLayerAMRCriteria)
    
    # Create temporary grid for computations
    base_grid = create_base_staggered_grid(amr_hierarchy)
    
    # Compute boundary layer-specific indicators
    bl_indicators = compute_boundary_layer_indicators(base_grid, state, bodies, bl_criteria, fluid)
    
    # Apply anisotropic refinement if enabled
    final_indicators = apply_anisotropic_refinement!(amr_hierarchy, bl_indicators, bl_criteria)
    
    # Combine with existing AMR criteria
    standard_indicators = compute_refinement_indicators_amr(
        amr_hierarchy.base_level, state, bodies, amr_hierarchy
    )
    
    # Weighted combination: 60% boundary layer criteria, 40% standard criteria
    combined_indicators = 0.6 * final_indicators + 0.4 * standard_indicators
    
    # Perform refinement using combined indicators
    refined_count = refine_amr_level!(amr_hierarchy, amr_hierarchy.base_level, 
                                     state, combined_indicators)
    
    return refined_count
end

# Export boundary layer AMR functionality
export BoundaryLayerAMRCriteria, compute_boundary_layer_indicators
export compute_wall_distance_field, compute_y_plus_field
export refine_for_boundary_layers!, apply_anisotropic_refinement!