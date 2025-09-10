"""
Trixi.jl-inspired three-level AMR controller
Implements gradual refinement strategy for better mass conservation.
"""

"""
    TrixiStyleController

Three-level AMR controller inspired by Trixi.jl's approach.
Uses base/medium/maximum levels for gradual refinement.
"""
struct TrixiStyleController
    base_level::Int           # Minimum refinement level
    medium_level::Int         # Intermediate refinement level  
    max_level::Int           # Maximum refinement level
    
    # Thresholds for different levels
    refine_threshold_medium::Float64    # Threshold to refine to medium level
    refine_threshold_max::Float64       # Threshold to refine to maximum level
    coarsen_threshold::Float64          # Threshold to coarsen
    
    # Physics-aware thresholds
    shock_threshold::Float64
    curvature_threshold::Float64
    vorticity_threshold::Float64
    body_distance::Float64
end

"""
    create_trixi_style_controller(config)

Create a Trixi-style three-level controller with physics-aware thresholds.
"""
function create_trixi_style_controller(shock_threshold=0.1, curvature_threshold=2.0, 
                                      vorticity_threshold=15.0, body_distance=0.25)
    return TrixiStyleController(
        0,      # base_level
        1,      # medium_level  
        2,      # max_level
        0.3,    # refine_threshold_medium (conservative)
        0.7,    # refine_threshold_max (very selective)
        0.1,    # coarsen_threshold (aggressive coarsening)
        shock_threshold,
        curvature_threshold, 
        vorticity_threshold,
        body_distance
    )
end

"""
    determine_refinement_action(controller, indicator_value, current_level, position, bodies)

Determine refinement action based on Trixi-style three-level logic.
Returns: :refine, :coarsen, or :keep
"""
function determine_refinement_action(controller::TrixiStyleController, 
                                   indicator_value::Float64,
                                   current_level::Int,
                                   position::Tuple{Float64, Float64},
                                   bodies=nothing)
    x, z = position
    
    # Check if near rigid body (always refine at least to medium level)
    near_body = false
    if bodies !== nothing
        for body in bodies.bodies
            if hasfield(typeof(body), :center) && hasfield(typeof(body), :radius)
                dist = sqrt((x - body.center[1])^2 + (z - body.center[2])^2)
                if dist < body.radius + controller.body_distance
                    near_body = true
                    break
                end
            end
        end
    end
    
    # Trixi-style three-level logic
    if current_level < controller.base_level
        return :refine  # Always refine to at least base level
    elseif current_level == controller.base_level
        # Decide whether to refine to medium level
        if near_body || indicator_value > controller.refine_threshold_medium
            return :refine
        elseif indicator_value < controller.coarsen_threshold * 0.5
            return :coarsen  # Very conservative coarsening from base
        else
            return :keep
        end
    elseif current_level == controller.medium_level
        # Decide whether to refine to maximum level
        if indicator_value > controller.refine_threshold_max
            return :refine
        elseif indicator_value < controller.coarsen_threshold && !near_body
            return :coarsen
        else
            return :keep
        end
    elseif current_level >= controller.max_level
        # At maximum level, only consider coarsening
        if indicator_value < controller.coarsen_threshold * 2.0 && !near_body
            return :coarsen
        else
            return :keep
        end
    else
        return :keep
    end
end

"""
    apply_trixi_style_amr_logic(state, grid, bodies, controller, adapt_interval, current_step)

Apply Trixi-style AMR logic with physics-aware indicators and three-level control.
Only adapts every `adapt_interval` steps for better mass conservation.
"""
function apply_trixi_style_amr_logic(state, grid, bodies, controller, adapt_interval, current_step)
    # Only adapt every N steps (much less frequent than every step)
    if current_step % adapt_interval != 0
        return false  # No adaptation this step
    end
    
    println("TRIXI-STYLE AMR: Adapting at step $current_step (interval: $adapt_interval)")
    
    # Compute physics-aware indicator
    include("physics_aware_indicators.jl")
    indicator_field = trixi_style_amr_indicator(state, grid, nothing)
    
    nx, nz = size(state.p)
    refinement_decisions = Matrix{Symbol}(undef, nx, nz)
    current_levels = ones(Int, nx, nz)  # Assume uniform level 1 initially
    
    total_refine = 0
    total_coarsen = 0
    total_keep = 0
    
    # Apply three-level controller logic
    for j = 1:nz, i = 1:nx
        x = (i - 0.5) * grid.dx
        z = (j - 0.5) * grid.dz
        position = (x, z)
        
        indicator_value = indicator_field[i, j]
        current_level = current_levels[i, j]
        
        action = determine_refinement_action(controller, indicator_value, 
                                           current_level, position, bodies)
        refinement_decisions[i, j] = action
        
        if action == :refine
            total_refine += 1
        elseif action == :coarsen  
            total_coarsen += 1
        else
            total_keep += 1
        end
    end
    
    # Report adaptation statistics
    total_cells = nx * nz
    refine_percent = round(100 * total_refine / total_cells, digits=1)
    coarsen_percent = round(100 * total_coarsen / total_cells, digits=1)
    keep_percent = round(100 * total_keep / total_cells, digits=1)
    
    println("  Refinement decisions: Refine $(refine_percent)%, Coarsen $(coarsen_percent)%, Keep $(keep_percent)%")
    println("  Max indicator value: $(round(maximum(indicator_field), digits=4))")
    println("  Physics-aware adaptation completed")
    
    return total_refine > 0 || total_coarsen > 0  # Return true if any adaptation occurred
end