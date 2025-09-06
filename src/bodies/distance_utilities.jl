"""
Distance Measurement Utilities

This module provides functions for measuring distances between flexible bodies
and other geometric utilities needed for the control system.
"""

using Printf

"""
Functions:
- compute_body_distance: Main distance calculation function
- get_body_point: Extract specific points from flexible bodies
- validate_control_points: Validation utilities
- distance_statistics: Analysis functions
"""

"""
    compute_body_distance(body1::FlexibleBody, body2::FlexibleBody, 
                         point1::Symbol, point2::Symbol, 
                         custom_idx1::Int=0, custom_idx2::Int=0)

Compute distance between specified points on two flexible bodies.

# Arguments
- `body1, body2::FlexibleBody`: Bodies to measure distance between
- `point1, point2::Symbol`: Control points (:leading_edge, :trailing_edge, :center, :custom)
- `custom_idx1, custom_idx2::Int`: Point indices if using :custom option

# Returns
- `Float64`: Euclidean distance between the specified points

# Control Point Options
- `:leading_edge`: First point (s=0) of the body
- `:trailing_edge`: Last point (s=L) of the body  
- `:center`: Middle point of the body
- `:custom`: Use custom point index (requires custom_idx parameter)
"""
function compute_body_distance(body1::FlexibleBody, body2::FlexibleBody, 
                              point1::Symbol, point2::Symbol, 
                              custom_idx1::Int=0, custom_idx2::Int=0)
    
    # Get coordinates for first body
    x1, z1 = get_body_point(body1, point1, custom_idx1)
    
    # Get coordinates for second body  
    x2, z2 = get_body_point(body2, point2, custom_idx2)
    
    return sqrt((x2 - x1)^2 + (z2 - z1)^2)
end

"""
    get_body_point(body::FlexibleBody, point_type::Symbol, custom_idx::Int=0)

Extract coordinates of a specific point on a flexible body.

# Arguments
- `body::FlexibleBody`: Body to extract point from
- `point_type::Symbol`: Type of point (:leading_edge, :trailing_edge, :center, :custom)
- `custom_idx::Int`: Point index for :custom option

# Returns
- `Tuple{Float64, Float64}`: (x, z) coordinates of the point
"""
function get_body_point(body::FlexibleBody, point_type::Symbol, custom_idx::Int=0)
    
    if point_type == :leading_edge
        return body.X[1, 1], body.X[1, 2]
        
    elseif point_type == :trailing_edge
        return body.X[end, 1], body.X[end, 2]
        
    elseif point_type == :center
        mid_idx = max(1, div(body.n_points, 2))
        return body.X[mid_idx, 1], body.X[mid_idx, 2]
        
    elseif point_type == :custom
        idx = max(1, min(body.n_points, custom_idx))
        if custom_idx <= 0 || custom_idx > body.n_points
            @warn "Custom index $custom_idx out of range [1, $(body.n_points)], using clamped value $idx"
        end
        return body.X[idx, 1], body.X[idx, 2]
        
    else
        error("Unknown control point type: $point_type. Valid options: :leading_edge, :trailing_edge, :center, :custom")
    end
end

"""
    validate_control_points(control_points::Vector{Symbol}, n_bodies::Int)

Validate that control point specifications are valid.

# Arguments
- `control_points::Vector{Symbol}`: Control points for each body
- `n_bodies::Int`: Number of bodies

# Returns
- `Bool`: True if valid, false otherwise
"""
function validate_control_points(control_points::Vector{Symbol}, n_bodies::Int)
    
    if length(control_points) != n_bodies
        @error "Control points vector length ($(length(control_points))) doesn't match number of bodies ($n_bodies)"
        return false
    end
    
    valid_points = [:leading_edge, :trailing_edge, :center, :custom]
    
    for (i, point) in enumerate(control_points)
        if point ∉ valid_points
            @error "Invalid control point '$point' for body $i. Valid options: $valid_points"
            return false
        end
    end
    
    return true
end

"""
    compute_multi_body_distances(bodies::Vector{FlexibleBody}, 
                                control_points::Vector{Symbol},
                                custom_indices::Vector{Int} = Int[])

Compute all pairwise distances between a collection of bodies.

# Arguments
- `bodies::Vector{FlexibleBody}`: Collection of bodies
- `control_points::Vector{Symbol}`: Control point type for each body
- `custom_indices::Vector{Int}`: Custom indices for bodies using :custom control points

# Returns
- `Matrix{Float64}`: Distance matrix [i,j] where entry is distance from body i to body j
"""
function compute_multi_body_distances(bodies::Vector{FlexibleBody}, 
                                    control_points::Vector{Symbol},
                                    custom_indices::Vector{Int} = Int[])
    
    n_bodies = length(bodies)
    
    if !validate_control_points(control_points, n_bodies)
        error("Invalid control points specification")
    end
    
    distance_matrix = zeros(n_bodies, n_bodies)
    
    for i in 1:n_bodies
        for j in i+1:n_bodies
            custom_idx_i = i <= length(custom_indices) ? custom_indices[i] : 0
            custom_idx_j = j <= length(custom_indices) ? custom_indices[j] : 0
            
            dist = compute_body_distance(bodies[i], bodies[j], 
                                       control_points[i], control_points[j],
                                       custom_idx_i, custom_idx_j)
            
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric matrix
        end
    end
    
    return distance_matrix
end

"""
    compute_body_center_of_mass(body::FlexibleBody)

Compute the center of mass of a flexible body.

# Arguments
- `body::FlexibleBody`: Body to analyze

# Returns
- `Tuple{Float64, Float64}`: (x, z) coordinates of center of mass
"""
function compute_body_center_of_mass(body::FlexibleBody)
    
    # Simple center of mass calculation (uniform mass distribution assumed)
    x_cm = mean(body.X[:, 1])
    z_cm = mean(body.X[:, 2])
    
    return x_cm, z_cm
end

"""
    compute_body_bounding_box(body::FlexibleBody)

Compute the bounding box of a flexible body.

# Arguments
- `body::FlexibleBody`: Body to analyze

# Returns
- `NamedTuple`: Bounding box with fields (x_min, x_max, z_min, z_max, width, height)
"""
function compute_body_bounding_box(body::FlexibleBody)
    
    x_min = minimum(body.X[:, 1])
    x_max = maximum(body.X[:, 1])
    z_min = minimum(body.X[:, 2])
    z_max = maximum(body.X[:, 2])
    
    return (
        x_min = x_min,
        x_max = x_max,
        z_min = z_min,
        z_max = z_max,
        width = x_max - x_min,
        height = z_max - z_min
    )
end

"""
    find_closest_points(body1::FlexibleBody, body2::FlexibleBody)

Find the closest points between two flexible bodies.

# Arguments
- `body1, body2::FlexibleBody`: Bodies to analyze

# Returns
- `NamedTuple`: Closest point information (indices, coordinates, distance)
"""
function find_closest_points(body1::FlexibleBody, body2::FlexibleBody)
    
    min_distance = Inf
    best_i, best_j = 1, 1
    
    for i in 1:body1.n_points
        for j in 1:body2.n_points
            x1, z1 = body1.X[i, 1], body1.X[i, 2]
            x2, z2 = body2.X[j, 1], body2.X[j, 2]
            
            dist = sqrt((x2 - x1)^2 + (z2 - z1)^2)
            
            if dist < min_distance
                min_distance = dist
                best_i, best_j = i, j
            end
        end
    end
    
    return (
        body1_index = best_i,
        body2_index = best_j,
        body1_point = (body1.X[best_i, 1], body1.X[best_i, 2]),
        body2_point = (body2.X[best_j, 1], body2.X[best_j, 2]),
        distance = min_distance
    )
end

"""
    distance_statistics(bodies::Vector{FlexibleBody}, 
                       control_points::Vector{Symbol},
                       reference_distances::Matrix{Float64})

Compute statistical measures of distance control performance.

# Arguments
- `bodies::Vector{FlexibleBody}`: Bodies to analyze
- `control_points::Vector{Symbol}`: Control points for each body
- `reference_distances::Matrix{Float64}`: Target/reference distances

# Returns
- `NamedTuple`: Statistics including mean error, RMS error, max error, etc.
"""
function distance_statistics(bodies::Vector{FlexibleBody}, 
                           control_points::Vector{Symbol},
                           reference_distances::Matrix{Float64})
    
    current_distances = compute_multi_body_distances(bodies, control_points)
    
    n_bodies = length(bodies)
    errors = Float64[]
    relative_errors = Float64[]
    
    for i in 1:n_bodies
        for j in i+1:n_bodies
            if reference_distances[i, j] > 0.0
                error = current_distances[i, j] - reference_distances[i, j]
                rel_error = abs(error) / reference_distances[i, j] * 100  # Percentage
                
                push!(errors, error)
                push!(relative_errors, rel_error)
            end
        end
    end
    
    if isempty(errors)
        return (
            n_comparisons = 0,
            mean_error = 0.0,
            rms_error = 0.0,
            max_error = 0.0,
            mean_relative_error = 0.0,
            max_relative_error = 0.0
        )
    end
    
    return (
        n_comparisons = length(errors),
        mean_error = mean(errors),
        rms_error = sqrt(mean(errors.^2)),
        max_error = maximum(abs.(errors)),
        mean_relative_error = mean(relative_errors),
        max_relative_error = maximum(relative_errors)
    )
end

"""
    print_distance_analysis(bodies::Vector{FlexibleBody}, 
                           control_points::Vector{Symbol},
                           target_distances::Matrix{Float64})

Print detailed distance analysis for debugging.
"""
function print_distance_analysis(bodies::Vector{FlexibleBody}, 
                                control_points::Vector{Symbol},
                                target_distances::Matrix{Float64})
    
    println("\nDistance Analysis:")
    println("   Bodies: $(length(bodies))")
    
    current_distances = compute_multi_body_distances(bodies, control_points)
    stats = distance_statistics(bodies, control_points, target_distances)
    
    # Print pairwise distances
    n_bodies = length(bodies)
    for i in 1:n_bodies
        for j in i+1:n_bodies
            if target_distances[i, j] > 0.0
                current = current_distances[i, j]
                target = target_distances[i, j]
                error = current - target
                rel_error = abs(error) / target * 100
                
                @printf "   Body %d--%d: current=%.4f, target=%.4f, error=%.4f (%.1f%%)\n" i j current target error rel_error
            end
        end
    end
    
    # Print summary statistics
    @printf "\n   Summary Statistics:\n"
    @printf "     Comparisons: %d\n" stats.n_comparisons
    @printf "     Mean error: %.4f\n" stats.mean_error
    @printf "     RMS error: %.4f\n" stats.rms_error
    @printf "     Max error: %.4f\n" stats.max_error
    @printf "     Mean relative error: %.2f%%\n" stats.mean_relative_error
    @printf "     Max relative error: %.2f%%\n" stats.max_relative_error
end