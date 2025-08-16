"""
Horizontal Plane Distance Control Utilities

This module provides specialized functions for coordinating N-number of flags
that are initially positioned on the same horizontal plane (same z-coordinate).
The system maintains horizontal distances between flags while allowing vertical motion.

Key Features:
- Automatic detection of flags on same horizontal plane
- Distance matrix generation for horizontal alignment constraints
- Specialized control strategies for horizontal formations
- Validation utilities for horizontal plane configurations
"""

"""
    detect_horizontal_groups(flag_configs::Vector; tolerance::Float64 = 1e-6)

Detect groups of flags that are initially on the same horizontal plane.

# Arguments
- `flag_configs::Vector`: Vector of flag configuration NamedTuples
- `tolerance::Float64`: Tolerance for considering flags on same plane

# Returns
- `Vector{Vector{Int}}`: Groups of flag indices that are on same horizontal plane
- `Vector{Float64}`: Z-coordinates of each horizontal plane

# Example
```julia
configs = [
    (start_point=[0.5, 1.0], length=0.4, width=0.02),  # z=1.0
    (start_point=[1.0, 1.0], length=0.4, width=0.02),  # z=1.0  
    (start_point=[1.5, 1.0], length=0.4, width=0.02),  # z=1.0
    (start_point=[0.7, 0.5], length=0.3, width=0.02),  # z=0.5
    (start_point=[1.2, 0.5], length=0.3, width=0.02)   # z=0.5
]

groups, z_coords = detect_horizontal_groups(configs)
# groups = [[1,2,3], [4,5]]  # flags 1,2,3 at z=1.0, flags 4,5 at z=0.5
# z_coords = [1.0, 0.5]
```
"""
function detect_horizontal_groups(flag_configs::Vector; tolerance::Float64 = 1e-6)
    
    n_flags = length(flag_configs)
    
    # Extract z-coordinates
    z_coords = Float64[]
    for config in flag_configs
        if haskey(config, :start_point) && length(config.start_point) >= 2
            push!(z_coords, config.start_point[2])
        else
            error("Flag configuration must have start_point with [x, z] coordinates")
        end
    end
    
    # Group flags by similar z-coordinates
    groups = Vector{Int}[]
    plane_z_coords = Float64[]
    
    for i in 1:n_flags
        z_i = z_coords[i]
        
        # Find existing group with similar z-coordinate
        group_found = false
        for (group_idx, group) in enumerate(groups)
            if abs(z_i - plane_z_coords[group_idx]) < tolerance
                push!(group, i)
                group_found = true
                break
            end
        end
        
        # Create new group if no matching z-coordinate found
        if !group_found
            push!(groups, [i])
            push!(plane_z_coords, z_i)
        end
    end
    
    return groups, plane_z_coords
end

"""
    create_horizontal_distance_matrix(flag_configs::Vector, 
                                     target_separations::Dict{Int, Float64} = Dict{Int, Float64}();
                                     tolerance::Float64 = 1e-6)

Create distance matrix for maintaining horizontal distances between flags on same plane.

# Arguments  
- `flag_configs::Vector`: Vector of flag configurations
- `target_separations::Dict{Int, Float64}`: Custom separations for specific horizontal planes
- `tolerance::Float64`: Tolerance for plane detection

# Returns
- `Matrix{Float64}`: Distance matrix with constraints only between flags on same horizontal plane
- `Vector{Vector{Int}}`: Groups of flags on same horizontal planes

# Example
```julia
configs = [
    (start_point=[0.5, 1.0], length=0.4, width=0.02),
    (start_point=[1.2, 1.0], length=0.4, width=0.02), 
    (start_point=[1.9, 1.0], length=0.4, width=0.02)
]

# Maintain 0.6 unit horizontal separation for plane at z=1.0
target_seps = Dict(1 => 0.6)  # Plane 1 (z=1.0) uses 0.6 separation

distance_matrix, groups = create_horizontal_distance_matrix(configs, target_seps)
```
"""
function create_horizontal_distance_matrix(flag_configs::Vector, 
                                         target_separations::Dict{Int, Float64} = Dict{Int, Float64}();
                                         tolerance::Float64 = 1e-6)
    
    n_flags = length(flag_configs)
    distance_matrix = zeros(n_flags, n_flags)
    
    # Detect horizontal groups
    groups, plane_z_coords = detect_horizontal_groups(flag_configs; tolerance=tolerance)
    
    println("📐 Detected horizontal planes:")
    for (i, (group, z_coord)) in enumerate(zip(groups, plane_z_coords))
        println("   Plane $i (z=$(z_coord)): flags $(group)")
    end
    
    # Set distances for each horizontal plane
    for (plane_idx, group) in enumerate(groups)
        if length(group) < 2
            continue  # Need at least 2 flags to set distances
        end
        
        # Get target separation for this plane
        target_sep = get(target_separations, plane_idx, _compute_default_separation(flag_configs, group))
        
        println("   Using separation $(target_sep) for plane $plane_idx")
        
        # Sort flags by x-coordinate for consistent ordering
        x_coords = [flag_configs[idx].start_point[1] for idx in group]
        sorted_indices = sortperm(x_coords)
        sorted_group = group[sorted_indices]
        
        # Set distances between consecutive flags in this plane
        for i in 1:(length(sorted_group)-1)
            flag_i = sorted_group[i]
            flag_j = sorted_group[i+1]
            
            distance_matrix[flag_i, flag_j] = target_sep
            distance_matrix[flag_j, flag_i] = target_sep  # Symmetric
        end
        
        # Optionally set distances between non-consecutive flags
        # (cumulative distances for chain formation)
        for i in 1:length(sorted_group)
            for j in (i+2):length(sorted_group)
                flag_i = sorted_group[i]
                flag_j = sorted_group[j]
                
                cumulative_distance = target_sep * (j - i)
                distance_matrix[flag_i, flag_j] = cumulative_distance
                distance_matrix[flag_j, flag_i] = cumulative_distance
            end
        end
    end
    
    return distance_matrix, groups
end

"""
    _compute_default_separation(flag_configs::Vector, group::Vector{Int})

Compute default separation based on flag lengths and initial positions.
"""
function _compute_default_separation(flag_configs::Vector, group::Vector{Int})
    if length(group) < 2
        return 0.5  # Default fallback
    end
    
    # Get x-coordinates and flag lengths
    positions = [(flag_configs[idx].start_point[1], get(flag_configs[idx], :length, 0.4)) for idx in group]
    sort!(positions)
    
    # Compute average spacing from initial configuration
    total_span = positions[end][1] - positions[1][1]
    n_gaps = length(positions) - 1
    
    if n_gaps > 0
        avg_separation = total_span / n_gaps
        # Add some buffer based on flag length
        max_flag_length = maximum([pos[2] for pos in positions])
        return max(avg_separation, max_flag_length * 1.2)
    else
        return 0.5
    end
end

"""
    setup_horizontal_plane_system(flag_configs::Vector, 
                                 target_separations::Dict{Int, Float64} = Dict{Int, Float64}();
                                 control_options...)

Setup coordinated flag system for maintaining horizontal plane distances.

# Arguments
- `flag_configs::Vector`: Vector of flag configurations
- `target_separations::Dict{Int, Float64}`: Target separations for each horizontal plane
- Control options: passed to main factory function

# Returns
- `FlexibleBodyCollection`: Collection of flags
- `FlexibleBodyController`: Control system
- `Vector{Vector{Int}}`: Groups of flags on same horizontal planes

# Example
```julia
configs = [
    (start_point=[0.3, 1.0], length=0.5, width=0.03, prescribed_motion=(type=:sinusoidal, amplitude=0.08, frequency=2.0)),
    (start_point=[1.0, 1.0], length=0.4, width=0.03, prescribed_motion=(type=:sinusoidal, amplitude=0.06, frequency=2.0)),
    (start_point=[1.6, 1.0], length=0.4, width=0.03, prescribed_motion=(type=:sinusoidal, amplitude=0.05, frequency=2.0)),
    (start_point=[2.2, 1.0], length=0.35, width=0.03, prescribed_motion=(type=:sinusoidal, amplitude=0.04, frequency=2.0))
]

# All flags at z=1.0 with 0.5 unit separation
target_seps = Dict(1 => 0.5)

flags, controller, groups = setup_horizontal_plane_system(configs, target_seps;
                                                         base_frequency=2.0,
                                                         kp=0.7, ki=0.15, kd=0.08)
```
"""
function setup_horizontal_plane_system(flag_configs::Vector, 
                                      target_separations::Dict{Int, Float64} = Dict{Int, Float64}();
                                      kwargs...)
    
    # Generate distance matrix for horizontal plane constraints
    distance_matrix, groups = create_horizontal_distance_matrix(flag_configs, target_separations)
    
    # Set control points to trailing edges by default for horizontal distance control
    control_points = fill(:trailing_edge, length(flag_configs))
    
    # Create coordinated system
    flags, controller = create_coordinated_flag_system(flag_configs, distance_matrix;
                                                      control_points = control_points,
                                                      kwargs...)
    
    return flags, controller, groups
end

"""
    validate_horizontal_plane_configuration(flag_configs::Vector, 
                                           target_separations::Dict{Int, Float64} = Dict{Int, Float64}())

Validate configuration for horizontal plane distance control.

# Arguments
- `flag_configs::Vector`: Flag configurations to validate
- `target_separations::Dict{Int, Float64}`: Target separations

# Returns  
- `Bool`: True if configuration is valid
- `Vector{String}`: Validation messages
"""
function validate_horizontal_plane_configuration(flag_configs::Vector, 
                                                target_separations::Dict{Int, Float64} = Dict{Int, Float64}())
    
    messages = String[]
    is_valid = true
    
    # Basic validation
    if isempty(flag_configs)
        push!(messages, "ERROR: No flag configurations provided")
        return false, messages
    end
    
    # Check for required fields
    for (i, config) in enumerate(flag_configs)
        if !haskey(config, :start_point)
            push!(messages, "ERROR: Flag $i missing start_point")
            is_valid = false
        elseif length(config.start_point) < 2
            push!(messages, "ERROR: Flag $i start_point must have [x, z] coordinates")
            is_valid = false
        end
    end
    
    if !is_valid
        return false, messages
    end
    
    # Detect horizontal groups
    groups, plane_z_coords = detect_horizontal_groups(flag_configs)
    
    # Validate groups
    total_flags_in_groups = sum(length(group) for group in groups)
    if total_flags_in_groups != length(flag_configs)
        push!(messages, "ERROR: Grouping algorithm failed - flag count mismatch")
        is_valid = false
    end
    
    # Check each horizontal plane
    for (plane_idx, (group, z_coord)) in enumerate(zip(groups, plane_z_coords))
        if length(group) == 1
            push!(messages, "WARNING: Plane $plane_idx (z=$z_coord) has only 1 flag - no distance control possible")
        end
        
        # Check for reasonable x-spacing
        x_coords = [flag_configs[idx].start_point[1] for idx in group]
        sort!(x_coords)
        
        min_spacing = minimum(diff(x_coords)) if length(x_coords) > 1 else Inf
        if min_spacing < 0.05  # Very close flags
            push!(messages, "WARNING: Flags on plane $plane_idx are very close (min spacing: $(min_spacing))")
        end
        
        # Check target separation if provided
        if haskey(target_separations, plane_idx)
            target_sep = target_separations[plane_idx]
            if target_sep <= 0.0
                push!(messages, "ERROR: Plane $plane_idx target separation must be positive, got $target_sep")
                is_valid = false
            elseif target_sep < min_spacing * 0.5
                push!(messages, "WARNING: Plane $plane_idx target separation ($target_sep) much smaller than current spacing ($min_spacing)")
            end
        end
    end
    
    # Summary information
    push!(messages, "INFO: Found $(length(groups)) horizontal planes with $(length(flag_configs)) total flags")
    for (i, (group, z_coord)) in enumerate(zip(groups, plane_z_coords))
        push!(messages, "INFO: Plane $i at z=$z_coord contains $(length(group)) flags")
    end
    
    return is_valid, messages
end

"""
    print_horizontal_plane_analysis(flag_configs::Vector, 
                                   target_separations::Dict{Int, Float64} = Dict{Int, Float64}())

Print detailed analysis of horizontal plane configuration.
"""
function print_horizontal_plane_analysis(flag_configs::Vector, 
                                        target_separations::Dict{Int, Float64} = Dict{Int, Float64}())
    
    println("\nHorizontal Plane Configuration Analysis:")
    println("   Total flags: $(length(flag_configs))")
    
    # Detect and analyze groups
    groups, plane_z_coords = detect_horizontal_groups(flag_configs)
    
    println("   Horizontal planes detected: $(length(groups))")
    
    for (plane_idx, (group, z_coord)) in enumerate(zip(groups, plane_z_coords))
        println("\n   Plane $plane_idx (z = $z_coord):")
        println("     Flags: $(group) ($(length(group)) total)")
        
        if length(group) > 1
            # Analyze x-positions
            x_positions = [(idx, flag_configs[idx].start_point[1]) for idx in group]
            sort!(x_positions, by=x->x[2])
            
            println("     X-positions (sorted):")
            for (flag_idx, x_pos) in x_positions
                flag_length = get(flag_configs[flag_idx], :length, 0.0)
                @printf "       Flag %d: x=%.3f, length=%.3f\n" flag_idx x_pos flag_length
            end
            
            # Current spacings
            x_coords = [pos[2] for pos in x_positions]
            spacings = diff(x_coords)
            println("     Current spacings: $(round.(spacings, digits=3))")
            @printf "     Average spacing: %.3f\n" mean(spacings)
            
            # Target separation
            target_sep = get(target_separations, plane_idx, _compute_default_separation(flag_configs, group))
            @printf "     Target separation: %.3f\n" target_sep
            
            # Control strategy
            n_controls = length(group) - 1  # consecutive pairs
            n_total_controls = length(group) * (length(group) - 1) ÷ 2  # all pairs
            println("     Control pairs: $n_controls consecutive, $n_total_controls total possible")
        else
            println("     Single flag - no distance control needed")
        end
    end
    
    # Validation summary
    is_valid, messages = validate_horizontal_plane_configuration(flag_configs, target_separations)
    
    println("\n   Validation: $(is_valid ? "VALID" : "INVALID")")
    for msg in messages
        if startswith(msg, "ERROR")
            println("     ERROR: $msg")
        elseif startswith(msg, "WARNING")
            println("     WARNING: $msg")
        elseif startswith(msg, "INFO")
            println("     INFO: $msg")
        end
    end
end