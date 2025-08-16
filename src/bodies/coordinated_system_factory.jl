"""
Coordinated System Factory

This module provides high-level functions for creating and setting up
coordinated flexible body systems with harmonic control.

Functions:
- create_coordinated_flag_system: Main factory function
- setup_simple_two_flag_system: Simplified setup for common use case
- setup_multi_flag_chain: Chain of flags with sequential control
- validate_system_configuration: Configuration validation
"""

"""
    create_coordinated_flag_system(flag_configs::Vector, distance_matrix::Matrix{Float64};
                                  control_options...)

Create a coordinated system of flags with distance-based harmonic control.

# Arguments
- `flag_configs::Vector`: Vector of flag configuration NamedTuples
- `distance_matrix::Matrix{Float64}`: Target distances between flags [i,j]
- Control options: `base_frequency`, `phase_coordination`, PID gains, etc.

# Returns
- `FlexibleBodyCollection`: Collection of flags
- `FlexibleBodyController`: Control system for coordination

# Example
```julia
configs = [
    (start_point=[0.5, 1.0], length=0.8, width=0.05, 
     prescribed_motion=(type=:sinusoidal, amplitude=0.1, frequency=2.0)),
    (start_point=[1.5, 1.0], length=0.6, width=0.03,
     prescribed_motion=(type=:sinusoidal, amplitude=0.08, frequency=2.0))
]

# Maintain 1.0 unit distance between flag trailing edges
target_distances = [0.0 1.0; 1.0 0.0]

flags, controller = create_coordinated_flag_system(configs, target_distances;
                                                  base_frequency=2.0,
                                                  phase_coordination=:synchronized)
```
"""
function create_coordinated_flag_system(flag_configs::Vector, distance_matrix::Matrix{Float64};
                                      base_frequency::Float64 = 1.0,
                                      phase_coordination::Symbol = :synchronized,
                                      kp::Float64 = 0.5,
                                      ki::Float64 = 0.1,
                                      kd::Float64 = 0.05,
                                      control_points::Vector{Symbol} = Symbol[],
                                      amplitude_limits::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[],
                                      control_scale_factor::Float64 = 0.1,
                                      custom_control_indices::Vector{Int} = Int[])
    
    # Validate input
    n_flags = length(flag_configs)
    if size(distance_matrix) != (n_flags, n_flags)
        error("Distance matrix size $(size(distance_matrix)) doesn't match number of flags ($n_flags)")
    end
    
    # Create flag collection
    flag_collection = create_flag_collection(flag_configs)
    
    # Ensure all flags have sinusoidal boundary conditions for control
    _setup_flags_for_control!(flag_collection.bodies, base_frequency)
    
    # Create controller
    controller = FlexibleBodyController(flag_collection.bodies;
                                      base_frequency = base_frequency,
                                      kp = kp, ki = ki, kd = kd,
                                      phase_coordination = phase_coordination,
                                      control_scale_factor = control_scale_factor)
    
    # Set target distances
    set_target_distances!(controller, distance_matrix)
    
    # Set control points if provided
    if !isempty(control_points)
        if !validate_control_points(control_points, n_flags)
            error("Invalid control points specification")
        end
        controller.control_points = copy(control_points)
    end
    
    # Set amplitude limits if provided
    if !isempty(amplitude_limits)
        if length(amplitude_limits) != n_flags
            error("Amplitude limits length ($(length(amplitude_limits))) doesn't match number of flags ($n_flags)")
        end
        controller.amplitude_limits = copy(amplitude_limits)
    end
    
    # Set custom control indices if provided
    if !isempty(custom_control_indices)
        controller.custom_control_indices = copy(custom_control_indices)
    end
    
    return flag_collection, controller
end

"""
    setup_simple_two_flag_system(; leading_flag_config::NamedTuple, 
                                   trailing_flag_config::NamedTuple,
                                   target_separation::Float64,
                                   control_options...)

Simplified setup for a common two-flag system.

# Arguments
- `leading_flag_config::NamedTuple`: Configuration for leading flag
- `trailing_flag_config::NamedTuple`: Configuration for trailing flag
- `target_separation::Float64`: Target distance between trailing edges
- Control options passed to main factory function

# Returns
- `FlexibleBodyCollection`: Two-flag collection
- `FlexibleBodyController`: Control system

# Example
```julia
leading_flag = (start_point=[0.5, 1.0], length=0.8, width=0.04, 
                prescribed_motion=(type=:sinusoidal, amplitude=0.12, frequency=2.5))
trailing_flag = (start_point=[1.8, 1.0], length=0.7, width=0.03,
                 prescribed_motion=(type=:sinusoidal, amplitude=0.10, frequency=2.5))

flags, controller = setup_simple_two_flag_system(
    leading_flag_config=leading_flag,
    trailing_flag_config=trailing_flag,
    target_separation=0.8,
    kp=0.8, ki=0.15, kd=0.08
)
```
"""
function setup_simple_two_flag_system(; leading_flag_config::NamedTuple, 
                                       trailing_flag_config::NamedTuple,
                                       target_separation::Float64,
                                       kwargs...)
    
    # Create flag configuration vector
    flag_configs = [leading_flag_config, trailing_flag_config]
    
    # Create simple distance matrix for two flags
    distance_matrix = [0.0 target_separation; target_separation 0.0]
    
    # Set default control points to trailing edges
    control_points = [:trailing_edge, :trailing_edge]
    
    # Call main factory function
    return create_coordinated_flag_system(flag_configs, distance_matrix;
                                        control_points = control_points,
                                        kwargs...)
end

"""
    setup_multi_flag_chain(flag_configs::Vector{NamedTuple}, 
                          separations::Vector{Float64};
                          control_options...)

Setup a chain of flags with sequential distance control.

# Arguments
- `flag_configs::Vector{NamedTuple}`: Configuration for each flag
- `separations::Vector{Float64}`: Distance between consecutive flags
- Control options passed to main factory function

# Returns
- `FlexibleBodyCollection`: Multi-flag collection
- `FlexibleBodyController`: Control system

# Example
```julia
configs = [flag1_config, flag2_config, flag3_config]
separations = [0.8, 0.6]  # Distance from flag1 to flag2, flag2 to flag3

flags, controller = setup_multi_flag_chain(configs, separations;
                                          phase_coordination=:sequential)
```
"""
function setup_multi_flag_chain(flag_configs::Vector{NamedTuple}, 
                               separations::Vector{Float64};
                               kwargs...)
    
    n_flags = length(flag_configs)
    
    if length(separations) != n_flags - 1
        error("Need $(n_flags-1) separations for $n_flags flags, got $(length(separations))")
    end
    
    # Build distance matrix for chain configuration
    distance_matrix = zeros(n_flags, n_flags)
    
    for i in 1:n_flags
        for j in i+1:n_flags
            # Cumulative distance for chain
            cumulative_distance = sum(separations[i:j-1])
            distance_matrix[i, j] = cumulative_distance
            distance_matrix[j, i] = cumulative_distance
        end
    end
    
    # Set control points to trailing edges by default
    control_points = fill(:trailing_edge, n_flags)
    
    # Call main factory function
    return create_coordinated_flag_system(flag_configs, distance_matrix;
                                        control_points = control_points,
                                        kwargs...)
end

"""
    _setup_flags_for_control!(bodies::Vector{FlexibleBody}, base_frequency::Float64)

Internal function to ensure flags are configured for harmonic control.
"""
function _setup_flags_for_control!(bodies::Vector{FlexibleBody}, base_frequency::Float64)
    
    for flag in bodies
        # Ensure sinusoidal boundary condition
        if flag.bc_type != :sinusoidal_front
            flag.bc_type = :sinusoidal_front
            
            # Set default amplitude if not specified
            if flag.amplitude == 0.0
                flag.amplitude = 0.1  # Default amplitude
                @info "Set default amplitude 0.1 for flag with ID $(flag.id)"
            end
            
            # Set frequency to match base frequency
            flag.frequency = base_frequency
        end
    end
end

"""
    validate_system_configuration(flag_configs::Vector, distance_matrix::Matrix{Float64})

Validate the configuration for a coordinated system.

# Arguments
- `flag_configs::Vector`: Flag configurations
- `distance_matrix::Matrix{Float64}`: Target distance matrix

# Returns
- `Bool`: True if configuration is valid
- `Vector{String}`: List of validation messages/warnings
"""
function validate_system_configuration(flag_configs::Vector, distance_matrix::Matrix{Float64})
    
    messages = String[]
    is_valid = true
    
    n_flags = length(flag_configs)
    
    # Check distance matrix dimensions
    if size(distance_matrix) != (n_flags, n_flags)
        push!(messages, "ERROR: Distance matrix size $(size(distance_matrix)) doesn't match number of flags ($n_flags)")
        is_valid = false
    else
        # Check diagonal is zero
        for i in 1:n_flags
            if distance_matrix[i, i] != 0.0
                push!(messages, "WARNING: Distance matrix diagonal should be zero, found $(distance_matrix[i, i]) at position [$i, $i]")
            end
        end
        
        # Check for symmetric matrix
        for i in 1:n_flags
            for j in i+1:n_flags
                if abs(distance_matrix[i, j] - distance_matrix[j, i]) > 1e-12
                    push!(messages, "WARNING: Distance matrix not symmetric at positions [$i, $j] and [$j, $i]")
                end
            end
        end
        
        # Check for reasonable distances
        for i in 1:n_flags
            for j in i+1:n_flags
                if distance_matrix[i, j] < 0.0
                    push!(messages, "ERROR: Negative distance $(distance_matrix[i, j]) at position [$i, $j]")
                    is_valid = false
                end
            end
        end
    end
    
    # Check flag configurations
    for (i, config) in enumerate(flag_configs)
        # Check required fields
        required_fields = [:start_point, :length, :width]
        for field in required_fields
            if !haskey(config, field)
                push!(messages, "ERROR: Flag $i missing required field '$field'")
                is_valid = false
            end
        end
        
        # Check start point format
        if haskey(config, :start_point) && length(config.start_point) != 2
            push!(messages, "ERROR: Flag $i start_point must be 2-element vector [x, z]")
            is_valid = false
        end
        
        # Check positive dimensions
        if haskey(config, :length) && config.length <= 0.0
            push!(messages, "ERROR: Flag $i length must be positive, got $(config.length)")
            is_valid = false
        end
        
        if haskey(config, :width) && config.width <= 0.0
            push!(messages, "ERROR: Flag $i width must be positive, got $(config.width)")
            is_valid = false
        end
        
        # Validate prescribed motion if present
        if haskey(config, :prescribed_motion)
            motion = config.prescribed_motion
            if haskey(motion, :type) && motion.type == :sinusoidal
                if haskey(motion, :amplitude) && motion.amplitude < 0.0
                    push!(messages, "WARNING: Flag $i has negative amplitude $(motion.amplitude)")
                end
                if haskey(motion, :frequency) && motion.frequency <= 0.0
                    push!(messages, "ERROR: Flag $i frequency must be positive, got $(motion.frequency)")
                    is_valid = false
                end
            end
        end
    end
    
    return is_valid, messages
end

"""
    print_system_summary(flags::FlexibleBodyCollection, controller::FlexibleBodyController)

Print a summary of the coordinated system configuration.
"""
function print_system_summary(flags::FlexibleBodyCollection, controller::FlexibleBodyController)
    
    println("\nCoordinated Flag System Summary:")
    println("   Number of flags: $(flags.n_bodies)")
    println("   Base frequency: $(controller.base_frequency) Hz")
    println("   Phase coordination: $(length(unique(controller.phase_offsets)) == 1 ? "Synchronized" : "Custom")")
    
    # PID parameters
    println("\n   PID Control Parameters:")
    println("     Proportional gain (Kp): $(controller.kp)")
    println("     Integral gain (Ki): $(controller.ki)")
    println("     Derivative gain (Kd): $(controller.kd)")
    println("     Control scale factor: $(controller.control_scale_factor)")
    
    # Target distances
    println("\n   Target Distances:")
    n_bodies = length(controller.bodies)
    active_controls = 0
    for i in 1:n_bodies
        for j in i+1:n_bodies
            if controller.target_distances[i, j] > 0.0
                println("     Flag $i ←→ Flag $j: $(controller.target_distances[i, j])")
                active_controls += 1
            end
        end
    end
    println("     Total active controls: $active_controls")
    
    # Flag properties
    println("\n   Flag Properties:")
    for (i, flag) in enumerate(flags.bodies)
        start_x, start_z = flag.X[1, 1], flag.X[1, 2]
        @printf "     Flag %d: L=%.3f, W=%.3f, Start=[%.2f, %.2f], Amp=%.3f, Control=:%s\n" i flag.length flag.thickness start_x start_z flag.amplitude controller.control_points[i]
    end
    
    # Amplitude limits
    println("\n   Amplitude Limits:")
    for (i, (min_amp, max_amp)) in enumerate(controller.amplitude_limits)
        println("     Flag $i: $(min_amp*100)% - $(max_amp*100)% of original")
    end
end