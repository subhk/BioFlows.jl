"""
Flexible Body Controller Module

This module provides PID-based control systems for coordinating harmonic oscillations
between multiple flexible bodies while maintaining specified distances.

Main components:
- FlexibleBodyController: Main control system struct
- PID control algorithms
- Harmonic coordination functions
- Performance monitoring utilities
"""

"""
    FlexibleBodyController

Control system for coordinating harmonic oscillations between multiple flexible bodies
while maintaining specified distances.

# Fields
- `bodies::Vector{FlexibleBody}`: Bodies under control
- `target_distances::Matrix{Float64}`: Target distances between bodies [i,j] 
- `kp::Float64`: Proportional gain for PID control
- `ki::Float64`: Integral gain for PID control  
- `kd::Float64`: Derivative gain for PID control
- `error_integral::Matrix{Float64}`: Accumulated distance errors
- `error_previous::Matrix{Float64}`: Previous distance errors
- `base_frequency::Float64`: Base oscillation frequency
- `phase_offsets::Vector{Float64}`: Phase offset for each body
- `amplitude_limits::Vector{Tuple{Float64, Float64}}`: Min/max amplitude constraints
- `control_points::Vector{Symbol}`: Which points to use for distance calculation
- `custom_control_indices::Vector{Int}`: Custom point indices if needed
- `control_scale_factor::Float64`: Scaling factor for control signal amplitude
"""
mutable struct FlexibleBodyController
    bodies::Vector{FlexibleBody}
    target_distances::Matrix{Float64}
    kp::Float64  # PID gains
    ki::Float64
    kd::Float64
    error_integral::Matrix{Float64}
    error_previous::Matrix{Float64}
    base_frequency::Float64
    phase_offsets::Vector{Float64}
    amplitude_limits::Vector{Tuple{Float64, Float64}}
    control_points::Vector{Symbol}
    custom_control_indices::Vector{Int}
    control_scale_factor::Float64
    
    function FlexibleBodyController(bodies::Vector{FlexibleBody};
                                  base_frequency::Float64 = 1.0,
                                  kp::Float64 = 0.5,
                                  ki::Float64 = 0.1,
                                  kd::Float64 = 0.05,
                                  phase_coordination::Symbol = :synchronized,
                                  control_scale_factor::Float64 = 0.1)
        
        n_bodies = length(bodies)
        target_distances = zeros(n_bodies, n_bodies)
        error_integral = zeros(n_bodies, n_bodies)
        error_previous = zeros(n_bodies, n_bodies)
        
        # Set phase coordination strategy
        phase_offsets = _compute_phase_offsets(n_bodies, phase_coordination)
        
        # Default amplitude limits (10% to 200% of original)
        amplitude_limits = [(0.1, 2.0) for _ in 1:n_bodies]
        
        # Default to trailing edge control
        control_points = fill(:trailing_edge, n_bodies)
        custom_control_indices = Int[]
        
        new(bodies, target_distances, kp, ki, kd, error_integral, error_previous,
            base_frequency, phase_offsets, amplitude_limits, control_points, 
            custom_control_indices, control_scale_factor)
    end
end

"""
    _compute_phase_offsets(n_bodies::Int, coordination::Symbol)

Compute phase offsets for different coordination strategies.
"""
function _compute_phase_offsets(n_bodies::Int, coordination::Symbol)
    if coordination == :synchronized
        return zeros(n_bodies)
    elseif coordination == :alternating
        return [(i-1) * π for i in 1:n_bodies]
    elseif coordination == :sequential
        return [(i-1) * 2π/n_bodies for i in 1:n_bodies]
    else
        @warn "Unknown phase coordination: $coordination, using synchronized"
        return zeros(n_bodies)
    end
end

"""
    set_target_distances!(controller::FlexibleBodyController, distance_matrix::Matrix{Float64})

Set target distances between bodies in the controller.

# Arguments
- `controller::FlexibleBodyController`: Controller to update
- `distance_matrix::Matrix{Float64}`: Target distances [i,j] where i,j are body indices
"""
function set_target_distances!(controller::FlexibleBodyController, distance_matrix::Matrix{Float64})
    n_bodies = length(controller.bodies)
    
    if size(distance_matrix) != (n_bodies, n_bodies)
        error("Distance matrix size $(size(distance_matrix)) doesn't match number of bodies ($n_bodies)")
    end
    
    controller.target_distances = copy(distance_matrix)
    
    # Ensure diagonal is zero and matrix is symmetric for consistency
    for i in 1:n_bodies
        controller.target_distances[i, i] = 0.0
        for j in i+1:n_bodies
            if controller.target_distances[i, j] != controller.target_distances[j, i]
                @warn "Target distance matrix not symmetric at [$i,$j], using average"
                avg_distance = (controller.target_distances[i, j] + controller.target_distances[j, i]) / 2
                controller.target_distances[i, j] = avg_distance
                controller.target_distances[j, i] = avg_distance
            end
        end
    end
end

"""
    set_control_parameters!(controller::FlexibleBodyController; kp=nothing, ki=nothing, kd=nothing)

Update PID control parameters.

# Arguments
- `controller::FlexibleBodyController`: Controller to update
- `kp`, `ki`, `kd`: New PID gains (optional, only specified ones are updated)
"""
function set_control_parameters!(controller::FlexibleBodyController; 
                                kp::Union{Nothing, Float64}=nothing, 
                                ki::Union{Nothing, Float64}=nothing, 
                                kd::Union{Nothing, Float64}=nothing)
    
    if kp !== nothing
        controller.kp = kp
    end
    if ki !== nothing  
        controller.ki = ki
    end
    if kd !== nothing
        controller.kd = kd
    end
    
    println("Updated PID gains: Kp=$(controller.kp), Ki=$(controller.ki), Kd=$(controller.kd)")
end

"""
    reset_controller_state!(controller::FlexibleBodyController)

Reset integral and derivative terms in the controller.
"""
function reset_controller_state!(controller::FlexibleBodyController)
    fill!(controller.error_integral, 0.0)
    fill!(controller.error_previous, 0.0)
    println("Controller state reset: integral and derivative terms cleared")
end

"""
    update_controller!(controller::FlexibleBodyController, current_time::Float64, dt::Float64)

Update the PID control system to maintain target distances between bodies.

# Arguments  
- `controller::FlexibleBodyController`: Control system state
- `current_time::Float64`: Current simulation time
- `dt::Float64`: Time step

# Returns
- `Vector{Float64}`: Updated amplitudes for each body
"""
function update_controller!(controller::FlexibleBodyController, current_time::Float64, dt::Float64)
    
    n_bodies = length(controller.bodies)
    updated_amplitudes = zeros(n_bodies)
    
    # Initialize with current amplitudes
    for i in 1:n_bodies
        updated_amplitudes[i] = controller.bodies[i].amplitude
    end
    
    # Compute current distances and apply control
    for i in 1:n_bodies
        for j in i+1:n_bodies
            if controller.target_distances[i, j] > 0.0  # Only if target distance is set
                
                # Measure current distance
                current_distance = compute_body_distance(
                    controller.bodies[i], controller.bodies[j],
                    controller.control_points[i], controller.control_points[j],
                    _get_custom_index(controller, i), _get_custom_index(controller, j)
                )
                
                # Calculate error
                error = controller.target_distances[i, j] - current_distance
                
                # PID control calculation
                controller.error_integral[i, j] += error * dt
                error_derivative = (error - controller.error_previous[i, j]) / dt
                
                control_signal = (controller.kp * error + 
                                controller.ki * controller.error_integral[i, j] + 
                                controller.kd * error_derivative)
                
                # Apply control signal as amplitude adjustment to trailing body (j)
                amplitude_adjustment = control_signal * controller.control_scale_factor
                
                # Update trailing body amplitude with constraints
                new_amplitude = controller.bodies[j].amplitude + amplitude_adjustment
                
                # Apply amplitude limits
                min_amp, max_amp = controller.amplitude_limits[j]
                original_amplitude = controller.bodies[j].amplitude
                new_amplitude = max(min_amp * abs(original_amplitude), 
                                  min(max_amp * abs(original_amplitude), abs(new_amplitude)))
                
                # Preserve sign of original amplitude
                new_amplitude = sign(original_amplitude) * new_amplitude
                
                updated_amplitudes[j] = new_amplitude
                
                # Store current error for next iteration
                controller.error_previous[i, j] = error
            end
        end
    end
    
    return updated_amplitudes
end

"""
    _get_custom_index(controller::FlexibleBodyController, body_index::Int)

Get custom control index for a body, with bounds checking.
"""
function _get_custom_index(controller::FlexibleBodyController, body_index::Int)
    if body_index <= length(controller.custom_control_indices)
        return controller.custom_control_indices[body_index]
    else
        return 0
    end
end

"""
    apply_harmonic_boundary_conditions!(controller::FlexibleBodyController, 
                                       current_time::Float64, dt::Float64)

Apply coordinated harmonic boundary conditions to all bodies in the controller.

# Arguments
- `controller::FlexibleBodyController`: Control system
- `current_time::Float64`: Current simulation time  
- `dt::Float64`: Time step
"""
function apply_harmonic_boundary_conditions!(controller::FlexibleBodyController, 
                                           current_time::Float64, dt::Float64)
    
    # Update control system
    updated_amplitudes = update_controller!(controller, current_time, dt)
    
    # Apply boundary conditions to each body
    for (i, body) in enumerate(controller.bodies)
        if body.bc_type == :sinusoidal_front
            # Use updated amplitude
            amplitude = updated_amplitudes[i]
            
            # Apply harmonic motion with phase offset
            phase = 2π * controller.base_frequency * current_time + controller.phase_offsets[i]
            displacement = amplitude * sin(phase)
            
            # Update leading edge position (assuming z-direction motion)
            # Store original position for reference
            if !hasfield(typeof(body), :original_position)
                # This is a conceptual update - in practice, the flexible body dynamics
                # would handle the actual position updates
            end
            
            # Update body amplitude for consistency
            body.amplitude = amplitude
        end
    end
end

"""
    monitor_distance_control(controller::FlexibleBodyController, current_time::Float64)

Monitor and report the performance of the distance control system.

# Arguments
- `controller::FlexibleBodyController`: Control system to monitor
- `current_time::Float64`: Current simulation time

# Returns
- `NamedTuple`: Control performance metrics
"""
function monitor_distance_control(controller::FlexibleBodyController, current_time::Float64)
    
    n_bodies = length(controller.bodies)
    current_distances = zeros(n_bodies, n_bodies)
    distance_errors = zeros(n_bodies, n_bodies)
    control_active = zeros(Bool, n_bodies, n_bodies)
    
    # Compute current state
    for i in 1:n_bodies
        for j in i+1:n_bodies
            if controller.target_distances[i, j] > 0.0
                current_distances[i, j] = compute_body_distance(
                    controller.bodies[i], controller.bodies[j],
                    controller.control_points[i], controller.control_points[j]
                )
                distance_errors[i, j] = controller.target_distances[i, j] - current_distances[i, j]
                control_active[i, j] = true
                
                # Symmetric entries
                current_distances[j, i] = current_distances[i, j]
                distance_errors[j, i] = distance_errors[i, j]
                control_active[j, i] = true
            end
        end
    end
    
    # Collect body amplitudes and phases
    amplitudes = [body.amplitude for body in controller.bodies]
    phases = [controller.base_frequency * current_time + offset for offset in controller.phase_offsets]
    
    # Calculate performance metrics
    active_errors = distance_errors[control_active]
    max_error = isempty(active_errors) ? 0.0 : maximum(abs.(active_errors))
    rms_error = isempty(active_errors) ? 0.0 : sqrt(mean(active_errors.^2))
    
    return (
        time = current_time,
        target_distances = controller.target_distances,
        current_distances = current_distances,
        distance_errors = distance_errors,
        control_active = control_active,
        body_amplitudes = amplitudes,
        body_phases = phases,
        error_integrals = controller.error_integral,
        max_distance_error = max_error,
        rms_distance_error = rms_error,
        n_active_controls = sum(control_active) ÷ 2  # Divide by 2 since matrix is symmetric
    )
end

"""
    print_controller_status(controller::FlexibleBodyController, current_time::Float64)

Print detailed status of the controller for debugging.
"""
function print_controller_status(controller::FlexibleBodyController, current_time::Float64)
    println("\n Controller Status at t=$(current_time):")
    println("   Bodies: $(length(controller.bodies))")
    println("   Base frequency: $(controller.base_frequency) Hz")
    println("   PID gains: Kp=$(controller.kp), Ki=$(controller.ki), Kd=$(controller.kd)")
    
    # Show active target distances
    n_bodies = length(controller.bodies)
    active_targets = []
    for i in 1:n_bodies
        for j in i+1:n_bodies
            if controller.target_distances[i, j] > 0.0
                push!(active_targets, (i, j, controller.target_distances[i, j]))
            end
        end
    end
    
    if !isempty(active_targets)
        println("   Active distance targets:")
        for (i, j, target) in active_targets
            current_dist = compute_body_distance(
                controller.bodies[i], controller.bodies[j],
                controller.control_points[i], controller.control_points[j]
            )
            error = target - current_dist
            @printf "     Body %d--%d: target=%.3f, current=%.3f, error=%.3f\n" i j target current_dist error
        end
    end
    
    # Show current amplitudes
    println("   Current amplitudes:")
    for (i, body) in enumerate(controller.bodies)
        @printf "     Body %d: %.4f\n" i body.amplitude
    end
end