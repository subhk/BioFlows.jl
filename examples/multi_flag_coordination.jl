# Multi-Flag Coordination Example
# Three flexible flags with coordinated motion and distance control
# Demonstrates phase coordination and multi-body PID control
# Run: julia --project examples/multi_flag_coordination.jl

using BioFlows

function main()
    println("=== Multi-Flag Coordination Example ===")
    
    # Create three flags in a triangular formation
    flag_positions = [
        [1.5, 1.0],   # Flag 1: bottom left
        [1.5, 3.0],   # Flag 2: top
        [3.0, 2.0]    # Flag 3: right
    ]
    
    flag_length = 1.2
    flag_width = 0.04
    
    println("Creating coordinated flag system...")
    println("Flag positions:")
    for (i, pos) in enumerate(flag_positions)
        println("  Flag $i: [$(pos[1]), $(pos[2])]")
    end
    
    # Create flags with different initial phases
    flags = FlexibleBody[]
    
    for (i, pos) in enumerate(flag_positions)
        flag = create_flag(pos, flag_length, flag_width;
                          material = :flexible,
                          prescribed_motion = (type = :sinusoidal, amplitude = 0.15, frequency = 2.0),
                          attachment = :fixed_leading_edge,
                          n_points = 12,
                          id = i)
        push!(flags, flag)
    end
    
    # Create PID controller with sequential phase coordination
    controller = FlexibleBodyController(flags;
                                       base_frequency = 2.0,
                                       kp = 0.8,
                                       ki = 0.2,
                                       kd = 0.1,
                                       phase_coordination = :sequential)  # 120° phase differences
    
    println("\nPID Controller Setup:")
    println("  Base frequency: $(controller.base_frequency) Hz")
    println("  PID gains: Kp=$(controller.kp), Ki=$(controller.ki), Kd=$(controller.kd)")
    println("  Phase coordination: sequential")
    println("  Phase offsets: $(round.(controller.phase_offsets, digits=3)) rad")
    
    # Set target distances to form equilateral triangle
    target_distance = 2.0  # Desired side length
    target_matrix = [0.0           target_distance  target_distance;
                    target_distance 0.0            target_distance;
                    target_distance target_distance 0.0]
    
    set_target_distances!(controller, target_matrix)
    
    println("\nTarget Formation: Equilateral triangle")
    println("  Target side length: $(target_distance)")
    
    # Simulate coordinated motion
    dt = 0.01
    total_time = 15.0  # 15 seconds
    n_steps = Int(total_time / dt)
    
    # Storage for analysis
    time_history = Float64[]
    distance_history = Matrix{Float64}(undef, 3, 3, 0)  # 3x3 distance matrix over time
    amplitude_history = Vector{Float64}[]
    
    println("\nSimulating coordinated motion...")
    println("Steps: $n_steps, Time step: $dt, Total time: $(total_time)s")
    
    current_time = 0.0
    
    for step = 1:n_steps
        # Update PID controller
        updated_amplitudes = update_controller!(controller, current_time, dt)
        apply_harmonic_boundary_conditions!(controller, current_time, dt)
        
        # Store history every 50 steps (0.5 second intervals)
        if step % 50 == 0
            push!(time_history, current_time)
            
            # Compute all pairwise distances
            current_distances = zeros(3, 3)
            for i = 1:3, j = 1:3
                if i != j
                    current_distances[i, j] = compute_body_distance(
                        flags[i], flags[j], :trailing_edge, :trailing_edge
                    )
                end
            end
            
            # Store distances
            if isempty(distance_history)
                distance_history = reshape(current_distances, 3, 3, 1)
            else
                distance_history = cat(distance_history, reshape(current_distances, 3, 3, 1), dims=3)
            end
            
            push!(amplitude_history, copy(updated_amplitudes))
            
            # Print progress every 5 seconds
            if step % 500 == 0
                println("  t=$(round(current_time, digits=1))s: " *
                       "distances=[$(round.(current_distances[1, 2:3], digits=3))] " *
                       "amplitudes=[$(round.(updated_amplitudes, digits=3))]")
            end
        end
        
        current_time += dt
    end
    
    # Analysis of coordination performance
    println("\n=== Coordination Analysis ===")
    
    # Final distances
    final_distances = distance_history[:, :, end]
    
    println("Final Inter-Flag Distances:")
    for i = 1:3, j = i+1:3
        actual = final_distances[i, j]
        target = target_matrix[i, j]
        error = abs(actual - target)
        error_pct = 100 * error / target
        
        println("  Flag $i ↔ Flag $j: $(round(actual, digits=3)) " *
               "(target: $(target), error: $(round(error, digits=3)), $(round(error_pct, digits=1))%)")
    end
    
    # Phase coordination analysis
    println("\nPhase Coordination:")
    final_amplitudes = amplitude_history[end]
    for (i, (amplitude, phase)) in enumerate(zip(final_amplitudes, controller.phase_offsets))
        phase_deg = round(rad2deg(phase), digits=1)
        println("  Flag $i: amplitude=$(round(amplitude, digits=3)), phase=$(phase_deg)°")
    end
    
    # Distance stability over time
    if length(time_history) > 10
        recent_distances = distance_history[:, :, end-9:end]  # Last 10 measurements
        distance_12 = [recent_distances[1, 2, k] for k in 1:size(recent_distances, 3)]
        distance_13 = [recent_distances[1, 3, k] for k in 1:size(recent_distances, 3)]
        distance_23 = [recent_distances[2, 3, k] for k in 1:size(recent_distances, 3)]
        
        stability_12 = std(distance_12)
        stability_13 = std(distance_13)
        stability_23 = std(distance_23)
        
        println("\nDistance Stability (last 5 seconds):")
        println("  Flag 1-2 std dev: $(round(stability_12, digits=4))")
        println("  Flag 1-3 std dev: $(round(stability_13, digits=4))")
        println("  Flag 2-3 std dev: $(round(stability_23, digits=4))")
        
        avg_stability = (stability_12 + stability_13 + stability_23) / 3
        if avg_stability < 0.05
            println("  Excellent coordination stability!")
        elseif avg_stability < 0.1
            println("  Good coordination stability")
        else
            println("  WARNING: Coordination needs improvement")
        end
    end
    
    # Performance metrics
    performance = monitor_distance_control(controller, current_time)
    println("\nOverall Controller Performance:")
    println("  RMS distance error: $(round(performance.rms_distance_error, digits=4))")
    println("  Maximum distance error: $(round(performance.max_distance_error, digits=4))")
    println("  Active control pairs: $(performance.n_active_controls)")
    
    # Formation quality assessment
    formation_error = 0.0
    n_pairs = 0
    for i = 1:3, j = i+1:3
        formation_error += abs(final_distances[i, j] - target_distance)
        n_pairs += 1
    end
    formation_error /= n_pairs
    formation_error_pct = 100 * formation_error / target_distance
    
    println("\nFormation Quality:")
    println("  Average distance error: $(round(formation_error, digits=4)) ($(round(formation_error_pct, digits=1))%)")
    
    if formation_error_pct < 5.0
        println("  Excellent formation control!")
    elseif formation_error_pct < 10.0
        println("  Good formation control")
    else
        println("  WARNING: Formation control needs tuning")
    end
    
    return controller, time_history, distance_history, amplitude_history
end

# Helper function to analyze phase relationships
function analyze_phase_coordination(amplitude_history, phase_offsets, time_history)
    println("\n=== Phase Coordination Analysis ===")
    
    n_flags = length(phase_offsets)
    final_time = time_history[end]
    
    # Theoretical phase differences
    println("Theoretical Phase Differences:")
    for i = 1:n_flags
        phase_deg = round(rad2deg(phase_offsets[i]), digits=1)
        println("  Flag $i: $(phase_deg)° relative to Flag 1")
    end
    
    # Check if amplitudes maintain phase relationships
    recent_amps = amplitude_history[max(1, end-10):end]  # Last few measurements
    avg_amps = [mean([amps[i] for amps in recent_amps]) for i in 1:n_flags]
    
    println("\nAverage Final Amplitudes:")
    for (i, amp) in enumerate(avg_amps)
        println("  Flag $i: $(round(amp, digits=3))")
    end
    
    return avg_amps
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    controller, time_history, distance_history, amplitude_history = main()
    analyze_phase_coordination(amplitude_history, controller.phase_offsets, time_history)
    println("\nMulti-flag coordination example completed!")
end