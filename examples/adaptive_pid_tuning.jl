# Adaptive PID Tuning Example
# Demonstrates automatic PID gain tuning based on performance metrics
# Run: julia --project examples/adaptive_pid_tuning.jl

using BioFlows

function main()
    println("=== Adaptive PID Tuning Example ===")
    
    # Create two flags with challenging initial conditions
    flag1 = create_flag([2.0, 1.0], 1.0, 0.05;
                        prescribed_motion = (type = :sinusoidal, amplitude = 0.3, frequency = 1.8),
                        material = :very_flexible)  # More challenging dynamics
    
    flag2 = create_flag([2.0, 4.0], 1.0, 0.05;  # Large initial separation
                        prescribed_motion = (type = :sinusoidal, amplitude = 0.3, frequency = 1.8),
                        material = :very_flexible)
    
    println("Initial setup:")
    println("  Flag 1 position: [2.0, 1.0]")
    println("  Flag 2 position: [2.0, 4.0]") 
    println("  Initial distance: 3.0")
    println("  Material: very flexible (challenging dynamics)")
    
    # Start with conservative PID gains
    initial_gains = (kp = 0.2, ki = 0.05, kd = 0.02)
    controller = FlexibleBodyController([flag1, flag2];
                                       kp = initial_gains.kp,
                                       ki = initial_gains.ki,
                                       kd = initial_gains.kd,
                                       base_frequency = 1.8)
    
    # Set challenging target distance
    target_distance = 1.5  # 50% reduction from initial
    set_target_distances!(controller, [0.0 target_distance; target_distance 0.0])
    
    println("\nInitial PID gains: $(initial_gains)")
    println("Target distance: $(target_distance) (50% reduction)")
    
    # Adaptive tuning parameters
    adaptation_period = 100  # Steps between adaptations
    max_adaptations = 10
    performance_window = 50   # Steps to average for performance
    
    # Performance tracking
    performance_history = []
    gain_history = []
    distance_history = []
    
    # Simulation parameters
    dt = 0.01
    current_time = 0.0
    step = 0
    adaptation_count = 0
    
    println("\nStarting adaptive PID tuning simulation...")
    println("Adaptation period: $adaptation_period steps")
    println("Maximum adaptations: $max_adaptations")
    
    # Main simulation loop with adaptive tuning
    while adaptation_count < max_adaptations
        step += 1
        current_time = step * dt
        
        # Update controller
        updated_amplitudes = update_controller!(controller, current_time, dt)
        apply_harmonic_boundary_conditions!(controller, current_time, dt)
        
        # Measure performance
        current_distance = compute_body_distance(flag1, flag2, :trailing_edge, :trailing_edge)
        error = abs(target_distance - current_distance)
        
        # Store performance data
        push!(distance_history, current_distance)
        
        # Print regular updates
        if step % 200 == 0
            println("  Step $step: distance=$(round(current_distance, digits=3)), " *
                   "error=$(round(error, digits=3)), " *
                   "gains=[$(round(controller.kp, digits=3)), $(round(controller.ki, digits=3)), $(round(controller.kd, digits=3))]")
        end
        
        # Adaptive tuning check
        if step % adaptation_period == 0
            adaptation_count += 1
            
            # Calculate recent performance metrics
            recent_distances = distance_history[max(1, end-performance_window+1):end]
            recent_errors = [abs(target_distance - d) for d in recent_distances]
            
            avg_error = mean(recent_errors)
            max_error = maximum(recent_errors)
            error_std = std(recent_errors)
            
            # Store performance
            performance_metrics = (
                step = step,
                avg_error = avg_error,
                max_error = max_error,
                stability = error_std,
                distance = mean(recent_distances)
            )
            push!(performance_history, performance_metrics)
            
            # Store current gains
            current_gains = (kp = controller.kp, ki = controller.ki, kd = controller.kd)
            push!(gain_history, current_gains)
            
            println("\n--- Adaptation $adaptation_count ---")
            println("Recent performance:")
            println("  Average error: $(round(avg_error, digits=4))")
            println("  Maximum error: $(round(max_error, digits=4))")
            println("  Stability (std): $(round(error_std, digits=4))")
            println("  Current gains: Kp=$(round(controller.kp, digits=3)), Ki=$(round(controller.ki, digits=3)), Kd=$(round(controller.kd, digits=3))")
            
            # Adaptive tuning logic
            new_gains = adapt_pid_gains(controller, avg_error, max_error, error_std, target_distance)
            
            if new_gains != current_gains
                set_control_parameters!(controller; 
                                      kp = new_gains.kp,
                                      ki = new_gains.ki, 
                                      kd = new_gains.kd)
                println("  → Updated gains: Kp=$(round(new_gains.kp, digits=3)), Ki=$(round(new_gains.ki, digits=3)), Kd=$(round(new_gains.kd, digits=3))")
                
                # Reset integral term when gains change significantly
                if abs(new_gains.ki - current_gains.ki) > 0.02
                    reset_controller_state!(controller)
                    println("  → Reset controller state due to significant Ki change")
                end
            else
                println("  → No gain adjustment needed")
            end
        end
    end
    
    # Final analysis
    println("\n=== Adaptive Tuning Results ===")
    
    final_distance = distance_history[end]
    final_error = abs(target_distance - final_distance)
    final_error_pct = 100 * final_error / target_distance
    
    println("Performance Evolution:")
    println("  Initial gains: $(gain_history[1])")
    println("  Final gains:   (kp=$(round(controller.kp, digits=3)), ki=$(round(controller.ki, digits=3)), kd=$(round(controller.kd, digits=3)))")
    println("  Initial error: $(round(abs(target_distance - distance_history[1]), digits=4))")
    println("  Final error:   $(round(final_error, digits=4)) ($(round(final_error_pct, digits=1))%)")
    
    # Performance improvement analysis
    initial_performance = performance_history[1]
    final_performance = performance_history[end]
    
    error_improvement = (initial_performance.avg_error - final_performance.avg_error) / initial_performance.avg_error * 100
    stability_improvement = (initial_performance.stability - final_performance.stability) / initial_performance.stability * 100
    
    println("\nImprovement Metrics:")
    println("  Error reduction: $(round(error_improvement, digits=1))%")
    println("  Stability improvement: $(round(stability_improvement, digits=1))%")
    
    # Convergence assessment
    recent_errors = [abs(target_distance - d) for d in distance_history[end-50:end]]
    convergence_quality = assess_convergence(recent_errors, target_distance)
    
    println("\nConvergence Assessment:")
    println("  Quality: $(convergence_quality.quality)")
    println("  Steady-state error: $(round(convergence_quality.steady_state_error, digits=4))")
    println("  Oscillation amplitude: $(round(convergence_quality.oscillation_amplitude, digits=4))")
    
    if convergence_quality.quality == "Excellent"
        println("  ✓ Adaptive tuning successful!")
    elseif convergence_quality.quality == "Good"
        println("  ✓ Adaptive tuning effective")
    else
        println("  ⚠ Further tuning may be needed")
    end
    
    return controller, performance_history, gain_history, distance_history
end

"""
Adaptive PID gain tuning based on performance metrics
"""
function adapt_pid_gains(controller, avg_error, max_error, error_std, target_distance)
    current_kp = controller.kp
    current_ki = controller.ki
    current_kd = controller.kd
    
    # Performance thresholds
    error_threshold_high = 0.15 * target_distance  # 15% error is high
    error_threshold_low = 0.05 * target_distance   # 5% error is good
    stability_threshold = 0.08 * target_distance   # High oscillation threshold
    
    new_kp = current_kp
    new_ki = current_ki
    new_kd = current_kd
    
    # Tuning logic
    if avg_error > error_threshold_high
        # High error: increase proportional and integral gains
        new_kp = min(2.0, current_kp * 1.2)  # Increase Kp by 20%, max 2.0
        new_ki = min(0.5, current_ki * 1.15) # Increase Ki by 15%, max 0.5
        println("    High error detected → Increasing Kp and Ki")
        
    elseif avg_error < error_threshold_low && error_std > stability_threshold
        # Low error but high oscillation: increase derivative gain
        new_kd = min(0.3, current_kd * 1.3)  # Increase Kd by 30%, max 0.3
        println("    High oscillation detected → Increasing Kd")
        
    elseif avg_error < error_threshold_low && error_std < stability_threshold / 2
        # Very stable and accurate: can be more aggressive
        if current_kp < 1.0
            new_kp = min(1.0, current_kp * 1.1)  # Slightly increase Kp
            println("    Excellent performance → Slightly increasing Kp")
        end
        
    elseif max_error > 2 * error_threshold_high
        # Very high peak errors: need stronger derivative action
        new_kd = min(0.25, current_kd * 1.4)
        println("    High peak errors → Increasing Kd significantly")
    end
    
    # Stability constraints
    if new_kp > 1.5 && new_ki > 0.3
        new_ki = 0.25  # Reduce Ki if Kp is high to maintain stability
        println("    Applying stability constraint → Reducing Ki")
    end
    
    return (kp = new_kp, ki = new_ki, kd = new_kd)
end

"""
Assess convergence quality of the control system
"""
function assess_convergence(recent_errors, target_distance)
    steady_state_error = mean(recent_errors)
    oscillation_amplitude = maximum(recent_errors) - minimum(recent_errors)
    
    # Normalize by target distance
    ss_error_pct = steady_state_error / target_distance * 100
    osc_amp_pct = oscillation_amplitude / target_distance * 100
    
    # Quality assessment
    if ss_error_pct < 3.0 && osc_amp_pct < 5.0
        quality = "Excellent"
    elseif ss_error_pct < 7.0 && osc_amp_pct < 10.0
        quality = "Good"
    elseif ss_error_pct < 15.0 && osc_amp_pct < 20.0
        quality = "Fair"
    else
        quality = "Poor"
    end
    
    return (
        quality = quality,
        steady_state_error = steady_state_error,
        oscillation_amplitude = oscillation_amplitude,
        ss_error_pct = ss_error_pct,
        osc_amp_pct = osc_amp_pct
    )
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    controller, performance_history, gain_history, distance_history = main()
    println("\nAdaptive PID tuning example completed!")
end