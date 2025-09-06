abstract type TimeSteppingScheme <: AbstractTimeStepping end

struct AdamsBashforth <: TimeSteppingScheme
    order::Int
    history::Vector{Vector{Float64}}
    coefficients::Vector{Float64}
end

struct RungeKutta2 <: TimeSteppingScheme end

struct RungeKutta4 <: TimeSteppingScheme end

function AdamsBashforth(order::Int=3)
    if order == 1
        coeffs = [1.0]
    elseif order == 2
        coeffs = [3/2, -1/2]
    elseif order == 3
        coeffs = [23/12, -16/12, 5/12]
    elseif order == 4
        coeffs = [55/24, -59/24, 37/24, -9/24]
    else
        error("Adams-Bashforth order $order not implemented")
    end
    
    AdamsBashforth(order, Vector{Float64}[], coeffs)
end

function adams_bashforth_step!(state_new::SolutionState, state_old::SolutionState,
                              rhs_func::Function, scheme::AdamsBashforth, 
                              dt::Float64, args...)
    # Compute current RHS
    current_rhs = rhs_func(state_old, args...)
    
    # Store current RHS in history
    push!(scheme.history, deepcopy(current_rhs))
    
    # Keep only required history
    if length(scheme.history) > scheme.order
        popfirst!(scheme.history)
    end
    
    # Apply Adams-Bashforth formula
    if length(scheme.history) == 1
        # First step: use forward Euler
        adams_bashforth_update!(state_new, state_old, scheme.history[1], dt, [1.0])
    else
        # Multi-step Adams-Bashforth
        n_steps = min(length(scheme.history), scheme.order)
        coeffs = scheme.coefficients[1:n_steps]
        adams_bashforth_update!(state_new, state_old, scheme.history, dt, coeffs)
    end
    
    state_new.t = state_old.t + dt
    state_new.step = state_old.step + 1
end

function adams_bashforth_update!(state_new::SolutionState, state_old::SolutionState,
                               rhs_history::Vector, dt::Float64, coefficients::Vector{Float64})
    # Update velocities using Adams-Bashforth scheme
    state_new.u .= state_old.u
    state_new.v .= state_old.v
    if !isempty(state_old.w)
        state_new.w .= state_old.w
    end
    state_new.p .= state_old.p
    
    # Apply Adams-Bashforth formula: u^{n+1} = u^n + dt * Σ(β_k * f^{n-k})
    for (k, coeff) in enumerate(coefficients)
        rhs = rhs_history[end-k+1]  # Most recent first
        state_new.u .+= dt * coeff .* rhs.u
        state_new.v .+= dt * coeff .* rhs.v
        if !isempty(state_old.w) && haskey(rhs, :w)
            state_new.w .+= dt * coeff .* rhs.w
        end
    end
end

function adams_bashforth_update!(state_new::SolutionState, state_old::SolutionState,
                               rhs::NamedTuple, dt::Float64, coefficients::Vector{Float64})
    # Single RHS case (first step)
    state_new.u .= state_old.u .+ dt .* coefficients[1] .* rhs.u
    state_new.v .= state_old.v .+ dt .* coefficients[1] .* rhs.v
    if !isempty(state_old.w) && haskey(rhs, :w)
        state_new.w .= state_old.w .+ dt .* coefficients[1] .* rhs.w
    end
    state_new.p .= state_old.p
end

function runge_kutta2_step!(state_new::SolutionState, state_old::SolutionState,
                           rhs_func::Function, scheme::RungeKutta2,
                           dt::Float64, args...)
    # Heun's method (RK2): average of Euler predictor and corrected slope
    # k1 at old state
    k1 = rhs_func(state_old, args...)
    # Predictor
    temp = deepcopy(state_old)
    temp.u .+= dt .* k1.u
    temp.v .+= dt .* k1.v
    if !isempty(state_old.w)
        temp.w .+= dt .* k1.w
    end
    temp.t = state_old.t + dt
    # k2 at predicted state
    k2 = rhs_func(temp, args...)
    # Final update
    state_new.u .= state_old.u .+ 0.5 .* dt .* (k1.u .+ k2.u)
    state_new.v .= state_old.v .+ 0.5 .* dt .* (k1.v .+ k2.v)
    if !isempty(state_old.w)
        state_new.w .= state_old.w .+ 0.5 .* dt .* (k1.w .+ k2.w)
    end
    state_new.p .= state_old.p
    state_new.t = state_old.t + dt
    state_new.step = state_old.step + 1
end

function runge_kutta4_step!(state_new::SolutionState, state_old::SolutionState,
                           rhs_func::Function, scheme::RungeKutta4,
                           dt::Float64, args...)
    # Classic RK4
    # Stage 1
    k1 = rhs_func(state_old, args...)
    state_temp1 = deepcopy(state_old)
    state_temp1.u .+= dt .* k1.u ./ 2.0
    state_temp1.v .+= dt .* k1.v ./ 2.0
    if !isempty(state_old.w)
        state_temp1.w .+= dt .* k1.w ./ 2.0
    end
    state_temp1.t = state_old.t + dt/2
    
    # Stage 2
    k2 = rhs_func(state_temp1, args...)
    state_temp2 = deepcopy(state_old)
    state_temp2.u .+= dt .* k2.u ./ 2.0
    state_temp2.v .+= dt .* k2.v ./ 2.0
    if !isempty(state_old.w)
        state_temp2.w .+= dt .* k2.w ./ 2.0
    end
    state_temp2.t = state_old.t + dt/2
    
    # Stage 3
    k3 = rhs_func(state_temp2, args...)
    state_temp3 = deepcopy(state_old)
    state_temp3.u .+= dt .* k3.u
    state_temp3.v .+= dt .* k3.v
    if !isempty(state_old.w)
        state_temp3.w .+= dt .* k3.w
    end
    state_temp3.t = state_old.t + dt
    
    # Stage 4
    k4 = rhs_func(state_temp3, args...)
    
    # Final update
    state_new.u .= state_old.u .+ dt .* (k1.u .+ 2.0 .* k2.u .+ 2.0 .* k3.u .+ k4.u) ./ 6.0
    state_new.v .= state_old.v .+ dt .* (k1.v .+ 2.0 .* k2.v .+ 2.0 .* k3.v .+ k4.v) ./ 6.0
    if !isempty(state_old.w)
        state_new.w .= state_old.w .+ dt .* (k1.w .+ 2.0 .* k2.w .+ 2.0 .* k3.w .+ k4.w) ./ 6.0
    end
    state_new.p .= state_old.p
    state_new.t = state_old.t + dt
    state_new.step = state_old.step + 1
end

function time_step!(state_new::SolutionState, state_old::SolutionState,
                   rhs_func::Function, scheme::TimeSteppingScheme,
                   dt::Float64, args...)
    if scheme isa AdamsBashforth
        adams_bashforth_step!(state_new, state_old, rhs_func, scheme, dt, args...)
    elseif scheme isa RungeKutta2
        runge_kutta2_step!(state_new, state_old, rhs_func, scheme, dt, args...)
    elseif scheme isa RungeKutta4
        runge_kutta4_step!(state_new, state_old, rhs_func, scheme, dt, args...)
    else
        error("Unknown time stepping scheme: $(typeof(scheme))")
    end
end
