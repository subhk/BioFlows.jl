# =============================================================================
# FLUID-STRUCTURE INTERACTION COUPLING
# =============================================================================
# Two-way coupling between fluid solver and Euler-Bernoulli beam:
#
# 1. Fluid → Structure: Compute pressure load q(s) from fluid solution
#    q(s) = ∮ p·n_z dℓ  (integral of pressure around body cross-section)
#
# 2. Structure → Fluid: Update body SDF from beam displacement
#    φ(x, z, t) = |z - z_body(x, t)| - h(x)
#    where z_body(x, t) = z_center + w(x, t)
#
# Coupling schemes:
# - Explicit (staggered): Simple but may be unstable for light structures
# - Implicit (sub-iterations): More stable, requires convergence check
# =============================================================================

using StaticArrays

"""
    FlexibleBodyFSI{T}

Flexible body for FSI simulation combining a beam structure with fluid.

# Fields
- `beam`: EulerBernoulliBeam structure
- `x_head`: x-coordinate of beam head (m)
- `z_center`: z-coordinate of undeformed centerline (m)
- `h_func`: Thickness function h(s) for the body profile
- `ρ_fluid`: Fluid density (kg/m³)
- `active_forcing`: Optional active forcing function f(s, t)
"""
mutable struct FlexibleBodyFSI{T<:Real, F1<:Function, F2<:Union{Function, Nothing}}
    beam::EulerBernoulliBeam{T}
    x_head::T              # Head position x-coordinate
    z_center::T            # Centerline z-coordinate (undeformed)
    h_func::F1             # Thickness function h(s)
    ρ_fluid::T             # Fluid density
    active_forcing::F2     # Active forcing function f(s, t) or nothing

    # Coupling parameters
    sub_iterations::Int    # Number of sub-iterations for implicit coupling
    relaxation::T          # Under-relaxation factor (0 < ω ≤ 1)
    tolerance::T           # Convergence tolerance for sub-iterations
end

"""
    FlexibleBodyFSI(beam; x_head, z_center, h_func, ρ_fluid=1000.0, kwargs...)

Create a flexible body for FSI simulation.

# Arguments
- `beam`: EulerBernoulliBeam structure

# Keyword Arguments
- `x_head`: x-coordinate of beam head (default: 0.0)
- `z_center`: z-coordinate of undeformed centerline
- `h_func`: Thickness function h(s), or use beam geometry
- `ρ_fluid`: Fluid density (default: 1000.0 kg/m³)
- `active_forcing`: Function f(s, t) for muscle activation (default: nothing)
- `sub_iterations`: Number of FSI sub-iterations (default: 3)
- `relaxation`: Under-relaxation factor (default: 0.5)
- `tolerance`: Convergence tolerance (default: 1e-6)
"""
function FlexibleBodyFSI(beam::EulerBernoulliBeam{T};
                         x_head::Real=0f0,
                         z_center::Real=0f0,
                         h_func::Union{Function, Nothing}=nothing,
                         ρ_fluid::Real=1000f0,
                         active_forcing::Union{Function, Nothing}=nothing,
                         sub_iterations::Int=3,
                         relaxation::Real=0.5f0,
                         tolerance::Real=1f-6) where T

    # Use beam geometry thickness if not provided
    h = h_func === nothing ? beam.geometry.thickness : h_func

    FlexibleBodyFSI{T, typeof(h), typeof(active_forcing)}(
        beam, T(x_head), T(z_center), h, T(ρ_fluid), active_forcing,
        sub_iterations, T(relaxation), T(tolerance)
    )
end

"""
    sdf(body::FlexibleBodyFSI, x, t)

Compute the signed distance function for the flexible body at position x and time t.

The body surface is defined by:
    φ(x, z, t) = |z - z_body(x, t)| - h(s)

where:
- z_body(x, t) = z_center + w(s, t) is the deformed centerline
- h(s) is the local half-thickness
- s = x - x_head is the arc length coordinate
"""
function sdf(body::FlexibleBodyFSI{T}, x::AbstractVector, t::Real) where T
    s = x[1] - body.x_head
    L = body.beam.geometry.L

    # Outside the body (before head or after tail)
    if s < 0
        # Distance to head
        w_head = body.beam.w[1]
        z_head = body.z_center + w_head
        h_head = body.h_func(zero(T))
        return sqrt(s^2 + (x[2] - z_head)^2) - h_head
    elseif s > L
        # Distance to tail
        w_tail = body.beam.w[end]
        z_tail = body.z_center + w_tail
        h_tail = body.h_func(L)
        return sqrt((s - L)^2 + (x[2] - z_tail)^2) - h_tail
    end

    # Interpolate displacement from beam nodes
    w_local = interpolate_displacement(body.beam, s)
    z_body = body.z_center + w_local
    h_local = body.h_func(s)

    # Signed distance: positive outside, negative inside
    return abs(x[2] - z_body) - h_local
end

"""
    interpolate_displacement(beam, s)

Interpolate beam displacement at arc length s using linear interpolation.
"""
function interpolate_displacement(beam::EulerBernoulliBeam{T}, s::Real) where T
    n = beam.geometry.n
    L = beam.geometry.L
    Δs = beam.Δs

    # Clamp s to valid range
    s_clamped = clamp(s, zero(T), L)

    # Find bracketing indices
    idx_float = s_clamped / Δs + 1
    i_low = floor(Int, idx_float)
    i_low = clamp(i_low, 1, n-1)
    i_high = i_low + 1

    # Linear interpolation weight
    α = (s_clamped - beam.s[i_low]) / Δs

    return (1 - α) * beam.w[i_low] + α * beam.w[i_high]
end

"""
    interpolate_velocity(beam, s)

Interpolate beam velocity at arc length s.
"""
function interpolate_velocity(beam::EulerBernoulliBeam{T}, s::Real) where T
    n = beam.geometry.n
    L = beam.geometry.L
    Δs = beam.Δs

    s_clamped = clamp(s, zero(T), L)
    idx_float = s_clamped / Δs + 1
    i_low = clamp(floor(Int, idx_float), 1, n-1)
    i_high = i_low + 1
    α = (s_clamped - beam.s[i_low]) / Δs

    return (1 - α) * beam.w_dot[i_low] + α * beam.w_dot[i_high]
end

"""
    body_velocity(body::FlexibleBodyFSI, x, t)

Compute the body velocity at position x and time t.
Returns [0, w_dot] since the beam only moves in the z-direction.
"""
function body_velocity(body::FlexibleBodyFSI{T}, x::AbstractVector, t::Real) where T
    s = x[1] - body.x_head
    L = body.beam.geometry.L

    if s < 0 || s > L
        # Outside body - use nearest endpoint velocity
        s_clamped = clamp(s, zero(T), L)
    else
        s_clamped = s
    end

    w_dot = interpolate_velocity(body.beam, s_clamped)
    return SVector{2,T}(zero(T), w_dot)
end

"""
    compute_fluid_load!(body::FlexibleBodyFSI, flow::Flow, t)

Compute the fluid pressure load on the beam from the flow solution.

The load at each beam node is computed by integrating pressure around
the local cross-section:
    q(s) = ∫ p(s, z) · n_z dz ≈ Δp(s) · b(s)

where Δp is the pressure difference across the body and b is the width.
"""
function compute_fluid_load!(body::FlexibleBodyFSI{T}, flow, t::Real) where T
    beam = body.beam
    n = beam.geometry.n

    for i in 1:n
        s = beam.s[i]
        x_pos = body.x_head + s
        w_local = beam.w[i]
        z_body = body.z_center + w_local
        h_local = body.h_func(s)

        # Sample pressure above and below the body (half-cell offset)
        z_offset = T(0.5) * flow.Δx[2]
        p_above = sample_pressure(flow, x_pos, z_body + h_local + z_offset)
        p_below = sample_pressure(flow, x_pos, z_body - h_local - z_offset)

        # Pressure load (force per unit length in z-direction)
        # Positive pressure below pushes up, positive pressure above pushes down
        Δp = p_below - p_above
        b_local = beam.geometry.width(s)

        beam.q[i] = Δp * b_local
    end

    return beam.q
end

"""
    sample_pressure(flow, x, z)

Sample the pressure field at position (x, z) using bilinear interpolation.
"""
function sample_pressure(flow, x::Real, z::Real)
    T = eltype(flow.p)
    nx, nz = size(flow.p)

    # Convert physical coordinates to grid indices
    # Assuming uniform grid with origin at (0, 0)
    Δx = flow.Δx[1]
    Δz = flow.Δx[2]

    # Grid indices (1-based, cell-centered)
    i_float = T(x) / T(Δx) + T(1.5)
    j_float = T(z) / T(Δz) + T(1.5)

    # Clamp to valid range
    i_float = clamp(i_float, one(T), T(nx))
    j_float = clamp(j_float, one(T), T(nz))

    i_low = floor(Int, i_float)
    j_low = floor(Int, j_float)
    i_low = clamp(i_low, 1, nx-1)
    j_low = clamp(j_low, 1, nz-1)
    i_high = i_low + 1
    j_high = j_low + 1

    # Bilinear interpolation weights
    α = i_float - i_low
    β = j_float - j_low

    # Bilinear interpolation
    p = (1-α)*(1-β)*flow.p[i_low, j_low] +
        α*(1-β)*flow.p[i_high, j_low] +
        (1-α)*β*flow.p[i_low, j_high] +
        α*β*flow.p[i_high, j_high]

    return T(p)
end

"""
    fsi_step!(body::FlexibleBodyFSI, flow, t, Δt; explicit=false)

Perform one FSI time step with two-way coupling.

# Coupling Algorithm (Implicit with sub-iterations):
1. Save current beam state
2. For each sub-iteration:
   a. Compute fluid load from current flow
   b. Set active forcing (if any)
   c. Advance beam by Δt
   d. Check convergence
3. Update body SDF for next fluid step

# Arguments
- `body`: FlexibleBodyFSI structure
- `flow`: Flow structure
- `t`: Current time
- `Δt`: Time step
- `explicit`: If true, use explicit (single-step) coupling (default: false)
"""
function fsi_step!(body::FlexibleBodyFSI{T}, flow, t::Real, Δt::Real;
                   explicit::Bool=false) where T

    beam = body.beam

    if explicit
        # Simple explicit coupling (one iteration)
        compute_fluid_load!(body, flow, t)
        if body.active_forcing !== nothing
            set_active_forcing!(beam, body.active_forcing, t)
        end
        step!(beam, T(Δt))
    else
        # Implicit coupling with sub-iterations
        u_base = copy(beam.u)
        u_dot_base = copy(beam.u_dot)
        u_ddot_base = copy(beam.u_ddot)
        u_iter = copy(beam.u)
        u_dot_iter = copy(beam.u_dot)
        u_ddot_iter = copy(beam.u_ddot)

        for iter in 1:body.sub_iterations
            # Compute fluid load using current iterate geometry
            beam.u .= u_iter
            compute_fluid_load!(body, flow, t)

            # Set active forcing
            if body.active_forcing !== nothing
                set_active_forcing!(beam, body.active_forcing, t)
            end

            # Advance beam from the same base state for this time step
            beam.u .= u_base
            beam.u_dot .= u_dot_base
            beam.u_ddot .= u_ddot_base
            step!(beam, T(Δt))

            # Apply under-relaxation to the full state
            ω = body.relaxation
            u_prev = copy(u_iter)
            u_iter .= ω .* beam.u .+ (one(T) - ω) .* u_iter
            u_dot_iter .= ω .* beam.u_dot .+ (one(T) - ω) .* u_dot_iter
            u_ddot_iter .= ω .* beam.u_ddot .+ (one(T) - ω) .* u_ddot_iter

            # Check convergence
            Δw = maximum(abs.(u_iter[1:2:end] .- u_prev[1:2:end]))
            if Δw < body.tolerance
                break
            end
        end

        # Store relaxed state
        beam.u .= u_iter
        beam.u_dot .= u_dot_iter
        beam.u_ddot .= u_ddot_iter
    end

    return body
end

"""
    create_fsi_body(body::FlexibleBodyFSI)

Create an AutoBody from the FSI body for use with the fluid solver.
"""
function create_fsi_body(body::FlexibleBodyFSI{T}) where T
    # SDF function that queries the current beam state
    sdf_func = (x, t) -> sdf(body, x, t)

    # No coordinate mapping - deformation is handled by time-varying SDF
    velocity_func = (x, t) -> body_velocity(body, x, t)
    AutoBody(sdf_func; velocity=velocity_func)
end

"""
    FSISimulation

Combined fluid-structure simulation.

# Fields
- `flow`: Flow structure
- `poisson`: Poisson solver
- `body`: FlexibleBodyFSI structure
- `auto_body`: AutoBody created from FSI body
- `t`: Current simulation time
"""
mutable struct FSISimulation{T<:Real, F<:Flow, P<:AbstractPoisson, B<:FlexibleBodyFSI}
    flow::F
    poisson::P
    body::B
    t::T
end

"""
    FSISimulation(dims, domain; beam_params, fsi_params, kwargs...)

Create an FSI simulation with a flexible body.

# Arguments
- `dims`: Grid dimensions (nx, nz)
- `domain`: Domain size (Lx, Lz)

# Keyword Arguments
- `beam_length`: Beam length (default: domain[1]/5)
- `beam_thickness`: Max beam thickness (default: beam_length/10)
- `beam_n`: Number of beam nodes (default: 51)
- `material`: BeamMaterial (default: flexible rubber)
- `x_head`: Head x-position (default: domain[1]/4)
- `z_center`: Centerline z-position (default: domain[2]/2)
- `active_forcing`: Active forcing function f(s, t)
- `ν`: Kinematic viscosity
- `ρ`: Fluid density
- `inletBC`: Inlet boundary condition
- Other standard Simulation parameters
"""
function FSISimulation(dims::Tuple{Int,Int}, domain::Tuple{Real,Real};
                       beam_length::Real=domain[1]/5,
                       beam_thickness::Real=beam_length/10,
                       beam_n::Int=51,
                       material::BeamMaterial=BeamMaterial(),
                       x_head::Real=domain[1]/4,
                       z_center::Real=domain[2]/2,
                       active_forcing::Union{Function, Nothing}=nothing,
                       bc_left::BeamBoundaryCondition=CLAMPED,
                       bc_right::BeamBoundaryCondition=FREE,
                       damping::Real=0.1f0,
                       tension::Real=0f0,
                       ν::Real=0.001f0,
                       ρ::Real=1000f0,
                       inletBC::Tuple=(1f0, 0f0),
                       mem=Array,
                       T::Type=Float32,
                       kwargs...)

    # Check backend/memory compatibility (same guard as Simulation constructor)
    # The @loop macro generates code at compile time based on the backend preference.
    # If backend="SIMD" but GPU arrays are used, scalar indexing will occur.
    if backend == "SIMD" && mem !== Array
        error("Backend mismatch: The @loop backend is set to \"SIMD\" (serial CPU), " *
              "but GPU arrays (mem=$mem) were requested.\n" *
              "This would cause extremely slow scalar indexing on GPU.\n" *
              "Solutions:\n" *
              "  1. Use CPU arrays: mem=Array (default)\n" *
              "  2. Switch to KernelAbstractions backend for GPU support:\n" *
              "     using BioFlows; BioFlows.set_backend(\"KernelAbstractions\")\n" *
              "     Then restart Julia for the change to take effect.")
    end

    # Use single precision by default for GPU efficiency

    # Create beam geometry with fish-like thickness profile
    h_func = fish_thickness_profile(beam_length, beam_thickness)
    geometry = BeamGeometry(beam_length, beam_n;
                            thickness=h_func,
                            width=beam_thickness)  # Assume square cross-section

    # Create beam
    beam = EulerBernoulliBeam(geometry, material;
                              bc_left=bc_left,
                              bc_right=bc_right,
                              damping=damping,
                              tension=tension)

    # Create FSI body
    fsi_body = FlexibleBodyFSI(beam;
                               x_head=x_head,
                               z_center=z_center,
                               h_func=h_func,
                               ρ_fluid=ρ,
                               active_forcing=active_forcing)

    # Create AutoBody from FSI body
    auto_body = create_fsi_body(fsi_body)

    # Create flow (body is passed to measure!, not Flow constructor)
    # Note: Flow takes N as positional and L as keyword argument
    # mem parameter is mapped to f for GPU support
    flow = Flow(dims; L=domain,
                inletBC=inletBC,
                ν=ν,
                ρ=ρ,
                f=mem,
                kwargs...)

    # Create Poisson solver
    poisson = MultiLevelPoisson(flow.p, flow.μ₀, flow.σ)

    # Initial measure
    measure!(flow, auto_body; t=zero(T))
    update!(poisson)

    FSISimulation{T, typeof(flow), typeof(poisson), typeof(fsi_body)}(
        flow, poisson, fsi_body, zero(T)
    )
end

"""
    sim_step!(sim::FSISimulation; kwargs...)

Advance the FSI simulation by one time step.

# Algorithm:
1. Advance fluid with current body shape
2. Compute fluid forces on structure
3. Advance structure with forces
4. Update body shape for next step
"""
function sim_step!(sim::FSISimulation{T}; explicit_fsi::Bool=false) where T
    flow = sim.flow
    poisson = sim.poisson
    body = sim.body

    Δt = flow.Δt[end]
    t_new = sim.t + Δt

    # Advance fluid (using current body shape)
    mom_step!(flow, poisson)

    # FSI coupling: compute loads and advance structure
    fsi_step!(body, flow, sim.t, Δt; explicit=explicit_fsi)

    # Update body for next step
    auto_body = create_fsi_body(body)
    measure!(flow, auto_body; t=t_new)
    update!(poisson)

    sim.t = t_new

    return sim
end

"""
    sim_time(sim::FSISimulation)

Get the current simulation time.
"""
sim_time(sim::FSISimulation) = sim.t

"""
    get_beam(sim::FSISimulation)

Get the beam structure from the simulation.
"""
get_beam(sim::FSISimulation) = sim.body.beam

"""
    get_displacement(sim::FSISimulation)

Get the current beam displacement field.
"""
get_displacement(sim::FSISimulation) = get_displacement(sim.body.beam)

"""
    get_fluid_load(sim::FSISimulation)

Get the current fluid load on the beam.
"""
get_fluid_load(sim::FSISimulation) = sim.body.beam.q

# =============================================================================
# ACTIVE FORCING FUNCTIONS (MUSCLE ACTIVATION)
# =============================================================================

"""
    traveling_wave_forcing(; amplitude, frequency, wavelength, envelope=:carangiform)

Create a traveling wave active forcing function for swimming.

# Arguments
- `amplitude`: Maximum forcing amplitude (N/m)
- `frequency`: Wave frequency (Hz)
- `wavelength`: Wave length relative to body length
- `envelope`: Amplitude envelope (:carangiform, :anguilliform, :uniform)
- `L`: Body length (for envelope calculation)
"""
function traveling_wave_forcing(; amplitude::Real, frequency::Real,
                                  wavelength::Real=1f0, envelope::Symbol=:carangiform,
                                  L::Real=1f0)
    ω = 2π * frequency
    k = 2π / (wavelength * L)

    # Envelope function
    env = if envelope == :carangiform
        s -> (s/L)^2
    elseif envelope == :anguilliform
        s -> 0.3f0 + 0.7f0 * s/L
    elseif envelope == :subcarangiform
        s -> (s/L)^1.5f0
    else  # uniform
        s -> 1f0
    end

    return (s, t) -> amplitude * env(s) * sin(k*s - ω*t)
end

"""
    heave_pitch_forcing(; heave_amp, pitch_amp, frequency, heave_phase=0, pitch_phase=π/2, L=1.0)

Create a heave + pitch forcing at the leading edge.

# Arguments
- `heave_amp`: Heave forcing amplitude (N/m)
- `pitch_amp`: Pitch moment amplitude (N·m/m)
- `frequency`: Oscillation frequency (Hz)
- `heave_phase`: Heave phase offset (default: 0)
- `pitch_phase`: Pitch phase offset (default: π/2)
- `L`: Body length for normalization
"""
function heave_pitch_forcing(; heave_amp::Real=0f0, pitch_amp::Real=0f0,
                               frequency::Real=1f0,
                               heave_phase::Real=0f0, pitch_phase::Real=π/2,
                               L::Real=1f0)
    ω = 2π * frequency

    return function(s, t)
        # Heave forcing concentrated at head
        f_heave = heave_amp * exp(-(s/L)^2 / 0.01f0) * sin(ω*t + heave_phase)

        # Pitch forcing (moment applied at head, creates distributed force)
        f_pitch = pitch_amp * (s/L) * exp(-(s/L)^2 / 0.1f0) * sin(ω*t + pitch_phase)

        return f_heave + f_pitch
    end
end
