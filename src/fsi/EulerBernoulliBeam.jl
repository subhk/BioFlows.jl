# =============================================================================
# EULER-BERNOULLI BEAM SOLVER FOR FLEXIBLE BODIES
# =============================================================================
# Full Finite Element Implementation with Hermite Cubic Shape Functions
#
# Solves the Euler-Bernoulli beam equation:
#   ρₛA ∂²w/∂t² + c ∂w/∂t + EI ∂⁴w/∂x⁴ - T ∂²w/∂x² = q(x,t) + f_active(x,t)
#
# Each node has 2 DOFs: displacement w and rotation θ = dw/dx
# Total system size: 2n DOFs for n nodes
#
# Time integration: Newmark-beta method (unconditionally stable)
# =============================================================================

using LinearAlgebra
using SparseArrays
using JLD2

"""
    BeamBoundaryCondition

Boundary condition types for the beam.
"""
@enum BeamBoundaryCondition begin
    CLAMPED    # w = 0, θ = 0 (fixed position and slope)
    FREE       # M = 0, V = 0 (no moment, no shear)
    PINNED     # w = 0, M = 0 (fixed position, free rotation)
    PRESCRIBED # w = w_prescribed(t), θ = θ_prescribed(t)
end

"""
    BeamMaterial

Material properties for the beam.
"""
struct BeamMaterial{T<:Real}
    ρ::T          # Density (kg/m³)
    E::T          # Young's modulus (Pa)
    ν_poisson::T  # Poisson's ratio
end

BeamMaterial(; ρ=1100.0f0, E=1f6, ν_poisson=0.45f0) = BeamMaterial(ρ, E, ν_poisson)

"""
    BeamGeometry

Geometric properties of the beam.
"""
struct BeamGeometry{T<:Real, F1<:Function, F2<:Function}
    L::T              # Total length (m)
    n::Int            # Number of nodes
    thickness::F1     # h(s): thickness at position s
    width::F2         # b(s): width at position s
end

function BeamGeometry(L::Real, n::Int; thickness=0.01f0, width=0.05f0, T::Type=Float32)
    h_func = thickness isa Function ? thickness : (s -> T(thickness))
    b_func = width isa Function ? width : (s -> T(width))
    BeamGeometry(T(L), n, h_func, b_func)
end

# NACA-like fish profile
function fish_thickness_profile(L::Real, h_max::Real)
    s -> h_max * 4 * clamp(s/L, 0, 1) * (1 - clamp(s/L, 0, 1))
end

"""
    EulerBernoulliBeam{T}

Euler-Bernoulli beam with full FEM discretization.
Uses 2 DOFs per node: displacement w and rotation θ.
"""
mutable struct EulerBernoulliBeam{T<:Real}
    # Geometry and material
    geometry::BeamGeometry{T}
    material::BeamMaterial{T}

    # Boundary conditions
    bc_left::BeamBoundaryCondition
    bc_right::BeamBoundaryCondition

    # Physical parameters
    damping::T
    tension::T

    # Discretization
    s::Vector{T}      # Node positions
    Δs::T             # Element length
    n_nodes::Int      # Number of nodes
    n_dof::Int        # Total DOFs (2 * n_nodes)

    # State variables (size n_dof = 2*n_nodes)
    # Ordering: [w₁, θ₁, w₂, θ₂, ..., wₙ, θₙ]
    u::Vector{T}      # Displacement/rotation vector
    u_dot::Vector{T}  # Velocity vector
    u_ddot::Vector{T} # Acceleration vector

    # External loads (size n_nodes, applied as distributed load on w DOFs)
    q::Vector{T}          # Fluid pressure load (N/m)
    f_active::Vector{T}   # Active forcing (N/m)

    # Cross-sectional properties at each node
    A::Vector{T}      # Cross-sectional area
    I::Vector{T}      # Second moment of area
    m::Vector{T}      # Mass per unit length
    EI::Vector{T}     # Bending stiffness

    # System matrices (sparse, size n_dof × n_dof)
    M_mat::SparseMatrixCSC{T, Int}
    C_mat::SparseMatrixCSC{T, Int}
    K_mat::SparseMatrixCSC{T, Int}

    # Constrained DOF mask
    constrained::Vector{Bool}

    # Newmark parameters
    β::T
    γ::T
    Δt::T
end

"""
    EulerBernoulliBeam(geometry, material; kwargs...)

Create an Euler-Bernoulli beam with full FEM discretization.
"""
function EulerBernoulliBeam(geometry::BeamGeometry{T}, material::BeamMaterial{T};
                            bc_left::BeamBoundaryCondition=CLAMPED,
                            bc_right::BeamBoundaryCondition=FREE,
                            damping::Real=0f0,
                            tension::Real=0f0,
                            β::Real=0.25f0,
                            γ::Real=0.5f0,
                            Δt::Real=1f-4) where T

    n = geometry.n
    L = geometry.L
    Δs = L / (n - 1)
    n_dof = 2 * n

    # Node positions
    s = collect(range(0, L, length=n))

    # Initialize state vectors
    u = zeros(T, n_dof)
    u_dot = zeros(T, n_dof)
    u_ddot = zeros(T, n_dof)

    # External loads
    q = zeros(T, n)
    f_active = zeros(T, n)

    # Compute cross-sectional properties
    A_vec = zeros(T, n)
    I_vec = zeros(T, n)
    m_vec = zeros(T, n)
    EI_vec = zeros(T, n)

    for i in 1:n
        h = geometry.thickness(s[i])
        b = geometry.width(s[i])
        A_vec[i] = h * b
        I_vec[i] = b * h^3 / 12
        m_vec[i] = material.ρ * A_vec[i]
        EI_vec[i] = material.E * I_vec[i]
    end

    # Build system matrices
    constrained = falses(n_dof)
    M_mat, C_mat, K_mat = build_system_matrices(
        m_vec, EI_vec, T(damping), T(tension), Δs, n,
        bc_left, bc_right, constrained
    )

    EulerBernoulliBeam{T}(
        geometry, material,
        bc_left, bc_right,
        T(damping), T(tension),
        s, Δs, n, n_dof,
        u, u_dot, u_ddot,
        q, f_active,
        A_vec, I_vec, m_vec, EI_vec,
        M_mat, C_mat, K_mat,
        constrained,
        T(β), T(γ), T(Δt)
    )
end

"""
    build_system_matrices(m, EI, damping, tension, h, n, bc_left, bc_right, constrained)

Build mass, damping, and stiffness matrices using Hermite FEM.
"""
function build_system_matrices(m::Vector{T}, EI::Vector{T}, damping::T, tension::T,
                                h::T, n::Int, bc_left::BeamBoundaryCondition,
                                bc_right::BeamBoundaryCondition,
                                constrained::AbstractVector{Bool}) where T

    n_dof = 2 * n
    h2 = h^2
    h3 = h^3

    # Initialize matrices
    M = zeros(T, n_dof, n_dof)
    C = zeros(T, n_dof, n_dof)
    K = zeros(T, n_dof, n_dof)

    # Assemble element matrices
    for e in 1:n-1
        # Node indices
        i1, i2 = e, e + 1

        # Average properties for element
        m_e = (m[i1] + m[i2]) / 2
        EI_e = (EI[i1] + EI[i2]) / 2

        # DOF indices: [w₁, θ₁, w₂, θ₂]
        dofs = [2*i1-1, 2*i1, 2*i2-1, 2*i2]

        # Element mass matrix (consistent mass, Hermite)
        # M_e = (ρA*L/420) * [156,   22L,   54,  -13L;
        #                      22L,  4L²,  13L,  -3L²;
        #                      54,   13L,  156,  -22L;
        #                     -13L, -3L², -22L,  4L²]
        mass_template = [
            156     22h     54     -13h  ;
            22h     4h2     13h    -3h2  ;
            54      13h     156    -22h  ;
            -13h    -3h2    -22h   4h2
        ]
        M_e = (m_e * h / 420) * mass_template

        # Element stiffness matrix (Hermite beam bending)
        # K_e = (EI/L³) * [12,   6L,  -12,   6L;
        #                   6L,  4L²,  -6L,  2L²;
        #                  -12, -6L,   12,  -6L;
        #                   6L,  2L², -6L,  4L²]
        K_e = (EI_e / h3) * [
            12    6h    -12   6h  ;
            6h    4h2   -6h   2h2 ;
            -12   -6h   12    -6h ;
            6h    2h2   -6h   4h2
        ]

        # Add tension stiffness (geometric stiffness)
        # K_g = (T/L) * [1, 0, -1, 0; 0, 0, 0, 0; -1, 0, 1, 0; 0, 0, 0, 0]
        # But for beam with rotations, use proper geometric stiffness:
        # K_g = (T/(30L)) * [36,   3L,  -36,   3L;
        #                     3L,  4L², -3L,  -L²;
        #                    -36, -3L,   36,  -3L;
        #                     3L,  -L², -3L,  4L²]
        if abs(tension) > 1e-12
            K_g = (tension / (30h)) * [
                36    3h    -36   3h  ;
                3h    4h2   -3h   -h2 ;
                -36   -3h   36    -3h ;
                3h    -h2   -3h   4h2
            ]
            K_e += K_g
        end

        # Element damping matrix for c * ∂w/∂t (c is per-length damping)
        C_e = (damping * h / 420) * mass_template

        # Assemble into global matrices
        for ii in 1:4
            for jj in 1:4
                M[dofs[ii], dofs[jj]] += M_e[ii, jj]
                C[dofs[ii], dofs[jj]] += C_e[ii, jj]
                K[dofs[ii], dofs[jj]] += K_e[ii, jj]
            end
        end
    end

    # Apply boundary conditions
    apply_boundary_conditions!(M, C, K, n, bc_left, bc_right, constrained)

    return sparse(M), sparse(C), sparse(K)
end

"""
    apply_boundary_conditions!(M, C, K, n, bc_left, bc_right, constrained)

Apply boundary conditions by modifying system matrices.
Uses penalty method for constrained DOFs.
"""
function apply_boundary_conditions!(M::Matrix{T}, C::Matrix{T}, K::Matrix{T},
                                     n::Int, bc_left::BeamBoundaryCondition,
                                     bc_right::BeamBoundaryCondition,
                                     constrained::AbstractVector{Bool}) where T

    maxK = maximum(abs.(K))
    penalty = max(maxK, one(T)) * T(1e8)

    # Left boundary
    if bc_left == CLAMPED || bc_left == PRESCRIBED
        # Constrain both w and θ
        constrained[1] = true  # w₁
        constrained[2] = true  # θ₁
    elseif bc_left == PINNED
        # Constrain only w
        constrained[1] = true  # w₁
    end
    # FREE: no constraints

    # Right boundary
    n_dof = 2 * n
    if bc_right == CLAMPED || bc_right == PRESCRIBED
        constrained[n_dof-1] = true  # wₙ
        constrained[n_dof] = true    # θₙ
    elseif bc_right == PINNED
        constrained[n_dof-1] = true  # wₙ
    end
    # FREE: no constraints

    # Apply penalty to constrained DOFs
    for i in 1:n_dof
        if constrained[i]
            # Zero out row and column
            M[i, :] .= zero(T)
            M[:, i] .= zero(T)
            C[i, :] .= zero(T)
            C[:, i] .= zero(T)
            K[i, :] .= zero(T)
            K[:, i] .= zero(T)

            # Set diagonal
            M[i, i] = one(T)
            K[i, i] = penalty
        end
    end
end

"""
    step!(beam::EulerBernoulliBeam, Δt)

Advance the beam solution by one time step using Newmark-beta method.
"""
function step!(beam::EulerBernoulliBeam{T}, Δt::T=beam.Δt) where T
    β, γ = beam.β, beam.γ
    n_dof = beam.n_dof
    n = beam.n_nodes

    # Newmark coefficients
    a0 = one(T) / (β * Δt^2)
    a1 = γ / (β * Δt)
    a2 = one(T) / (β * Δt)
    a3 = one(T) / (2β) - one(T)
    a4 = γ / β - one(T)
    a5 = Δt * (γ / (2β) - one(T))

    # Effective stiffness
    K_eff = beam.K_mat + a1 * beam.C_mat + a0 * beam.M_mat

    # Build force vector from distributed loads
    F = zeros(T, n_dof)
    h = beam.Δs

    # Consistent load vector for distributed load q
    # For each element, the consistent nodal forces from uniform load q are:
    # F_e = (q*L/2) * [1, L/6, 1, -L/6] for [w₁, θ₁, w₂, θ₂]
    for e in 1:n-1
        i1, i2 = e, e + 1
        q_e = (beam.q[i1] + beam.f_active[i1] + beam.q[i2] + beam.f_active[i2]) / 2
        dofs = [2*i1-1, 2*i1, 2*i2-1, 2*i2]

        # Consistent load vector for uniform distributed load
        F[dofs[1]] += q_e * h / 2
        F[dofs[2]] += q_e * h^2 / 12
        F[dofs[3]] += q_e * h / 2
        F[dofs[4]] += -q_e * h^2 / 12
    end

    # Zero force at constrained DOFs
    for i in 1:n_dof
        if beam.constrained[i]
            F[i] = zero(T)
        end
    end

    # Effective force
    F_eff = F + beam.M_mat * (a0 * beam.u + a2 * beam.u_dot + a3 * beam.u_ddot) +
                beam.C_mat * (a1 * beam.u + a4 * beam.u_dot + a5 * beam.u_ddot)

    # Solve for new displacement
    u_new = K_eff \ F_eff

    # Update acceleration and velocity
    u_ddot_new = a0 * (u_new - beam.u) - a2 * beam.u_dot - a3 * beam.u_ddot
    u_dot_new = beam.u_dot + Δt * ((one(T) - γ) * beam.u_ddot + γ * u_ddot_new)

    # Store new values
    beam.u .= u_new
    beam.u_dot .= u_dot_new
    beam.u_ddot .= u_ddot_new

    return beam
end

"""
    reset!(beam::EulerBernoulliBeam)

Reset beam to initial state (zero displacement/velocity).
"""
function reset!(beam::EulerBernoulliBeam{T}) where T
    fill!(beam.u, zero(T))
    fill!(beam.u_dot, zero(T))
    fill!(beam.u_ddot, zero(T))
    fill!(beam.q, zero(T))
    fill!(beam.f_active, zero(T))
    return beam
end

# =============================================================================
# ACCESSOR FUNCTIONS - Extract w (displacement) from full state vector
# =============================================================================

"""
    get_displacement(beam) -> Vector

Get the displacement field w(s) at each node.
"""
function get_displacement(beam::EulerBernoulliBeam)
    return @view beam.u[1:2:end]  # w values are at odd indices
end

# For compatibility, also expose as beam.w
function Base.getproperty(beam::EulerBernoulliBeam, sym::Symbol)
    if sym === :w
        return @view getfield(beam, :u)[1:2:end]
    elseif sym === :w_dot
        return @view getfield(beam, :u_dot)[1:2:end]
    elseif sym === :w_ddot
        return @view getfield(beam, :u_ddot)[1:2:end]
    elseif sym === :θ
        return @view getfield(beam, :u)[2:2:end]
    elseif sym === :θ_dot
        return @view getfield(beam, :u_dot)[2:2:end]
    else
        return getfield(beam, sym)
    end
end

"""
    get_velocity(beam) -> Vector

Get the velocity field ∂w/∂t at each node.
"""
get_velocity(beam::EulerBernoulliBeam) = @view beam.u_dot[1:2:end]

"""
    get_rotation(beam) -> Vector

Get the rotation field θ = ∂w/∂x at each node.
"""
get_rotation(beam::EulerBernoulliBeam) = @view beam.u[2:2:end]

"""
    get_curvature(beam) -> Vector

Compute the curvature κ = ∂θ/∂x ≈ ∂²w/∂x² at each node.
"""
function get_curvature(beam::EulerBernoulliBeam{T}) where T
    n = beam.n_nodes
    θ = get_rotation(beam)
    κ = zeros(T, n)
    Δs = beam.Δs

    # Central difference for interior
    for i in 2:n-1
        κ[i] = (θ[i+1] - θ[i-1]) / (2 * Δs)
    end

    # One-sided for boundaries
    if n >= 2
        κ[1] = (θ[2] - θ[1]) / Δs
        κ[n] = (θ[n] - θ[n-1]) / Δs
    end

    return κ
end

"""
    get_bending_moment(beam) -> Vector

Compute the bending moment M = EI * κ at each node.
"""
function get_bending_moment(beam::EulerBernoulliBeam{T}) where T
    κ = get_curvature(beam)
    return beam.EI .* κ
end

# =============================================================================
# LOAD APPLICATION
# =============================================================================

"""
    set_fluid_load!(beam, q_func, t)

Set the fluid load on the beam from a function q(s, t).
"""
function set_fluid_load!(beam::EulerBernoulliBeam{T}, q_func::Function, t::Real) where T
    for i in 1:beam.n_nodes
        beam.q[i] = T(q_func(beam.s[i], t))
    end
end

function set_fluid_load!(beam::EulerBernoulliBeam{T}, q_values::AbstractVector) where T
    @assert length(q_values) == beam.n_nodes
    beam.q .= q_values
end

"""
    set_active_forcing!(beam, f_func, t)

Set the active forcing (muscle activation) from a function f(s, t).
"""
function set_active_forcing!(beam::EulerBernoulliBeam{T}, f_func::Function, t::Real) where T
    for i in 1:beam.n_nodes
        beam.f_active[i] = T(f_func(beam.s[i], t))
    end
end

function set_active_forcing!(beam::EulerBernoulliBeam{T}, f_values::AbstractVector) where T
    @assert length(f_values) == beam.n_nodes
    beam.f_active .= f_values
end

"""
    get_fluid_load(beam) -> Vector

Get the current fluid load at each node.
"""
get_fluid_load(beam::EulerBernoulliBeam) = beam.q

# =============================================================================
# ENERGY FUNCTIONS
# =============================================================================

"""
    kinetic_energy(beam) -> Real

Compute the kinetic energy of the beam: KE = ½ u̇ᵀ M u̇
"""
function kinetic_energy(beam::EulerBernoulliBeam{T}) where T
    return T(0.5) * dot(beam.u_dot, beam.M_mat * beam.u_dot)
end

"""
    potential_energy(beam) -> Real

Compute the potential (strain) energy of the beam: PE = ½ uᵀ K u
"""
function potential_energy(beam::EulerBernoulliBeam{T}) where T
    return T(0.5) * dot(beam.u, beam.K_mat * beam.u)
end

"""
    total_energy(beam) -> Real

Compute the total mechanical energy: E = KE + PE
"""
total_energy(beam::EulerBernoulliBeam) = kinetic_energy(beam) + potential_energy(beam)

# =============================================================================
# Apply stiffness BC - dummy function for compatibility
# =============================================================================
function apply_stiffness_bc!(K, n, bc_left, bc_right)
    # Not used in new implementation
end

# =============================================================================
# BEAM STATE WRITER - Save beam positions to JLD2 files
# =============================================================================

"""
    BeamStateWriter(filename::AbstractString; interval::Real=0.01, overwrite::Bool=true)

Writer for saving flexible body (beam) state to JLD2 files at configurable intervals.

Each beam gets its own file containing time-series data of:
- Displacement field w(s,t)
- Rotation field θ(s,t)
- Velocity field ẇ(s,t)
- Arc-length coordinates s
- Curvature κ(s,t)
- Bending moment M(s,t)
- Kinetic and potential energy

# Arguments
- `filename`: Output JLD2 file path (e.g., "beam_1.jld2")
- `interval`: Time interval between saves (default: 0.01)
- `overwrite`: If true, overwrite existing file (default: true)

# Example
```julia
# Create writers for multiple beams
beam1_writer = BeamStateWriter("flag_1.jld2"; interval=0.01)
beam2_writer = BeamStateWriter("flag_2.jld2"; interval=0.01)

# In simulation loop
for step in 1:1000
    t = step * dt
    step!(beam1, dt)
    step!(beam2, dt)

    file_save!(beam1_writer, beam1, t)
    file_save!(beam2_writer, beam2, t)
end

# Close writers when done
close!(beam1_writer)
close!(beam2_writer)
```

# Reading the Output
```julia
using JLD2

jldopen("flag_1.jld2", "r") do file
    # Get metadata
    s = file["coordinates/s"]
    n_snapshots = file["metadata/n_snapshots"]

    # Read specific snapshot
    t = file["snapshots/1/time"]
    w = file["snapshots/1/displacement"]
    θ = file["snapshots/1/rotation"]
    κ = file["snapshots/1/curvature"]
end
```
"""
mutable struct BeamStateWriter
    filename::String
    interval::Float64
    next_time::Float64
    samples::Int
    initialized::Bool

    # History storage
    time_history::Vector{Float64}
    displacement_history::Vector{Vector{Float64}}
    rotation_history::Vector{Vector{Float64}}
    velocity_history::Vector{Vector{Float64}}
    curvature_history::Vector{Vector{Float64}}
    moment_history::Vector{Vector{Float64}}
    kinetic_energy_history::Vector{Float64}
    potential_energy_history::Vector{Float64}

    function BeamStateWriter(filename::AbstractString="beam_state.jld2";
                             interval::Real=0.01,
                             overwrite::Bool=true)
        interval > 0 || throw(ArgumentError("interval must be positive"))

        # Handle file creation/overwrite
        if overwrite && isfile(filename)
            rm(filename)
        end

        return new(
            String(filename),
            float(interval),
            0.0,  # Start saving from t=0
            0,
            false,
            Float64[],
            Vector{Float64}[],
            Vector{Float64}[],
            Vector{Float64}[],
            Vector{Float64}[],
            Vector{Float64}[],
            Float64[],
            Float64[]
        )
    end
end

"""
    file_save!(writer::BeamStateWriter, beam::EulerBernoulliBeam, t::Real)

Check the time and, if the configured interval has elapsed, save the current
beam state to the writer's storage.

Returns the writer for chaining.
"""
function file_save!(writer::BeamStateWriter, beam::EulerBernoulliBeam, t::Real)
    if t + eps(writer.interval) < writer.next_time
        return writer
    end

    while t + eps(writer.interval) >= writer.next_time
        _record_beam_state!(writer, beam, t)
        writer.samples += 1
        writer.next_time += writer.interval
    end

    return writer
end

"""
Internal function to record beam state to writer's storage.
"""
function _record_beam_state!(writer::BeamStateWriter, beam::EulerBernoulliBeam, t::Real)
    # Record time
    push!(writer.time_history, t)

    # Record displacement and rotation (copy to avoid aliasing)
    push!(writer.displacement_history, copy(Vector(beam.w)))
    push!(writer.rotation_history, copy(Vector(beam.θ)))
    push!(writer.velocity_history, copy(Vector(beam.w_dot)))

    # Compute and record curvature and moment
    κ = get_curvature(beam)
    M = get_bending_moment(beam)
    push!(writer.curvature_history, copy(κ))
    push!(writer.moment_history, copy(M))

    # Record energies
    push!(writer.kinetic_energy_history, kinetic_energy(beam))
    push!(writer.potential_energy_history, potential_energy(beam))
end

"""
    close!(writer::BeamStateWriter, beam::EulerBernoulliBeam)

Finalize and write all accumulated data to the JLD2 file.
Must be called to save data to disk.
"""
function close!(writer::BeamStateWriter, beam::EulerBernoulliBeam)
    writer.samples == 0 && return writer

    jldopen(writer.filename, "w") do file
        # Metadata
        file["metadata/n_snapshots"] = writer.samples
        file["metadata/interval"] = writer.interval
        file["metadata/n_nodes"] = beam.n_nodes
        file["metadata/length"] = beam.geometry.L

        # Coordinates (constant)
        file["coordinates/s"] = collect(beam.s)

        # Material properties
        file["material/density"] = beam.material.ρ
        file["material/youngs_modulus"] = beam.material.E

        # Time series as arrays (more efficient for analysis)
        file["time"] = writer.time_history
        file["kinetic_energy"] = writer.kinetic_energy_history
        file["potential_energy"] = writer.potential_energy_history

        # Snapshots
        for i in 1:writer.samples
            group = "snapshots/$i"
            file["$group/time"] = writer.time_history[i]
            file["$group/displacement"] = writer.displacement_history[i]
            file["$group/rotation"] = writer.rotation_history[i]
            file["$group/velocity"] = writer.velocity_history[i]
            file["$group/curvature"] = writer.curvature_history[i]
            file["$group/moment"] = writer.moment_history[i]
        end

        # Also store as matrices for easy analysis
        n = beam.n_nodes
        w_matrix = zeros(n, writer.samples)
        θ_matrix = zeros(n, writer.samples)
        v_matrix = zeros(n, writer.samples)
        κ_matrix = zeros(n, writer.samples)

        for i in 1:writer.samples
            w_matrix[:, i] = writer.displacement_history[i]
            θ_matrix[:, i] = writer.rotation_history[i]
            v_matrix[:, i] = writer.velocity_history[i]
            κ_matrix[:, i] = writer.curvature_history[i]
        end

        file["fields/displacement"] = w_matrix
        file["fields/rotation"] = θ_matrix
        file["fields/velocity"] = v_matrix
        file["fields/curvature"] = κ_matrix
    end

    return writer
end

"""
    reset!(writer::BeamStateWriter)

Reset the writer to start fresh. Does not delete the existing file.
"""
function reset!(writer::BeamStateWriter)
    writer.samples = 0
    writer.next_time = 0.0
    writer.initialized = false
    empty!(writer.time_history)
    empty!(writer.displacement_history)
    empty!(writer.rotation_history)
    empty!(writer.velocity_history)
    empty!(writer.curvature_history)
    empty!(writer.moment_history)
    empty!(writer.kinetic_energy_history)
    empty!(writer.potential_energy_history)
    return writer
end

"""
    BeamStateWriterGroup(prefix::AbstractString, n_beams::Int; kwargs...)

Create multiple BeamStateWriters for a group of beams.
Files are named `{prefix}_1.jld2`, `{prefix}_2.jld2`, etc.

# Example
```julia
# Create writers for 5 flags
writers = BeamStateWriterGroup("flag", 5; interval=0.01)

# In simulation loop
for step in 1:1000
    t = step * dt
    for (i, beam) in enumerate(beams)
        step!(beam, dt)
        file_save!(writers[i], beam, t)
    end
end

# Close all writers
close!(writers, beams)
```
"""
struct BeamStateWriterGroup
    writers::Vector{BeamStateWriter}

    function BeamStateWriterGroup(prefix::AbstractString, n_beams::Int;
                                   interval::Real=0.01,
                                   overwrite::Bool=true)
        writers = [BeamStateWriter("$(prefix)_$i.jld2";
                                    interval=interval,
                                    overwrite=overwrite)
                   for i in 1:n_beams]
        return new(writers)
    end
end

# Array-like access
Base.getindex(g::BeamStateWriterGroup, i::Int) = g.writers[i]
Base.length(g::BeamStateWriterGroup) = length(g.writers)
Base.iterate(g::BeamStateWriterGroup) = iterate(g.writers)
Base.iterate(g::BeamStateWriterGroup, state) = iterate(g.writers, state)

"""
    file_save!(group::BeamStateWriterGroup, beams::Vector{<:EulerBernoulliBeam}, t::Real)

Save state for all beams in a group.
"""
function file_save!(group::BeamStateWriterGroup, beams::AbstractVector{<:EulerBernoulliBeam}, t::Real)
    for (writer, beam) in zip(group.writers, beams)
        file_save!(writer, beam, t)
    end
    return group
end

"""
    close!(group::BeamStateWriterGroup, beams::Vector{<:EulerBernoulliBeam})

Close all writers in a group.
"""
function close!(group::BeamStateWriterGroup, beams::AbstractVector{<:EulerBernoulliBeam})
    for (writer, beam) in zip(group.writers, beams)
        close!(writer, beam)
    end
    return group
end
