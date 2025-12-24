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

BeamMaterial(; ρ=1100.0, E=1e6, ν_poisson=0.45) = BeamMaterial(ρ, E, ν_poisson)

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

function BeamGeometry(L::Real, n::Int; thickness=0.01, width=0.05)
    h_func = thickness isa Function ? thickness : (s -> thickness)
    b_func = width isa Function ? width : (s -> width)
    BeamGeometry(Float64(L), n, h_func, b_func)
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
                            damping::Real=0.0,
                            tension::Real=0.0,
                            β::Real=0.25,
                            γ::Real=0.5,
                            Δt::Real=1e-4) where T

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
        M_e = (m_e * h / 420) * [
            156     22h     54     -13h  ;
            22h     4h2     13h    -3h2  ;
            54      13h     156    -22h  ;
            -13h    -3h2    -22h   4h2
        ]

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

        # Element damping matrix (proportional to mass for simplicity)
        C_e = damping * M_e / m_e  # Rayleigh damping approximation

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

    penalty = maximum(abs.(K)) * 1e8

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
