# =============================================================================
# EULER-BERNOULLI BEAM SOLVER FOR FLEXIBLE BODIES
# =============================================================================
# Solves the Euler-Bernoulli beam equation for fluid-structure interaction:
#
#   ρₛA ∂²w/∂t² + c ∂w/∂t + EI ∂⁴w/∂x⁴ - T ∂²w/∂x² = q(x,t) + f_active(x,t)
#
# where:
#   ρₛ = beam material density (kg/m³)
#   A  = cross-sectional area (m²)
#   c  = damping coefficient (kg/(m·s))
#   E  = Young's modulus (Pa)
#   I  = second moment of area (m⁴)
#   T  = axial tension (N)
#   w  = transverse displacement (m)
#   q  = distributed fluid load (N/m)
#   f_active = active forcing (muscle activation) (N/m)
#
# Boundary conditions:
#   - Clamped: w = 0, w' = 0
#   - Free: w'' = 0, w''' = 0
#   - Pinned: w = 0, w'' = 0
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
    CLAMPED    # w = 0, w' = 0 (fixed position and slope)
    FREE       # w'' = 0, w''' = 0 (no moment, no shear)
    PINNED     # w = 0, w'' = 0 (fixed position, free rotation)
    PRESCRIBED # w = w_prescribed(t) (time-varying position)
end

"""
    BeamMaterial

Material properties for the beam.

# Fields
- `ρ`: Material density (kg/m³)
- `E`: Young's modulus (Pa)
- `ν_poisson`: Poisson's ratio (dimensionless)
"""
struct BeamMaterial{T<:Real}
    ρ::T          # Density (kg/m³)
    E::T          # Young's modulus (Pa)
    ν_poisson::T  # Poisson's ratio
end

# Default material: flexible silicone rubber
BeamMaterial(; ρ=1100.0, E=1e6, ν_poisson=0.45) = BeamMaterial(ρ, E, ν_poisson)

"""
    BeamGeometry

Geometric properties of the beam (can vary along length).

# Fields
- `L`: Total beam length (m)
- `n`: Number of discretization points
- `thickness`: Thickness function h(s) (m)
- `width`: Width function b(s) (m)
"""
struct BeamGeometry{T<:Real, F1<:Function, F2<:Function}
    L::T              # Total length (m)
    n::Int            # Number of points
    thickness::F1     # h(s): thickness at position s
    width::F2         # b(s): width at position s
end

# Default: uniform rectangular cross-section
function BeamGeometry(L::Real, n::Int; thickness=0.01, width=0.05)
    h_func = thickness isa Function ? thickness : (s -> thickness)
    b_func = width isa Function ? width : (s -> width)
    BeamGeometry(Float64(L), n, h_func, b_func)
end

# NACA-like fish profile: h(s) = h_max * 4 * (s/L) * (1 - s/L)
function fish_thickness_profile(L::Real, h_max::Real)
    s -> h_max * 4 * clamp(s/L, 0, 1) * (1 - clamp(s/L, 0, 1))
end

"""
    EulerBernoulliBeam{T}

Euler-Bernoulli beam structure for FSI simulation.

# Fields
- `geometry`: Beam geometry (length, discretization, cross-section)
- `material`: Material properties (density, Young's modulus)
- `bc_left`: Left boundary condition
- `bc_right`: Right boundary condition
- `damping`: Damping coefficient c (kg/(m·s))
- `tension`: Axial tension T (N)

# State variables (at discretization points)
- `w`: Transverse displacement (m)
- `w_dot`: Velocity (m/s)
- `w_ddot`: Acceleration (m/s²)
- `q`: Fluid load (N/m)
- `f_active`: Active forcing (N/m)

# Precomputed matrices
- `M`: Mass matrix
- `C`: Damping matrix
- `K`: Stiffness matrix (bending + tension)
"""
mutable struct EulerBernoulliBeam{T<:Real}
    # Geometry and material
    geometry::BeamGeometry{T}
    material::BeamMaterial{T}

    # Boundary conditions
    bc_left::BeamBoundaryCondition
    bc_right::BeamBoundaryCondition

    # Physical parameters
    damping::T        # Damping coefficient c
    tension::T        # Axial tension T

    # Discretization
    s::Vector{T}      # Arc length coordinates
    Δs::T             # Grid spacing

    # State variables
    w::Vector{T}      # Displacement
    w_dot::Vector{T}  # Velocity
    w_ddot::Vector{T} # Acceleration

    # External loads
    q::Vector{T}      # Fluid pressure load
    f_active::Vector{T}  # Active forcing (muscle)

    # Precomputed properties at each point
    A::Vector{T}      # Cross-sectional area
    I::Vector{T}      # Second moment of area
    m::Vector{T}      # Mass per unit length (ρA)
    EI::Vector{T}     # Bending stiffness

    # System matrices (sparse)
    M_mat::SparseMatrixCSC{T, Int}  # Mass matrix
    C_mat::SparseMatrixCSC{T, Int}  # Damping matrix
    K_mat::SparseMatrixCSC{T, Int}  # Stiffness matrix

    # Newmark-beta parameters
    β::T
    γ::T

    # Time step
    Δt::T
end

"""
    EulerBernoulliBeam(geometry, material; kwargs...)

Create an Euler-Bernoulli beam for FSI simulation.

# Arguments
- `geometry`: BeamGeometry specifying length and cross-section
- `material`: BeamMaterial specifying density and stiffness

# Keyword Arguments
- `bc_left`: Left boundary condition (default: CLAMPED)
- `bc_right`: Right boundary condition (default: FREE)
- `damping`: Damping coefficient (default: 0.0)
- `tension`: Axial tension (default: 0.0)
- `β`: Newmark-beta parameter (default: 0.25, unconditionally stable)
- `γ`: Newmark-gamma parameter (default: 0.5, no numerical damping)
- `Δt`: Time step (default: 1e-4)
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

    # Arc length coordinates
    s = collect(range(0, L, length=n))

    # Initialize state
    w = zeros(T, n)
    w_dot = zeros(T, n)
    w_ddot = zeros(T, n)
    q = zeros(T, n)
    f_active = zeros(T, n)

    # Compute cross-sectional properties at each point
    A_vec = zeros(T, n)
    I_vec = zeros(T, n)
    m_vec = zeros(T, n)
    EI_vec = zeros(T, n)

    for i in 1:n
        h = geometry.thickness(s[i])
        b = geometry.width(s[i])
        A_vec[i] = h * b                    # Rectangular cross-section area
        I_vec[i] = b * h^3 / 12             # Second moment of area
        m_vec[i] = material.ρ * A_vec[i]    # Mass per unit length
        EI_vec[i] = material.E * I_vec[i]   # Bending stiffness
    end

    # Build system matrices
    M_mat = build_mass_matrix(m_vec, Δs, n, bc_left, bc_right)
    C_mat = build_damping_matrix(T(damping), Δs, n, bc_left, bc_right)
    K_mat = build_stiffness_matrix(EI_vec, T(tension), Δs, n, bc_left, bc_right)

    EulerBernoulliBeam{T}(
        geometry, material,
        bc_left, bc_right,
        T(damping), T(tension),
        s, Δs,
        w, w_dot, w_ddot,
        q, f_active,
        A_vec, I_vec, m_vec, EI_vec,
        M_mat, C_mat, K_mat,
        T(β), T(γ), T(Δt)
    )
end

"""
    build_mass_matrix(m, Δs, n, bc_left, bc_right)

Build the mass matrix M for the beam.
Uses lumped mass approximation: M[i,i] = m[i] * Δs
"""
function build_mass_matrix(m::Vector{T}, Δs::T, n::Int,
                           bc_left::BeamBoundaryCondition,
                           bc_right::BeamBoundaryCondition) where T
    # Lumped mass matrix (diagonal)
    diag_vals = m .* Δs

    # Adjust for boundary conditions
    if bc_left == CLAMPED || bc_left == PINNED || bc_left == PRESCRIBED
        diag_vals[1] = one(T)  # Will be constrained
    end
    if bc_right == CLAMPED || bc_right == PINNED || bc_right == PRESCRIBED
        diag_vals[n] = one(T)  # Will be constrained
    end

    return spdiagm(0 => diag_vals)
end

"""
    build_damping_matrix(c, Δs, n, bc_left, bc_right)

Build the damping matrix C for the beam.
Uses proportional damping: C[i,i] = c * Δs
"""
function build_damping_matrix(c::T, Δs::T, n::Int,
                              bc_left::BeamBoundaryCondition,
                              bc_right::BeamBoundaryCondition) where T
    diag_vals = fill(c * Δs, n)

    # Adjust for boundary conditions
    if bc_left == CLAMPED || bc_left == PINNED || bc_left == PRESCRIBED
        diag_vals[1] = zero(T)
    end
    if bc_right == CLAMPED || bc_right == PINNED || bc_right == PRESCRIBED
        diag_vals[n] = zero(T)
    end

    return spdiagm(0 => diag_vals)
end

"""
    build_stiffness_matrix(EI, T_tension, Δs, n, bc_left, bc_right)

Build the stiffness matrix K for the beam using standard finite difference.
Uses the symmetric 5-point stencil for the 4th derivative.

For stability, uses a regularized approach near boundaries.
"""
function build_stiffness_matrix(EI::Vector{T}, tension::T, Δs::T, n::Int,
                                bc_left::BeamBoundaryCondition,
                                bc_right::BeamBoundaryCondition) where T

    h = Δs
    h2 = h^2
    h4 = h^4

    # Initialize matrix
    K = zeros(T, n, n)

    # Standard 5-point stencil for EI * d⁴w/dx⁴: [1, -4, 6, -4, 1] / h⁴
    # Standard 3-point stencil for -T * d²w/dx²: [-1, 2, -1] / h²

    # Interior nodes: full stencil
    for i in 3:n-2
        EI_i = EI[i]
        K[i, i-2] += EI_i / h4
        K[i, i-1] += -4 * EI_i / h4 - tension / h2
        K[i, i]   += 6 * EI_i / h4 + 2 * tension / h2
        K[i, i+1] += -4 * EI_i / h4 - tension / h2
        K[i, i+2] += EI_i / h4
    end

    # === Handle boundaries ===

    # CLAMPED left: w[1] = 0, w'[1] = 0
    # Ghost node: w[0] = w[2] (from w'[1] = 0 with w[1] = 0)
    if bc_left == CLAMPED || bc_left == PRESCRIBED
        # Node 1: constrained
        K[1, :] .= zero(T)
        K[1, 1] = one(T)

        # Node 2: use ghost w[0] = w[2]
        # Stencil: (w[0] - 4w[1] + 6w[2] - 4w[3] + w[4]) / h⁴
        # = (w[2] - 0 + 6w[2] - 4w[3] + w[4]) / h⁴ = (7w[2] - 4w[3] + w[4]) / h⁴
        EI_2 = EI[2]
        K[2, 2] = 7 * EI_2 / h4 + 2 * tension / h2
        K[2, 3] = -4 * EI_2 / h4 - tension / h2
        K[2, 4] = EI_2 / h4
    elseif bc_left == PINNED
        K[1, :] .= zero(T)
        K[1, 1] = one(T)
        # w''[1] = 0: ghost w[0] = -w[2]
        EI_2 = EI[2]
        K[2, 2] = 5 * EI_2 / h4 + 2 * tension / h2
        K[2, 3] = -4 * EI_2 / h4 - tension / h2
        K[2, 4] = EI_2 / h4
    else  # FREE left
        EI_1 = EI[1]
        EI_2 = EI[2]
        # Reduced accuracy stencils
        K[1, 1] = 2 * EI_1 / h4
        K[1, 2] = -4 * EI_1 / h4
        K[1, 3] = 2 * EI_1 / h4

        K[2, 1] = -4 * EI_2 / h4
        K[2, 2] = 6 * EI_2 / h4 + 2 * tension / h2
        K[2, 3] = -4 * EI_2 / h4 - tension / h2
        K[2, 4] = EI_2 / h4
    end

    # CLAMPED right: w[n] = 0, w'[n] = 0
    if bc_right == CLAMPED || bc_right == PRESCRIBED
        K[n, :] .= zero(T)
        K[n, n] = one(T)

        # Node n-1: ghost w[n+1] = w[n-1]
        EI_nm1 = EI[n-1]
        K[n-1, n-3] = EI_nm1 / h4
        K[n-1, n-2] = -4 * EI_nm1 / h4 - tension / h2
        K[n-1, n-1] = 7 * EI_nm1 / h4 + 2 * tension / h2
    elseif bc_right == PINNED
        K[n, :] .= zero(T)
        K[n, n] = one(T)
        EI_nm1 = EI[n-1]
        K[n-1, n-3] = EI_nm1 / h4
        K[n-1, n-2] = -4 * EI_nm1 / h4 - tension / h2
        K[n-1, n-1] = 5 * EI_nm1 / h4 + 2 * tension / h2
    else  # FREE right
        # Ghost: w[n+1] = 2w[n] - w[n-1], w[n+2] = 4w[n] - 4w[n-1] + w[n-2]
        EI_n = EI[n]
        EI_nm1 = EI[n-1]

        # Node n-1
        # Stencil at n-1: [1, -4, 6, -4, 1] at [n-3, n-2, n-1, n, n+1]
        # w[n+1] = 2w[n] - w[n-1]
        # → [1, -4, 6-1, -4+2, 0] = [1, -4, 5, -2, 0]
        K[n-1, n-3] = EI_nm1 / h4
        K[n-1, n-2] = -4 * EI_nm1 / h4 - tension / h2
        K[n-1, n-1] = 5 * EI_nm1 / h4 + 2 * tension / h2
        K[n-1, n]   = -2 * EI_nm1 / h4 - tension / h2

        # Node n
        # Stencil at n: [1, -4, 6, -4, 1] at [n-2, n-1, n, n+1, n+2]
        # Substitute ghost nodes:
        # = w[n-2] - 4w[n-1] + 6w[n] - 4(2w[n]-w[n-1]) + (4w[n]-4w[n-1]+w[n-2])
        # = 2w[n-2] - 4w[n-1] + 2w[n]
        K[n, n-2] = 2 * EI_n / h4
        K[n, n-1] = -4 * EI_n / h4
        K[n, n]   = 2 * EI_n / h4
    end

    # Symmetrize for stability
    K = (K + K') / 2

    # Ensure positive semi-definiteness by shifting eigenvalues if needed
    # This is a numerical regularization for stability
    K_dense = Matrix(K)
    eig_vals = eigvals(Symmetric(K_dense))
    eig_min = minimum(eig_vals)

    if eig_min < 0
        # Add regularization to make positive semi-definite
        # Shift all eigenvalues up so minimum is slightly positive
        shift = abs(eig_min) + 0.01 * maximum(abs.(eig_vals))
        for i in 1:n
            if K_dense[i, i] != one(T)  # Don't modify identity rows
                K_dense[i, i] += shift
            end
        end
    end

    return sparse(K_dense)
end

"""
    apply_stiffness_bc!(K, n, bc_left, bc_right)

Apply boundary conditions to the stiffness matrix.
"""
function apply_stiffness_bc!(K::SparseMatrixCSC{T}, n::Int,
                             bc_left::BeamBoundaryCondition,
                             bc_right::BeamBoundaryCondition) where T
    # Left boundary
    if bc_left == CLAMPED || bc_left == PINNED || bc_left == PRESCRIBED
        # w[1] = 0: set row to identity
        K[1, :] .= zero(T)
        K[1, 1] = one(T)
    end

    # Right boundary
    if bc_right == CLAMPED || bc_right == PINNED || bc_right == PRESCRIBED
        K[n, :] .= zero(T)
        K[n, n] = one(T)
    end
end

"""
    step!(beam::EulerBernoulliBeam, Δt=beam.Δt)

Advance the beam solution by one time step using Newmark-beta method.

The Newmark-beta method updates displacement and velocity as:
    w_{n+1} = w_n + Δt*ẇ_n + Δt²*[(1/2-β)*ẅ_n + β*ẅ_{n+1}]
    ẇ_{n+1} = ẇ_n + Δt*[(1-γ)*ẅ_n + γ*ẅ_{n+1}]

With β=0.25 and γ=0.5 (average acceleration), this is unconditionally stable.
"""
function step!(beam::EulerBernoulliBeam{T}, Δt::T=beam.Δt) where T
    β, γ = beam.β, beam.γ
    n = beam.geometry.n

    # Compute effective stiffness matrix: K_eff = K + (γ/(βΔt))*C + (1/(βΔt²))*M
    a0 = one(T) / (β * Δt^2)
    a1 = γ / (β * Δt)
    a2 = one(T) / (β * Δt)
    a3 = one(T) / (2β) - one(T)
    a4 = γ / β - one(T)
    a5 = Δt * (γ / (2β) - one(T))

    K_eff = beam.K_mat + a1 * beam.C_mat + a0 * beam.M_mat

    # Compute effective force: F_eff = F + M*(a0*w + a2*ẇ + a3*ẅ) + C*(a1*w + a4*ẇ + a5*ẅ)
    F = (beam.q + beam.f_active) .* beam.Δs

    # Apply boundary conditions to force
    apply_force_bc!(F, beam)

    F_eff = F + beam.M_mat * (a0 * beam.w + a2 * beam.w_dot + a3 * beam.w_ddot) +
                beam.C_mat * (a1 * beam.w + a4 * beam.w_dot + a5 * beam.w_ddot)

    # Solve for new displacement
    w_new = K_eff \ F_eff

    # Update acceleration and velocity
    w_ddot_new = a0 * (w_new - beam.w) - a2 * beam.w_dot - a3 * beam.w_ddot
    w_dot_new = beam.w_dot + Δt * ((one(T) - γ) * beam.w_ddot + γ * w_ddot_new)

    # Store new values
    beam.w .= w_new
    beam.w_dot .= w_dot_new
    beam.w_ddot .= w_ddot_new

    return beam
end

"""
    apply_force_bc!(F, beam)

Apply boundary conditions to the force vector.
"""
function apply_force_bc!(F::Vector{T}, beam::EulerBernoulliBeam{T}) where T
    n = beam.geometry.n

    # Left boundary
    if beam.bc_left == CLAMPED || beam.bc_left == PINNED || beam.bc_left == PRESCRIBED
        F[1] = zero(T)
    end

    # Right boundary
    if beam.bc_right == CLAMPED || beam.bc_right == PINNED || beam.bc_right == PRESCRIBED
        F[n] = zero(T)
    end
end

"""
    set_fluid_load!(beam, q_func, t)

Set the fluid load on the beam from a function q(s, t).
"""
function set_fluid_load!(beam::EulerBernoulliBeam{T}, q_func::Function, t::Real) where T
    for i in 1:beam.geometry.n
        beam.q[i] = T(q_func(beam.s[i], t))
    end
end

"""
    set_fluid_load!(beam, q_values)

Set the fluid load on the beam from an array of values.
"""
function set_fluid_load!(beam::EulerBernoulliBeam{T}, q_values::AbstractVector) where T
    @assert length(q_values) == beam.geometry.n
    beam.q .= q_values
end

"""
    set_active_forcing!(beam, f_func, t)

Set the active forcing (muscle activation) from a function f(s, t).

# Example: Traveling wave muscle activation
```julia
f_active(s, t) = A_muscle * sin(k*s - ω*t) * envelope(s)
set_active_forcing!(beam, f_active, t)
```
"""
function set_active_forcing!(beam::EulerBernoulliBeam{T}, f_func::Function, t::Real) where T
    for i in 1:beam.geometry.n
        beam.f_active[i] = T(f_func(beam.s[i], t))
    end
end

"""
    set_active_forcing!(beam, f_values)

Set the active forcing from an array of values.
"""
function set_active_forcing!(beam::EulerBernoulliBeam{T}, f_values::AbstractVector) where T
    @assert length(f_values) == beam.geometry.n
    beam.f_active .= f_values
end

"""
    get_displacement(beam)

Get the displacement field w(s).
"""
get_displacement(beam::EulerBernoulliBeam) = beam.w

"""
    get_velocity(beam)

Get the velocity field ∂w/∂t(s).
"""
get_velocity(beam::EulerBernoulliBeam) = beam.w_dot

"""
    get_curvature(beam)

Compute the curvature κ = ∂²w/∂s² at each point.
"""
function get_curvature(beam::EulerBernoulliBeam{T}) where T
    n = beam.geometry.n
    Δs = beam.Δs
    κ = zeros(T, n)

    # Central difference for interior points
    for i in 2:n-1
        κ[i] = (beam.w[i-1] - 2*beam.w[i] + beam.w[i+1]) / Δs^2
    end

    # One-sided for boundaries
    if n >= 3
        κ[1] = (beam.w[1] - 2*beam.w[2] + beam.w[3]) / Δs^2
        κ[n] = (beam.w[n-2] - 2*beam.w[n-1] + beam.w[n]) / Δs^2
    end

    return κ
end

"""
    get_bending_moment(beam)

Compute the bending moment M = EI * κ at each point.
"""
function get_bending_moment(beam::EulerBernoulliBeam{T}) where T
    κ = get_curvature(beam)
    return beam.EI .* κ
end

"""
    kinetic_energy(beam)

Compute the kinetic energy of the beam: KE = ∫ (1/2) m ẇ² ds
"""
function kinetic_energy(beam::EulerBernoulliBeam{T}) where T
    KE = zero(T)
    for i in 1:beam.geometry.n
        KE += 0.5 * beam.m[i] * beam.w_dot[i]^2 * beam.Δs
    end
    return KE
end

"""
    potential_energy(beam)

Compute the potential (strain) energy of the beam: PE = ∫ (1/2) EI κ² ds
"""
function potential_energy(beam::EulerBernoulliBeam{T}) where T
    κ = get_curvature(beam)
    PE = zero(T)
    for i in 1:beam.geometry.n
        PE += 0.5 * beam.EI[i] * κ[i]^2 * beam.Δs
    end
    return PE
end

"""
    total_energy(beam)

Compute the total mechanical energy of the beam.
"""
total_energy(beam::EulerBernoulliBeam) = kinetic_energy(beam) + potential_energy(beam)

"""
    reset!(beam)

Reset the beam to its initial (undeformed) state.
"""
function reset!(beam::EulerBernoulliBeam{T}) where T
    fill!(beam.w, zero(T))
    fill!(beam.w_dot, zero(T))
    fill!(beam.w_ddot, zero(T))
    fill!(beam.q, zero(T))
    fill!(beam.f_active, zero(T))
    return beam
end
