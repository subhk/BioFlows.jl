# =============================================================================
# AUTOBODY: AUTOMATIC GEOMETRY FROM SDF
# =============================================================================
# AutoBody is the primary way to define immersed bodies in BioFlows.
# It uses automatic differentiation (ForwardDiff) to compute:
#   - Surface normal: n = ∇sdf / |∇sdf|
#   - Body velocity: V = -J⁻¹ · ∂map/∂t (from coordinate mapping)
#
# This allows complex moving/deforming bodies with minimal user code.
# =============================================================================

"""
    AutoBody(sdf,map=(x,t)->x; compose=true) <: AbstractBody

  - `sdf(x::AbstractVector,t::Real)::Real`: signed distance function
  - `map(x::AbstractVector,t::Real)::AbstractVector`: coordinate mapping function
  - `compose::Bool=true`: Flag for composing `sdf=sdf∘map`

Implicitly define a geometry by its `sdf` and optional coordinate `map`. Note: the `map`
is composed automatically if `compose=true`, i.e. `sdf(x,t) = sdf(map(x,t),t)`.
Both parameters remain independent otherwise. It can be particularly heplful to set
`compose=false` when adding mulitple bodies together to create a more complex one.

# Examples
```julia
# Static cylinder
sdf(x,t) = sqrt(x[1]^2 + x[2]^2) - radius
body = AutoBody(sdf)

# Oscillating cylinder (vertical motion)
sdf(x,t) = sqrt(x[1]^2 + x[2]^2) - radius
map(x,t) = x .- [0, A*sin(ω*t)]  # Shift coordinate frame
body = AutoBody(sdf, map)

# Rotating ellipse
sdf(x,t) = sqrt((x[1]/a)^2 + (x[2]/b)^2) - 1
map(x,t) = [cos(ω*t) sin(ω*t); -sin(ω*t) cos(ω*t)] * x
body = AutoBody(sdf, map)
```
"""
struct AutoBody{F1<:Function,F2<:Function} <: AbstractBody
    sdf::F1  # Signed distance function (possibly composed with map)
    map::F2  # Coordinate mapping for body motion
    function AutoBody(sdf, map=(x,t)->x; compose=true)
        # Optionally compose sdf with map: sdf'(x,t) = sdf(map(x,t), t)
        comp(x,t) = compose ? sdf(map(x,t),t) : sdf(x,t)
        new{typeof(comp),typeof(map)}(comp, map)
    end
end

"""
    d = sdf(body::AutoBody,x,t) = body.sdf(x,t)
"""
sdf(body::AutoBody,x,t=0;kwargs...) = body.sdf(x,t)

using ForwardDiff

# =============================================================================
# AUTOMATIC GEOMETRY MEASUREMENT USING ForwardDiff
# =============================================================================
# ForwardDiff computes exact gradients of the SDF and Jacobians of the map
# to determine:
#   1. Surface normal: n = ∇sdf / |∇sdf|
#   2. Corrected distance for pseudo-SDFs: d' = sdf / |∇sdf|
#   3. Body velocity from map: V = -J⁻¹ · (∂map/∂t)
# =============================================================================

"""
    d,n,V = measure(body::AutoBody,x,t;fastd²=Inf)

Determine the implicit geometric properties from the `sdf` and `map`.
The gradient of `d=sdf(map(x,t))` is used to improve `d` for pseudo-sdfs.
The velocity is determined _solely_ from the optional `map` function.
Skips the `n,V` calculation when `d²>fastd²`.
"""
function measure(body::AutoBody,x,t;fastd²=Inf)
    # Evaluate SDF value
    d = body.sdf(x,t)
    d^2>fastd² && return (d,zero(x),zero(x))  # Far from body, skip expensive calculations

    # Compute gradient (surface normal direction before normalization)
    n = ForwardDiff.gradient(x->body.sdf(x,t), x)
    any(isnan.(n)) && return (d,zero(x),zero(x))  # Handle degenerate cases

    # Correct for pseudo-SDF: a general implicit function f(x)=0 has |∇f| ≠ 1
    # True distance ≈ f(x) / |∇f| (first-order Taylor expansion)
    m = √sum(abs2,n)  # |∇f|
    d /= m            # Corrected distance
    n /= m            # Unit normal

    # Compute body velocity from coordinate mapping
    # For a material point ξ = map(x,t), we have Dξ/Dt = 0 (Lagrangian view)
    # This gives: ∂map/∂t + J·ẋ = 0, where J = ∂map/∂x
    # Solving: ẋ = -J⁻¹ · (∂map/∂t)
    J = ForwardDiff.jacobian(x->body.map(x,t), x)    # Jacobian of map
    dot = ForwardDiff.derivative(t->body.map(x,t), t) # Time derivative of map
    return (d, n, -J\dot)  # (distance, normal, velocity)
end

using LinearAlgebra: tr
"""
    curvature(A::AbstractMatrix)

Return `H,K` the mean and Gaussian curvature from `A=hessian(sdf)`.
`K=tr(minor(A))` in 3D and `K=0` in 2D.
"""
function curvature(A::AbstractMatrix)
    H,K = 0.5*tr(A),0
    if size(A)==(3,3)
        K = A[1,1]*A[2,2]+A[1,1]*A[3,3]+A[2,2]*A[3,3]-A[1,2]^2-A[1,3]^2-A[2,3]^2
    end
    H,K
end
