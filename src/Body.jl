# =============================================================================
# IMMERSED BODY REPRESENTATIONS
# =============================================================================
# This module defines the interface for immersed bodies in the BDIM framework.
# Bodies are represented implicitly using signed distance functions (SDF):
#   - SDF(x) < 0: point x is inside the body
#   - SDF(x) = 0: point x is on the body surface
#   - SDF(x) > 0: point x is outside the body
#
# The BDIM method uses the SDF to compute "moment" fields (μ₀, μ₁) that
# smoothly blend the fluid equations with the solid body boundary condition.
# =============================================================================

using StaticArrays

"""
    AbstractBody

Immersed body Abstract Type. Any `AbstractBody` subtype must implement

    d,n,V = measure(body::AbstractBody, x, t=0, fastd²=Inf)

where `d` is the signed distance from `x` to the body at time `t`,
and `n` & `V` are the normal and velocity vectors implied at `x`.
A fast-approximate method can return `≈d,zero(x),zero(x)` if `d^2>fastd²`.

# Interface Methods
- `measure(body, x, t)`: Returns (distance, normal, velocity)
- `sdf(body, x, t)`: Returns just the signed distance

# Built-in Types
- `NoBody`: Empty domain (no immersed body)
- `AutoBody`: SDF-defined geometry with optional motion
- `SetBody`: Boolean operations on bodies (union, intersection, difference)
"""
abstract type AbstractBody end
"""
    measure!(flow::Flow, body::AbstractBody; t=0, ϵ=1)

Queries the body geometry to fill the arrays:

- `flow.μ₀`, Zeroth kernel moment
- `flow.μ₁`, First kernel moment scaled by the body normal
- `flow.V`,  Body velocity

at time `t` using an immersion kernel of size `ϵ`.

See Maertens & Weymouth, doi:[10.1016/j.cma.2014.09.007](https://doi.org/10.1016/j.cma.2014.09.007).
"""
function measure!(a::Flow{N,T},body::AbstractBody;t=zero(T),ϵ=1) where {N,T}
    a.V .= zero(T); a.μ₀ .= one(T); a.μ₁ .= zero(T); d²=(2+ϵ)^2
    @fastmath @inline function fill!(μ₀,μ₁,V,d,I)
        d[I] = sdf(body,loc(0,I,T),t,fastd²=d²)
        if d[I]^2<d²
            for i ∈ 1:N
                dᵢ,nᵢ,Vᵢ = measure(body,loc(i,I,T),t,fastd²=d²)
                V[I,i] = Vᵢ[i]
                μ₀[I,i] = BioFlows.μ₀(dᵢ,ϵ)
                for j ∈ 1:N
                    μ₁[I,i,j] = BioFlows.μ₁(dᵢ,ϵ)*nᵢ[j]
                end
            end
        elseif d[I]<zero(T)
            for i ∈ 1:N
                μ₀[I,i] = zero(T)
            end
        end
    end
    @loop fill!(a.μ₀,a.μ₁,a.V,a.σ,I) over I ∈ inside(a.p)
    BC!(a.μ₀,zeros(SVector{N,T}),false,a.perdir) # BC on μ₀, don't fill normal component yet
    BC!(a.V ,zeros(SVector{N,T}),a.exitBC,a.perdir)
end

# =============================================================================
# BDIM CONVOLUTION KERNEL
# =============================================================================
# The BDIM uses a smooth kernel to transition from fluid (d > ϵ) to solid (d < -ϵ)
# over a band of width 2ϵ centered at the body surface (d = 0).
#
# The kernel is a raised cosine: K(d) = 0.5 + 0.5*cos(πd) for d ∈ [-1, 1]
# This gives C¹ continuity at the transition boundaries.
#
# Moment integrals:
# - μ₀(d): ∫_{-1}^{d/ϵ} K(s) ds = volume fraction (0 = solid, 1 = fluid)
# - μ₁(d): ∫_{-1}^{d/ϵ} s*K(s) ds = first moment for gradient correction
# =============================================================================

# Kernel function (used for visualization, not in solver)
@fastmath kern(d) = 0.5+0.5cos(π*d)

# Zeroth moment: integrated kernel = volume fraction
@fastmath kern₀(d) = 0.5+0.5d+0.5sin(π*d)/π

# First moment: integrated kernel weighted by distance
@fastmath kern₁(d) = 0.25*(1-d^2)-0.5*(d*sin(π*d)+(1+cos(π*d))/π)/π

# μ₀: volume fraction at distance d with kernel width ϵ
# Returns 0 (inside solid), 1 (outside solid), smooth transition in between
μ₀(d,ϵ) = kern₀(clamp(d/ϵ,-1,1))

# μ₁: first moment scaled by ϵ for gradient correction
μ₁(d,ϵ) = ϵ*kern₁(clamp(d/ϵ,-1,1))

"""
    d = sdf(a::AbstractBody,x,t=0;fastd²=0)

Measure only the distance. Defaults to fastd²=0 for quick evaluation.
"""
sdf(body::AbstractBody,x,t=0;fastd²=0) = measure(body,x,t;fastd²)[1]

"""
    measure_sdf!(a::AbstractArray, body::AbstractBody, t=0; fastd²=0)

Uses `sdf(body,x,t)` to fill `a`. Defaults to fastd²=0 for quick evaluation.
"""
function measure_sdf!(a::AbstractArray{T},body::AbstractBody,t=zero(T);fastd²=zero(T)) where T
    @inside a[I] = sdf(body,loc(0,I,T),t;fastd²)
end

"""
    NoBody

Use for a simulation without a body.
"""
struct NoBody <: AbstractBody end
measure(::NoBody,x::AbstractVector,args...;kwargs...)=(Inf,zero(x),zero(x))
function measure!(::Flow,::NoBody;kwargs...) end # skip measure! entirely

# =============================================================================
# CONSTRUCTIVE SOLID GEOMETRY (CSG) OPERATIONS
# =============================================================================
# SetBody allows combining bodies using boolean operations:
# - Union (∪ or +): points inside either body
# - Intersection (∩): points inside both bodies
# - Difference (-): points inside first but not second
# - Complement (-a): swap inside/outside
#
# For SDFs, these operations are:
# - Union: min(sdf_a, sdf_b)
# - Intersection: max(sdf_a, sdf_b)
# - Complement: -sdf
# =============================================================================

"""
    SetBody

Body defined as a lazy set operation on two `AbstractBody`s.
The operations are only evaluated when `measure`d.
"""
struct SetBody{O<:Function,Ta<:AbstractBody,Tb<:AbstractBody} <: AbstractBody
    op::O   # Operation: min (union), max (intersection), or - (complement)
    a::Ta   # First body
    b::Tb   # Second body (NoBody for unary operations)
end

# Lazy constructors - create SetBody without evaluating SDFs
Base.:∪(a::AbstractBody, b::AbstractBody) = SetBody(min,a,b)  # Union: min SDF
Base.:+(a::AbstractBody, b::AbstractBody) = a ∪ b             # Alias for union
Base.:∩(a::AbstractBody, b::AbstractBody) = SetBody(max,a,b)  # Intersection: max SDF
Base.:-(a::AbstractBody) = SetBody(-,a,NoBody())              # Complement: negate SDF
Base.:-(a::AbstractBody, b::AbstractBody) = a ∩ (-b)          # Difference: a ∩ (not b)

# Measure SetBody by evaluating operation on component measurements
function measure(body::SetBody,x,t;fastd²=Inf)
    # Apply operation (min/max) to the full measurement tuples
    body.op(measure(body.a,x,t;fastd²),measure(body.b,x,t;fastd²))
end

# Special case for complement: negate distance and normal, keep velocity
measure(body::SetBody{typeof(-)},x,t;fastd²=Inf) = ((d,n,V) = measure(body.a,x,t;fastd²); (-d,-n,V))
