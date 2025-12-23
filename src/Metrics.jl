# =============================================================================
# FLOW METRICS AND DIAGNOSTICS
# =============================================================================
# This module provides functions to compute various flow quantities:
# - Kinetic energy
# - Velocity gradients (∂uᵢ/∂xⱼ)
# - Vorticity (curl of velocity)
# - Rate-of-strain tensor
# - Forces on immersed bodies (pressure and viscous)
# - Temporal statistics (mean flow, Reynolds stresses)
#
# All functions work on the staggered grid layout where velocities are
# face-centered and scalar fields are cell-centered.
# =============================================================================

using StaticArrays

# =============================================================================
# UTILITY FUNCTIONS FOR TENSOR OPERATIONS
# =============================================================================

# Create a StaticArray from a function f evaluated for i=1:n
Base.@propagate_inbounds @inline fSV(f,n) = SA[ntuple(f,n)...]

# Sum f(i) for i=1:n
Base.@propagate_inbounds @inline @fastmath fsum(f,n) = sum(ntuple(f,n))

# 2-norm of vector
norm2(x) = √(x'*x)

# Cyclic permutation for cross product: computes f(j,k) - f(k,j)
# where (i,j,k) is a cyclic permutation of (1,2,3)
Base.@propagate_inbounds @fastmath function permute(f,i)
    j,k = i%3+1,(i+1)%3+1  # Cyclic: 1→(2,3), 2→(3,1), 3→(1,2)
    f(j,k)-f(k,j)
end

# Cross product using permutation formula
×(a,b) = fSV(i->permute((j,k)->a[j]*b[k],i),3)

# Dot product (inner product)
@fastmath @inline function dot(a,b)
    init=zero(eltype(a))
    @inbounds for ij in eachindex(a)
     init += a[ij] * b[ij]
    end
    return init
end

"""
    ke(I::CartesianIndex,u,U=0)

Compute ``½∥𝐮-𝐔∥²`` at center of cell `I` where `U` can be used
to subtract a background flow (by default, `U=0`).
"""
ke(I::CartesianIndex{m},u,U=fSV(zero,m)) where m = 0.125fsum(m) do i
    abs2(@inbounds(u[I,i]+u[I+δ(i,I),i]-2U[i]))
end
"""
    ∂(i,j,I,u)

Compute ``∂uᵢ/∂xⱼ`` at center of cell `I`. Cross terms are computed
less accurately than inline terms because of the staggered grid.
"""
@fastmath @inline ∂(i,j,I,u) = (i==j ? ∂(i,I,u) :
        @inbounds(u[I+δ(j,I),i]+u[I+δ(j,I)+δ(i,I),i]
                 -u[I-δ(j,I),i]-u[I-δ(j,I)+δ(i,I),i])/4)

using LinearAlgebra: eigvals, Hermitian
"""
    λ₂(I::CartesianIndex{3},u)

λ₂ is a deformation tensor metric to identify vortex cores.
See [https://en.wikipedia.org/wiki/Lambda2_method](https://en.wikipedia.org/wiki/Lambda2_method) and
Jeong, J., & Hussain, F., doi:[10.1017/S0022112095000462](https://doi.org/10.1017/S0022112095000462)
"""
function λ₂(I::CartesianIndex{3},u)
    J = @SMatrix [∂(i,j,I,u) for i ∈ 1:3, j ∈ 1:3]
    S,Ω = (J+J')/2,(J-J')/2
    eigvals(Hermitian(S^2+Ω^2))[2]
end

"""
    curl(i,I,u)

Compute component `i` of ``𝛁×𝐮`` at the __edge__ of cell `I`.
For example `curl(3,CartesianIndex(2,2,2),u)` will compute
`ω₃(x=1.5,y=1.5,z=2)` as this edge produces the highest
accuracy for this mix of cross derivatives on a staggered grid.
"""
curl(i,I,u) = permute((j,k)->∂(j,CI(I,k),u), i)
"""
    ω(I::CartesianIndex{3},u)

Compute 3-vector ``𝛚=𝛁×𝐮`` at the center of cell `I`.
"""
ω(I::CartesianIndex{3},u) = fSV(i->permute((j,k)->∂(k,j,I,u),i),3)
"""
    ω_mag(I::CartesianIndex{3},u)

Compute ``∥𝛚∥`` at the center of cell `I`.
"""
ω_mag(I::CartesianIndex{3},u) = norm2(ω(I,u))
"""
    ω_mag(I::CartesianIndex{2},u)

Compute ``|ω₃|`` at the center of cell `I` for 2D flows.
In 2D, vorticity has only the out-of-plane component.
"""
ω_mag(I::CartesianIndex{2},u) = abs(curl(3,I,u))
"""
    ω_θ(I::CartesianIndex{3},z,center,u)

Compute ``𝛚⋅𝛉`` at the center of cell `I` where ``𝛉`` is the azimuth
direction around vector `z` passing through `center`.
"""
function ω_θ(I::CartesianIndex{3},z,center,u)
    θ = z × (loc(0,I,eltype(u))-SVector{3}(center))
    n = norm2(θ)
    n<=eps(n) ? 0. : θ'*ω(I,u) / n
end

# =============================================================================
# FORCE COMPUTATION ON IMMERSED BODIES
# =============================================================================
# Forces are computed by integrating pressure and viscous stresses over the
# body surface. The BDIM kernel weights the contributions smoothly.
#
# Pressure force: F_p = -∮ p n̂ dS ≈ -Σ p(I) * n(I) * K(d)
# Viscous force:  F_v = ∮ τ·n̂ dS ≈ Σ 2ν S·n̂ * K(d)
#
# where K(d) is the BDIM kernel that weights contributions near the surface.
# =============================================================================

"""
    nds(body,x,t)

BDIM-masked surface normal.
Returns n̂ weighted by the kernel K(d), which is 1 at the surface and
decays smoothly to 0 away from the body.
"""
@inline function nds(body,x,t)
    d,n,_ = measure(body,x,t,fastd²=1)
    n*BioFlows.kern(clamp(d,-1,1))  # Weight normal by kernel
end

"""
    pressure_force(sim::Simulation)

Compute the pressure force on an immersed body.
Integrates pressure times surface normal over the body using BDIM weighting.
Returns a vector [Fx, Fz] in 2D or [Fx, Fy, Fz] in 3D.
"""
pressure_force(sim) = pressure_force(sim.flow,sim.body)
pressure_force(flow,body) = pressure_force(flow.p,flow.f,body,time(flow))
function pressure_force(p,df,body,t=0)
    Tp = eltype(p); To = promote_type(Float64,Tp)
    df .= zero(Tp)
    # Compute contribution at each cell: F = Σ p * n̂ * K(d)
    @loop df[I,:] .= p[I]*nds(body,loc(0,I,Tp),t) over I ∈ inside(p)
    # Sum over all spatial dimensions to get total force vector
    sum(To,df,dims=ntuple(i->i,ndims(p)))[:] |> Array
end

"""
    S(I::CartesianIndex,u)

Rate-of-strain tensor.
"""
S(I::CartesianIndex{2},u) = @SMatrix [0.5*(∂(i,j,I,u)+∂(j,i,I,u)) for i ∈ 1:2, j ∈ 1:2]
S(I::CartesianIndex{3},u) = @SMatrix [0.5*(∂(i,j,I,u)+∂(j,i,I,u)) for i ∈ 1:3, j ∈ 1:3]
"""
   viscous_force(sim::Simulation)

Compute the viscous force on an immersed body.
"""
viscous_force(sim) = viscous_force(sim.flow,sim.body)
viscous_force(flow,body) = viscous_force(flow.u,flow.ν,flow.f,body,time(flow))
function viscous_force(u,ν,df,body,t=0)
    Tu = eltype(u); To = promote_type(Float64,Tu)
    df .= zero(Tu)
    @loop df[I,:] .= -2ν*S(I,u)*nds(body,loc(0,I,Tu),t) over I ∈ inside_u(u)
    sum(To,df,dims=ntuple(i->i,ndims(u)-1))[:] |> Array
end

"""
   total_force(sim::Simulation)

Compute the total force on an immersed body.
"""
total_force(sim) = pressure_force(sim) .+ viscous_force(sim)

using LinearAlgebra: cross
"""
    pressure_moment(x₀,sim::Simulation)

Computes the pressure moment on an immersed body relative to point x₀.
"""
pressure_moment(x₀,sim) = pressure_moment(x₀,sim.flow,sim.body)
pressure_moment(x₀,flow,body) = pressure_moment(x₀,flow.p,flow.f,body,time(flow))
function pressure_moment(x₀,p,df,body,t=0)
    Tp = eltype(p); To = promote_type(Float64,Tp)
    df .= zero(Tp)
    @loop df[I,:] .= p[I]*cross(loc(0,I,Tp)-x₀,nds(body,loc(0,I,Tp),t)) over I ∈ inside(p)
    sum(To,df,dims=ntuple(i->i,ndims(p)))[:] |> Array
end

# =============================================================================
# TEMPORAL STATISTICS (MEAN FLOW AND REYNOLDS STRESSES)
# =============================================================================
# MeanFlow accumulates running averages of flow quantities for turbulence
# statistics. Uses exponential moving average for numerical stability:
#   <f>_new = ε * f + (1-ε) * <f>_old
#   where ε = Δt / (total_time + Δt)
#
# This provides:
# - Mean velocity: U = <u>
# - Mean pressure: P = <p>
# - Reynolds stresses: τᵢⱼ = <uᵢuⱼ> - <uᵢ><uⱼ> (if uu_stats=true)
# =============================================================================

"""
     MeanFlow{T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Mf}

Holds temporal averages of pressure, velocity, and squared-velocity tensor.
The `Mf` type parameter can be `Nothing` when `uu_stats=false`, or an array type when enabled.

# Fields
- `P`: Mean pressure field
- `U`: Mean velocity field
- `UU`: Mean of uᵢuⱼ (for Reynolds stresses, optional)
- `t`: Time history vector (first and last entries define averaging window)
- `uu_stats`: Whether to track velocity correlations
"""
struct MeanFlow{T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Mf}
    P :: Sf   # Mean pressure <p>
    U :: Vf   # Mean velocity <u>
    UU :: Mf  # Mean velocity product <uᵢuⱼ> for Reynolds stresses
    t :: Vector{T}  # Time history [t_start, ..., t_current]
    uu_stats :: Bool  # Track velocity correlations?
    function MeanFlow(flow::Flow{D,T}; t_init=time(flow), uu_stats=false) where {D,T}
        mem = typeof(flow.u).name.wrapper  # Preserve array type (CPU/GPU)
        P = zeros(T, size(flow.p)) |> mem
        U = zeros(T, size(flow.u)) |> mem
        UU = uu_stats ? zeros(T, size(flow.p)..., D, D) |> mem : nothing
        new{T,typeof(P),typeof(U),typeof(UU)}(P,U,UU,T[t_init],uu_stats)
    end
    function MeanFlow(N::NTuple{D}; mem=Array, T=Float32, t_init=0, uu_stats=false) where {D}
        Ng = N .+ 2  # Include ghost cells
        P = zeros(T, Ng) |> mem
        U = zeros(T, Ng..., D) |> mem
        UU = uu_stats ? zeros(T, Ng..., D, D) |> mem : nothing
        new{T,typeof(P),typeof(U),typeof(UU)}(P,U,UU,T[t_init],uu_stats)
    end
end

# Total averaging time
time(meanflow::MeanFlow) = meanflow.t[end]-meanflow.t[1]

# Reset statistics to zero
function reset!(meanflow::MeanFlow; t_init=0.0)
    fill!(meanflow.P, 0); fill!(meanflow.U, 0)
    !isnothing(meanflow.UU) && fill!(meanflow.UU, 0)
    deleteat!(meanflow.t, collect(1:length(meanflow.t)))
    push!(meanflow.t, t_init)
end

# Update running averages with new flow state
# Uses exponential moving average: <f>_new = ε*f + (1-ε)*<f>_old
function update!(meanflow::MeanFlow, flow::Flow)
    dt = time(flow) - meanflow.t[end]
    # Weight for new sample: ε = Δt / (Δt + accumulated_time)
    ε = dt / (dt + time(meanflow) + eps(eltype(flow.p)))
    length(meanflow.t) == 1 && (ε = 1)  # First sample: just copy

    # Update mean pressure and velocity
    @loop meanflow.P[I] = ε * flow.p[I] + (1 - ε) * meanflow.P[I] over I in CartesianIndices(flow.p)
    @loop meanflow.U[Ii] = ε * flow.u[Ii] + (1 - ε) * meanflow.U[Ii] over Ii in CartesianIndices(flow.u)

    # Update velocity correlation tensor <uᵢuⱼ> for Reynolds stresses
    if meanflow.uu_stats
        for i in 1:ndims(flow.p), j in 1:ndims(flow.p)
            @loop meanflow.UU[I,i,j] = ε * (flow.u[I,i] .* flow.u[I,j]) + (1 - ε) * meanflow.UU[I,i,j] over I in CartesianIndices(flow.p)
        end
    end
    push!(meanflow.t, meanflow.t[end] + dt)
end

# Compute Reynolds stress tensor: τᵢⱼ = <uᵢuⱼ> - <uᵢ><uⱼ>
uu!(τ,a::MeanFlow) = for i in 1:ndims(a.P), j in 1:ndims(a.P)
    @loop τ[I,i,j] = a.UU[I,i,j] - a.U[I,i] * a.U[I,j] over I in CartesianIndices(a.P)
end

# Return new Reynolds stress tensor array
function uu(a::MeanFlow)
    τ = zeros(eltype(a.UU), size(a.UU)...) |> typeof(a.UU).name.wrapper
    uu!(τ,a)
    return τ
end

# Copy mean flow back to Flow struct
function Base.copy!(a::Flow, b::MeanFlow)
    a.u .= b.U
    a.p .= b.P
end
