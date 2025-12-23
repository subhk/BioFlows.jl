# =============================================================================
# POISSON SOLVER FOR PRESSURE PROJECTION
# =============================================================================
# This module implements an iterative solver for the pressure Poisson equation
# arising from the incompressibility constraint (∇·u = 0) in the Navier-Stokes
# equations.
#
# The pressure Poisson equation is:
#     ∇·(β∇p) = σ    where β = L (variable coefficient from BDIM)
#
# Discretized on a staggered grid (with anisotropic support), this becomes:
#     Ax = z    where A includes 1/Δx[d]² scaling per direction
# =============================================================================

"""
    Poisson{N,M}

Composite type for conservative variable coefficient Poisson equations:

    ∮ds β ∂x/∂n = σ

The resulting linear system is

    Ax = [L+D+L']x = z

where A is symmetric, block-tridiagonal and extremely sparse. Moreover,
`D[I]=-∑ᵢ(L[I,i]+L'[I,i])/Δx[i]²`. This means matrix storage, multiplication,
ect can be easily implemented and optimized without external libraries.

Supports anisotropic grids where Δx[1] ≠ Δx[2] ≠ Δx[3].

# Matrix Structure
- L: Lower diagonal coefficients (off-diagonal coupling)
- D: Diagonal coefficients (scaled by 1/Δx²)
- Δx: Grid spacing tuple for anisotropic grids
- The matrix is symmetric positive semi-definite (null space = constant)

# Solver Components
- x: Solution (pressure)
- z: RHS (divergence source)
- r: Residual (z - Ax)
- ϵ: Error estimate for updates
- iD: Inverse diagonal for Jacobi preconditioning
"""
abstract type AbstractPoisson{T,S,V} end
struct Poisson{T,S<:AbstractArray{T},V<:AbstractArray{T},N} <: AbstractPoisson{T,S,V}
    L :: V   # Lower diagonal: coupling coefficients L[I,d] = β at face (I,d)
    D :: S   # Diagonal: D[I] = -Σ(L[I,d] + L[I+δd,d])/Δx[d]² for all directions d
    iD :: S  # Inverse diagonal: 1/D for Jacobi preconditioning
    x :: S   # Solution vector (pressure field)
    ϵ :: S   # Error/increment for iterative updates
    r :: S   # Residual: r = z - Ax
    z :: S   # Right-hand side (divergence source term)
    n :: Vector{Int16}  # Iteration count history
    perdir :: NTuple    # Periodic boundary directions
    inv_Δx² :: NTuple{N,T}  # Precomputed 1/Δx[d]² for each direction
    function Poisson(x::AbstractArray{T,N},L::AbstractArray{T},z::AbstractArray{T};
                     Δx::NTuple{N}=ntuple(_->one(T),N), perdir=()) where {T,N}
        # Validate array dimensions match
        @assert axes(x) == axes(z) && axes(x) == Base.front(axes(L)) && last(axes(L)) == eachindex(axes(x))
        r = similar(x); fill!(r,0)
        ϵ,D,iD = copy(r),copy(r),copy(r)
        # Precompute inverse squared grid spacing for each direction
        inv_Δx² = ntuple(d -> T(1/Δx[d]^2), N)
        # Compute diagonal from L coefficients with anisotropic scaling
        set_diag!(D,iD,L,inv_Δx²)
        new{T,typeof(x),typeof(L),N}(L,D,iD,x,ϵ,r,z,[],perdir,inv_Δx²)
    end
end

# Support for ForwardDiff automatic differentiation
using ForwardDiff: Dual,Tag
Base.eps(::Type{D}) where D<:Dual{Tag{G,T}} where {G,T} = eps(T)

# Compute diagonal and inverse diagonal from L coefficients
# The diagonal ensures row sum = 0 (conservation property)
# For anisotropic grids: D[I] = -Σ_d (L[I,d] + L[I+δd,d]) / Δx[d]²
function set_diag!(D,iD,L,inv_Δx²)
    @inside D[I] = diag(I,L,inv_Δx²)
    # Safe inverse: return 0 if D is nearly zero (solid cells)
    @inside iD[I] = abs2(D[I])<2eps(eltype(D)) ? zero(eltype(D)) : inv(D[I])
end

# Recompute diagonal after L changes (e.g., body movement)
update!(p::Poisson) = set_diag!(p.D,p.iD,p.L,p.inv_Δx²)

# Compute diagonal entry at cell I: D[I] = -Σ(L[I,d] + L[I+1,d])/Δx[d]²
# For anisotropic grids, each direction has its own scaling
@fastmath @inline function diag(I::CartesianIndex{d},L,inv_Δx²) where {d}
    s = zero(eltype(L))
    for i in 1:d
        # L[I,i]: coefficient at left/bottom face
        # L[I+δ,i]: coefficient at right/top face
        # Scale by 1/Δx[i]² for this direction
        s -= @inbounds((L[I,i]+L[I+δ(i,I),i]) * inv_Δx²[i])
    end
    return s
end

"""
    mult!(p::Poisson,x)

Efficient function for Poisson matrix-vector multiplication.
Fills `p.z = p.A x` with 0 in the ghost cells.
"""
function mult!(p::Poisson,x)
    @assert axes(p.z)==axes(x)
    perBC!(x,p.perdir)
    fill!(p.z,0)
    @inside p.z[I] = mult(I,p.L,p.D,x)
    return p.z
end
@fastmath @inline function mult(I::CartesianIndex{d},L,D,x) where {d}
    s = @inbounds(x[I]*D[I])
    for i in 1:d
        s += @inbounds(x[I-δ(i,I)]*L[I,i]+x[I+δ(i,I)]*L[I+δ(i,I),i])
    end
    return s
end

"""
    residual!(p::Poisson)

Computes the resiual `r = z-Ax` and corrects it such that
`r = 0` if `iD==0` which ensures local satisfiability
    and
`sum(r) = 0` which ensures global satisfiability.

The global correction is done by adjusting all points uniformly,
minimizing the local effect. Other approaches are possible.

Note: These corrections mean `x` is not strictly solving `Ax=z`, but
without the corrections, no solution exists.
"""
function residual!(p::Poisson)
    perBC!(p.x,p.perdir)
    @inside p.r[I] = ifelse(p.iD[I]==0,0,p.z[I]-mult(I,p.L,p.D,p.x))
    s = sum(p.r)/length(inside(p.r))
    abs(s) <= 2eps(eltype(s)) && return
    @inside p.r[I] = p.r[I]-s
end

# Update solution with error estimate: x += ϵ, r -= Aϵ
# This maintains the residual without recomputing from scratch
function increment!(p::Poisson)
    perBC!(p.ϵ,p.perdir)  # Enforce periodic BC on increment
    @loop (p.r[I] = p.r[I]-mult(I,p.L,p.D,p.ϵ);  # Update residual
           p.x[I] = p.x[I]+p.ϵ[I]) over I ∈ inside(p.x)  # Update solution
end

"""
    Jacobi!(p::Poisson; it=1)

Jacobi smoother run `it` times.
Note: This runs for general backends, but is _very_ slow to converge.

Algorithm: ϵ = D⁻¹r, then x += ϵ, r -= Aϵ
"""
@fastmath Jacobi!(p;it=1) = for _ ∈ 1:it
    @inside p.ϵ[I] = p.r[I]*p.iD[I]  # Jacobi: ϵ = D⁻¹r
    increment!(p)
end

using LinearAlgebra: ⋅

# =============================================================================
# PRECONDITIONED CONJUGATE GRADIENT (PCG) SMOOTHER
# =============================================================================
# PCG is used as a smoother for the multigrid V-cycle. It provides fast
# convergence for the smooth error components while being GPU-friendly.
#
# Algorithm:
# 1. Initialize: z = ϵ = D⁻¹r (Jacobi preconditioner), ρ = r·z
# 2. For each iteration:
#    a. Compute search direction update: z = Aϵ
#    b. Compute step size: α = ρ/(z·ϵ)
#    c. Update: x += αϵ, r -= αz
#    d. Compute new preconditioned residual: z = D⁻¹r
#    e. Compute β = (r·z)/ρ for conjugate direction
#    f. Update search direction: ϵ = βϵ + z
# =============================================================================
"""
    pcg!(p::Poisson; it=6)

Conjugate-Gradient smoother with Jacobi preditioning. Runs at most `it` iterations,
but will exit early if the Gram-Schmidt update parameter `|α| < 1%` or `|r D⁻¹ r| < 1e-8`.
Note: This runs for general backends and is the default smoother.
"""
function pcg!(p::Poisson{T};it=6) where T
    x,r,ϵ,z = p.x,p.r,p.ϵ,p.z
    # Initialize: preconditioned residual and search direction
    @inside z[I] = ϵ[I] = r[I]*p.iD[I]
    rho = r⋅z  # ρ = r·D⁻¹r (preconditioned norm)
    abs(rho)<10eps(T) && return  # Already converged
    for i in 1:it
        perBC!(ϵ,p.perdir)
        @inside z[I] = mult(I,p.L,p.D,ϵ)  # z = Aϵ
        alpha = rho/(z⋅ϵ)  # Step size (Rayleigh quotient)
        (abs(alpha)<1e-2 || abs(alpha)>1e2) && return  # Convergence check
        @loop (x[I] += alpha*ϵ[I];  # Update solution
               r[I] -= alpha*z[I]) over I ∈ inside(x)  # Update residual
        i==it && return
        @inside z[I] = r[I]*p.iD[I]  # New preconditioned residual
        rho2 = r⋅z
        abs(rho2)<10eps(T) && return  # Converged
        beta = rho2/rho  # Conjugate direction coefficient
        @inside ϵ[I] = beta*ϵ[I]+z[I]  # Update search direction
        rho = rho2
    end
end
smooth!(p) = pcg!(p)  # Default smoother

L₂(p::Poisson) = p.r ⋅ p.r # special method since outside(p.r)≡0
L∞(p::Poisson) = maximum(abs,p.r)

"""
    solver!(A::Poisson;log,tol,itmx)

Approximate iterative solver for the Poisson matrix equation `Ax=b`.

  - `A`: Poisson matrix with working arrays.
  - `A.x`: Solution vector. Can start with an initial guess.
  - `A.z`: Right-Hand-Side vector. Will be overwritten!
  - `A.n[end]`: stores the number of iterations performed.
  - `log`: If `true`, this function returns a vector holding the `L₂`-norm of the residual at each iteration.
  - `tol`: Convergence tolerance on the `L₂`-norm residual.
  - `itmx`: Maximum number of iterations.
"""
function solver!(p::Poisson;tol=1e-4,itmx=1e3)
    residual!(p); r₂ = L₂(p)
    nᵖ=0; @log ", $nᵖ, $(L∞(p)), $r₂\n"
    while nᵖ<itmx
        smooth!(p); r₂ = L₂(p); nᵖ+=1
        @log ", $nᵖ, $(L∞(p)), $r₂\n"
        r₂<tol && break
    end
    perBC!(p.x,p.perdir)
    push!(p.n,nᵖ)
end
