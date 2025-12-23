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
# Discretized on a staggered grid, this becomes a symmetric, sparse linear system:
#     Ax = z    where A = L + D + L' (tridiagonal blocks)
# =============================================================================

"""
    Poisson{N,M}

Composite type for conservative variable coefficient Poisson equations:

    ∮ds β ∂x/∂n = σ

The resulting linear system is

    Ax = [L+D+L']x = z

where A is symmetric, block-tridiagonal and extremely sparse. Moreover,
`D[I]=-∑ᵢ(L[I,i]+L'[I,i])`. This means matrix storage, multiplication,
ect can be easily implemented and optimized without external libraries.

To help iteratively solve the system above, the Poisson structure holds
helper arrays for `inv(D)`, the error `ϵ`, and residual `r=z-Ax`. An iterative
solution method then estimates the error `ϵ=̃A⁻¹r` and increments `x+=ϵ`, `r-=Aϵ`.

# Matrix Structure
- L: Lower diagonal coefficients (off-diagonal coupling)
- D: Diagonal coefficients (negative sum of L entries, ensures row sum = 0)
- The matrix is symmetric positive semi-definite (null space = constant)

# Solver Components
- x: Solution (pressure)
- z: RHS (divergence source)
- r: Residual (z - Ax)
- ϵ: Error estimate for updates
- iD: Inverse diagonal for Jacobi preconditioning
"""
abstract type AbstractPoisson{T,S,V} end
struct Poisson{T,S<:AbstractArray{T},V<:AbstractArray{T}} <: AbstractPoisson{T,S,V}
    L :: V   # Lower diagonal: coupling coefficients L[I,d] = β at face (I,d)
    D :: S   # Diagonal: D[I] = -Σ(L[I,d] + L[I+δd,d]) for all directions d
    iD :: S  # Inverse diagonal: 1/D for Jacobi preconditioning
    x :: S   # Solution vector (pressure field)
    ϵ :: S   # Error/increment for iterative updates
    r :: S   # Residual: r = z - Ax
    z :: S   # Right-hand side (divergence source term)
    n :: Vector{Int16}  # Iteration count history
    perdir :: NTuple    # Periodic boundary directions
    function Poisson(x::AbstractArray{T},L::AbstractArray{T},z::AbstractArray{T};perdir=()) where T
        # Validate array dimensions match
        @assert axes(x) == axes(z) && axes(x) == Base.front(axes(L)) && last(axes(L)) == eachindex(axes(x))
        r = similar(x); fill!(r,0)
        ϵ,D,iD = copy(r),copy(r),copy(r)
        # Compute diagonal from L coefficients
        set_diag!(D,iD,L)
        new{T,typeof(x),typeof(L)}(L,D,iD,x,ϵ,r,z,[],perdir)
    end
end

# Support for ForwardDiff automatic differentiation
using ForwardDiff: Dual,Tag
Base.eps(::Type{D}) where D<:Dual{Tag{G,T}} where {G,T} = eps(T)

# Compute diagonal and inverse diagonal from L coefficients
# The diagonal ensures row sum = 0 (conservation property)
function set_diag!(D,iD,L)
    @inside D[I] = diag(I,L)
    # Safe inverse: return 0 if D is nearly zero (solid cells)
    @inside iD[I] = abs2(D[I])<2eps(eltype(D)) ? zero(eltype(D)) : inv(D[I])
end

# Recompute diagonal after L changes (e.g., body movement)
update!(p::Poisson) = set_diag!(p.D,p.iD,p.L)

# Compute diagonal entry at cell I: D[I] = -Σ(L[I,d] + L[I+1,d])
# This is the negative sum of all coupling coefficients touching cell I
@fastmath @inline function diag(I::CartesianIndex{d},L) where {d}
    s = zero(eltype(L))
    for i in 1:d
        # L[I,i]: coefficient at left/bottom face
        # L[I+δ,i]: coefficient at right/top face
        s -= @inbounds(L[I,i]+L[I+δ(i,I),i])
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

function increment!(p::Poisson)
    perBC!(p.ϵ,p.perdir)
    @loop (p.r[I] = p.r[I]-mult(I,p.L,p.D,p.ϵ);
           p.x[I] = p.x[I]+p.ϵ[I]) over I ∈ inside(p.x)
end
"""
    Jacobi!(p::Poisson; it=1)

Jacobi smoother run `it` times.
Note: This runs for general backends, but is _very_ slow to converge.
"""
@fastmath Jacobi!(p;it=1) = for _ ∈ 1:it
    @inside p.ϵ[I] = p.r[I]*p.iD[I]
    increment!(p)
end

using LinearAlgebra: ⋅
"""
    pcg!(p::Poisson; it=6)

Conjugate-Gradient smoother with Jacobi preditioning. Runs at most `it` iterations,
but will exit early if the Gram-Schmidt update parameter `|α| < 1%` or `|r D⁻¹ r| < 1e-8`.
Note: This runs for general backends and is the default smoother.
"""
function pcg!(p::Poisson{T};it=6) where T
    x,r,ϵ,z = p.x,p.r,p.ϵ,p.z
    @inside z[I] = ϵ[I] = r[I]*p.iD[I]
    rho = r⋅z
    abs(rho)<10eps(T) && return
    for i in 1:it
        perBC!(ϵ,p.perdir)
        @inside z[I] = mult(I,p.L,p.D,ϵ)
        alpha = rho/(z⋅ϵ)
        (abs(alpha)<1e-2 || abs(alpha)>1e2) && return # alpha should be O(1)
        @loop (x[I] += alpha*ϵ[I];
               r[I] -= alpha*z[I]) over I ∈ inside(x)
        i==it && return
        @inside z[I] = r[I]*p.iD[I]
        rho2 = r⋅z
        abs(rho2)<10eps(T) && return
        beta = rho2/rho
        @inside ϵ[I] = beta*ϵ[I]+z[I]
        rho = rho2
    end
end
smooth!(p) = pcg!(p)

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
