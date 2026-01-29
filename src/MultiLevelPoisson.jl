# =============================================================================
# GEOMETRIC MULTIGRID SOLVER
# =============================================================================
# This module implements a geometric multigrid V-cycle for solving the pressure
# Poisson equation. Multigrid achieves O(N) complexity by:
#
# 1. Smoothing high-frequency errors on fine grid (Jacobi/PCG)
# 2. Restricting residual to coarser grid (2:1 coarsening)
# 3. Recursively solving coarse problem
# 4. Prolongating correction to fine grid
#
# Grid Hierarchy:
#   Level 0 (finest): N×M cells
#   Level 1:          N/2 × M/2 cells
#   Level 2:          N/4 × M/4 cells
#   ...continues until grid is too small
#
# The L coefficients include physical grid spacing with direction-specific scaling:
#   L[I,d] = μ₀[I,d] / Δx[d]²  (supports anisotropic grids)
# For highly anisotropic grids (aspect ratio > 4), convergence may be slower.
# =============================================================================

# =============================================================================
# INTER-GRID TRANSFER OPERATORS
# =============================================================================

# Map coarse index I to corresponding fine indices (2:1 ratio)
# For residual restriction: sum 2^D fine cells into 1 coarse cell
@inline up(I::CartesianIndex,a=0) = (2I-2oneunit(I)):(2I-oneunit(I)-δ(a,I))

# Map fine index I to corresponding coarse index (2:1 ratio)
# For prolongation: copy coarse value to 2^D fine cells
@inline down(I::CartesianIndex) = CI((I+2oneunit(I)).I .÷2)

# Restrict scalar field: full weighting (average of 2^N fine cells)
# R: fine -> coarse, averages 4 fine cells (2D) or 8 fine cells (3D)
# This is standard full-weighting restriction for cell-centered multigrid.
@fastmath @inline function restrict(I::CartesianIndex,b::AbstractArray{T,N}) where {T,N}
    s = zero(T)
    for J ∈ up(I)
     s += @inbounds(b[J])
    end
    # Full weighting: average over 2^N fine cells
    # 2D: 4 cells → scale = 1/4, 3D: 8 cells → scale = 1/8
    scale = inv(T(2)^N)
    return s * scale
end

# Restrict coefficient field L: average face values
# Used to build coarse grid operator from fine grid coefficients.
# Each coarse face averages 2^(N-1) fine faces (2 in 2D, 4 in 3D).
@fastmath @inline function restrictL(I::CartesianIndex,i,b::AbstractArray{T,N}) where {T,N}
    s = zero(T)
    for J ∈ up(I,i)
     s += @inbounds(b[J,i])
    end
    # Average over 2^(N-1) fine faces
    # 2D: 2 faces → scale = 1/2, 3D: 4 faces → scale = 1/4
    scale = inv(T(2)^(N-1))
    return s * scale
end

# Create coarse level Poisson from fine level (Galerkin coarsening)
function restrictML(b::Poisson{T,S,V,N}) where {T,S,V,N}
    Nsize,n = size_u(b.L)
    Na = map(i->1+i÷2,Nsize)  # Coarse grid size (2:1 coarsening)
    aL = similar(b.L,(Na...,n)); fill!(aL,0)
    ax = similar(b.x,Na); fill!(ax,0)
    restrictL!(aL,b.L,perdir=b.perdir)  # Restrict coefficients
    Poisson(ax,aL,copy(ax);perdir=b.perdir)
end

# Restrict all L coefficients from fine to coarse grid
function restrictL!(a::AbstractArray{T},b;perdir=()) where T
    Na,n = size_u(a)
    for i ∈ 1:n
        @loop a[I,i] = restrictL(I,i,b) over I ∈ CartesianIndices(map(n->2:n-1,Na))
    end
    BC!(a,zeros(SVector{n,T}),false,perdir)  # Apply BC to coarse coefficients
end

# Restriction operator: fine residual -> coarse RHS (full weighting)
restrict!(a,b) = @inside a[I] = restrict(I,b)

# Prolongation operator: coarse correction -> fine correction (injection)
# Simple injection: each fine cell gets its parent's value
prolongate!(a,b) = @inside a[I] = b[down(I)]

# Check if grid can be coarsened further (must be even and > 4)
@inline divisible(N) = mod(N,2)==0 && N>4
@inline divisible(l::Poisson) = all(size(l.x) .|> divisible)
"""
    MultiLevelPoisson{T,S,V,N}

Composite type used to solve the pressure Poisson equation with a [geometric multigrid](https://en.wikipedia.org/wiki/Multigrid_method) method.
The only variable is `levels`, a vector of nested `Poisson` systems.

The L coefficients include physical grid spacing with direction-specific scaling
(L[I,d] = μ₀[I,d]/Δx[d]²), supporting anisotropic grids where Δx ≠ Δy ≠ Δz.
"""
struct MultiLevelPoisson{T,S<:AbstractArray{T},V<:AbstractArray{T},N} <: AbstractPoisson{T,S,V}
    x::S
    L::V
    z::S
    levels :: Vector{Poisson{T,S,V,N}}
    n :: Vector{Int16}
    perdir :: NTuple # direction of periodic boundary condition
    function MultiLevelPoisson(x::AbstractArray{T,N},L::AbstractArray{T},z::AbstractArray{T};
                               maxlevels=10, perdir=()) where {T,N}
        levels = Poisson{T,typeof(x),typeof(L),N}[Poisson(x,L,z;perdir)]
        while divisible(levels[end]) && length(levels) <= maxlevels
            push!(levels,restrictML(levels[end]))
        end
        text = "MultiLevelPoisson requires size=a2ⁿ, where n>2"
        @assert (length(levels)>2) text
        new{T,typeof(x),typeof(L),N}(x,L,z,levels,[],perdir)
    end
end

# Update all levels after coefficient changes (e.g., body movement)
# Must cascade from fine to coarse since coarse L depends on fine L
function update!(ml::MultiLevelPoisson)
    update!(ml.levels[1])  # Update finest level diagonal
    for l ∈ 2:length(ml.levels)
        restrictL!(ml.levels[l].L,ml.levels[l-1].L,perdir=ml.levels[l-1].perdir)
        update!(ml.levels[l])  # Recompute diagonal for this level
    end
end

# =============================================================================
# V-CYCLE ALGORITHM
# =============================================================================
# The V-cycle is the core multigrid algorithm:
#
#   Fine ──smooth──> restrict ──> Coarse ──V-cycle──> prolongate ──> Fine
#                                    │                     │
#                                    └──────correct───────┘
#
# Properties:
# - Smoothing removes high-frequency errors (oscillatory modes)
# - Restriction transfers smooth residual to coarse grid
# - Coarse solve captures low-frequency errors efficiently
# - Prolongation interpolates correction back to fine grid
# =============================================================================
function Vcycle!(ml::MultiLevelPoisson;l=1)
    fine,coarse = ml.levels[l],ml.levels[l+1]

    # PRE-SMOOTHING: reduce high-frequency errors on fine grid
    Jacobi!(fine)

    # RESTRICTION: transfer residual to coarse grid
    restrict!(coarse.r,fine.r)  # r_coarse = R * r_fine
    fill!(coarse.x,0.)          # Start coarse solve from zero

    # COARSE SOLVE: recursively apply V-cycle or smooth at coarsest
    l+1<length(ml.levels) && Vcycle!(ml,l=l+1)
    smooth!(coarse)  # PCG smoothing at this level

    # PROLONGATION + CORRECTION: interpolate and add to fine solution
    prolongate!(fine.ϵ,coarse.x)  # ϵ_fine = P * x_coarse
    increment!(fine)              # x_fine += ϵ, r_fine -= A*ϵ
end

# Forward operations to finest level
mult!(ml::MultiLevelPoisson,x) = mult!(ml.levels[1],x)
residual!(ml::MultiLevelPoisson) = residual!(ml.levels[1])

# =============================================================================
# MULTIGRID SOLVER
# =============================================================================
# Main solver loop: repeated V-cycles until convergence
# Each V-cycle reduces error by a constant factor (typically 0.1-0.2)
# =============================================================================
function solver!(ml::MultiLevelPoisson;tol=1e-4,itmx=32)
    p = ml.levels[1]  # Finest level
    residual!(p); r₂ = L₂(p)  # Initial residual
    nᵖ=0; @log ", $nᵖ, $(L∞(p)), $r₂\n"
    while nᵖ<itmx
        Vcycle!(ml)               # One V-cycle
        smooth!(p); r₂ = L₂(p); nᵖ+=1  # Post-smoothing + check convergence
        @log ", $nᵖ, $(L∞(p)), $r₂\n"
        r₂<tol && break           # Converged
    end
    perBC!(p.x,p.perdir)  # Ensure periodic BC on final solution
    push!(ml.n,nᵖ);       # Record iteration count
end
