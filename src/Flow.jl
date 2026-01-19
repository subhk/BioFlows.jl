# =============================================================================
# FINITE DIFFERENCE OPERATORS FOR STAGGERED (MAC) GRID
# =============================================================================
# Grid layout:
#   - Scalar fields (p, σ): cell-centered
#   - Vector fields (u): face-centered (u_x at x-faces, u_z at z-faces)
# =============================================================================

# Backward difference: ∂f/∂x ≈ f[I] - f[I-1] (for scalar field)
@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-δ(a,I)]

# Forward difference: ∂u_a/∂x_a ≈ u[I+1,a] - u[I,a] (for velocity component)
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]

# Face-to-cell interpolation: average neighboring face values
@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])/2

# =============================================================================
# CONVECTION SCHEMES: Compute face values from cell values
# Arguments: (u)pwind, (c)enter, (d)ownwind cell values
# =============================================================================

# QUICK: Quadratic Upwind - 3rd order with median limiter for stability
@fastmath quick(u,c,d) = median((5c+2d-u)/6,c,median(10c-9u,c,d))

# van Leer: 2nd order TVD scheme - prevents spurious oscillations
@fastmath vanLeer(u,c,d) = (c≤min(u,d) || c≥max(u,d)) ? c : c+(d-c)*(c-u)/(d-u)

# Central Difference Scheme: 2nd order, simple but can oscillate
@fastmath cds(u,c,d) = (c+d)/2

# =============================================================================
# CONVECTIVE FLUX FUNCTIONS
# Select upwind stencil based on flow direction, apply scheme λ
# =============================================================================

# Interior: full upwind stencil available on both sides
@inline ϕu(a,I,f,u,λ) = @inbounds u>0 ? u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])

# Periodic BC: wrap index Ip for upwind cell across boundary
@inline ϕuP(a,Ip,I,f,u,λ) = @inbounds u>0 ? u*λ(f[Ip],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])

# Left boundary: central diff for outflow (u>0), upwind for inflow (u<0)
@inline ϕuL(a,I,f,u,λ) = @inbounds u>0 ? u*ϕ(a,I,f) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])

# Right boundary: central diff for outflow (u<0), upwind for inflow (u>0)
@inline ϕuR(a,I,f,u,λ) = @inbounds u<0 ? u*ϕ(a,I,f) : u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I])

# =============================================================================
# DIVERGENCE AND BDIM OPERATORS
# =============================================================================

# Velocity divergence: ∇·u = Σ ∂u_i/∂x_i (unit spacing, Δx=1)
# Used to compute pressure source term in projection step
@fastmath @inline function div(I::CartesianIndex{m},u) where {m}
    init=zero(eltype(u))
    for i in 1:m
     init += @inbounds ∂(i,I,u)
    end
    return init
end

# Anisotropic divergence: ∇·u = Σ (∂u_i/∂x_i) with proper Δx scaling
# For anisotropic grids where Δx ≠ Δy ≠ Δz
@fastmath @inline function div_aniso(I::CartesianIndex{m},u,Δx) where {m}
    init=zero(eltype(u))
    for i in 1:m
     init += @inbounds ∂(i,I,u) / Δx[i]
    end
    return init
end

# BDIM first-moment correction: μ₁·∇f (directional derivative weighted by μ₁)
# Part of the immersed boundary forcing that smoothly transitions flow at body surface
@fastmath @inline function μddn(I::CartesianIndex{np1},μ,f) where np1
    s = zero(eltype(f))
    for j ∈ 1:np1-1
        s+= @inbounds μ[I,j]*(f[I+δ(j,I)]-f[I-δ(j,I)])
    end
    return 0.5s
end

# Median of three values - used by QUICK scheme as flux limiter
# Branchless-friendly implementation for performance
function median(a,b,c)
    if a>b
        b>=c && return b
        a>c && return c
    else
        b<=c && return b
        a<c && return c
    end
    return a
end

"""
    conv_diff!(r, u, Φ, λ; ν, Δx, perdir=())

Compute convective and diffusive fluxes for the momentum equation in conservative form.

The momentum equation RHS is computed as:
    r_i = -∑_j ∂(u_j u_i)/∂x_j + ν ∑_j ∂²u_i/∂x_j²

For 2D (i,j ∈ {1,2} = {x,z}):
- u-momentum (i=1): r₁ = -∂(uu)/∂x - ∂(wu)/∂z + ν(∂²u/∂x² + ∂²u/∂z²)
- w-momentum (i=2): r₂ = -∂(uw)/∂x - ∂(ww)/∂z + ν(∂²w/∂x² + ∂²w/∂z²)

The convective terms are in **conservative flux form** ∂(u_j u_i)/∂x_j,
not the non-conservative form u_j ∂u_i/∂x_j.

# Arguments
- `r`: RHS accumulator (output)
- `u`: Velocity field
- `Φ`: Flux work array
- `λ`: Convection scheme (quick, vanLeer, cds)
- `ν`: Kinematic viscosity (m²/s)
- `Δx`: Grid spacing tuple (m), e.g., `(Δx, Δz)` or `(Δx, Δy, Δz)`
- `perdir`: Tuple of periodic directions
"""
function conv_diff!(r,u,Φ,λ::F;ν=0.1f0,Δx=(1,1),perdir=()) where {F}
    r .= zero(eltype(r))
    N,n = size_u(u)
    T = eltype(r)
    for i ∈ 1:n, j ∈ 1:n
        # Direction-specific scaling for anisotropic grids
        inv_Δxj = T(1/Δx[j])          # For convective term in direction j
        ν_over_Δxj = T(ν/Δx[j]^2)     # For diffusive term in direction j
        # if it is periodic direction
        tagper = (j in perdir)
        # treatment for bottom boundary with BCs
        lowerBoundary!(r,u,Φ,ν_over_Δxj,inv_Δxj,i,j,N,λ,Val{tagper}())
        # inner cells: Φ = convective_flux/Δx[j] - ν*diffusive_flux/Δx[j]²
        @loop (Φ[I] = inv_Δxj*ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ) - ν_over_Δxj*∂(j,CI(I,i),u);
               r[I,i] += Φ[I]) over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        # treatment for upper boundary with BCs
        upperBoundary!(r,u,Φ,ν_over_Δxj,inv_Δxj,i,j,N,λ,Val{tagper}())
    end
end

# Neumann BC Building block (dimensional form)
lowerBoundary!(r,u,Φ,ν_Δx2,inv_Δx,i,j,N,λ,::Val{false}) = @loop r[I,i] += inv_Δx*ϕuL(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ) - ν_Δx2*∂(j,CI(I,i),u) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,Φ,ν_Δx2,inv_Δx,i,j,N,λ,::Val{false}) = @loop r[I-δ(j,I),i] += -inv_Δx*ϕuR(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ) + ν_Δx2*∂(j,CI(I,i),u) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block (dimensional form)
lowerBoundary!(r,u,Φ,ν_Δx2,inv_Δx,i,j,N,λ,::Val{true}) = @loop (
    Φ[I] = inv_Δx*ϕuP(j,CIj(j,CI(I,i),N[j]-2),CI(I,i),u,ϕ(i,CI(I,j),u),λ) - ν_Δx2*∂(j,CI(I,i),u); r[I,i] += Φ[I]) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,Φ,ν_Δx2,inv_Δx,i,j,N,λ,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)

# =============================================================================
# FINITE VOLUME METHOD (FVM) FLUX COMPUTATION
# =============================================================================
# These functions compute and store fluxes explicitly at cell faces for
# conservative finite volume discretization. Each flux is computed once
# and applied to both adjacent cells with opposite signs.
# =============================================================================

"""
    compute_face_flux!(F_conv, F_diff, u, λ; ν, Δx, perdir)

Compute and store convective and diffusive fluxes at all cell faces.

The convective flux of momentum component i through face j is:
    F_conv[I,j,i] = (u_j)_face · ϕ(u_i) / Δx_j

where ϕ(u_i) is the upwind-reconstructed value of u_i at the face.
This represents the conservative flux ∂(u_j u_i)/∂x_j.

For 2D, the fluxes computed are:
- F_conv[I,1,1] = ∂(uu)/∂x,  F_conv[I,2,1] = ∂(wu)/∂z  (u-momentum)
- F_conv[I,1,2] = ∂(uw)/∂x,  F_conv[I,2,2] = ∂(ww)/∂z  (w-momentum)

Fluxes are applied conservatively: the same flux value is added to one
cell and subtracted from its neighbor, ensuring exact momentum conservation.

# Arguments
- `F_conv`: Convective flux storage [I,j,i] = flux of u_i through face j at index I
- `F_diff`: Diffusive flux storage [I,j,i]
- `u`: Velocity field
- `λ`: Convection scheme (quick, vanLeer, cds)
- `ν`: Kinematic viscosity (m²/s)
- `Δx`: Grid spacing tuple (m)
- `perdir`: Tuple of periodic directions
"""
function compute_face_flux!(F_conv,F_diff,u,λ::F;ν=0.1f0,Δx=(1,1),perdir=()) where {F}
    N,n = size_u(u)
    T = eltype(u)
    # Clear flux arrays
    F_conv .= zero(T)
    F_diff .= zero(T)
    for i ∈ 1:n, j ∈ 1:n
        inv_Δxj = T(1/Δx[j])
        ν_Δxj = T(ν/Δx[j]^2)
        tagper = (j in perdir)
        # Compute interior fluxes: convective + diffusive
        # Note: ∂(j,CI(I,i),u) = u[CI(I,i)] - u[CI(I,i)-δ(j,CI(I,i))] with proper dimensions
        @loop (F_conv[I,j,i] = inv_Δxj * ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ);
               F_diff[I,j,i] = -ν_Δxj * ∂(j,CI(I,i),u)) over I ∈ inside_u(N,j)
        # Compute boundary fluxes
        compute_boundary_flux!(F_conv,F_diff,u,inv_Δxj,ν_Δxj,i,j,N,λ,Val{tagper}())
    end
end

# Neumann boundary flux (non-periodic)
function compute_boundary_flux!(F_conv,F_diff,u,inv_Δx,ν_Δx2,i,j,N,λ,::Val{false})
    # Lower boundary: use ϕuL stencil
    @loop (F_conv[I,j,i] = inv_Δx * ϕuL(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ);
           F_diff[I,j,i] = -ν_Δx2 * ∂(j,CI(I,i),u)) over I ∈ slice(N,2,j,2)
    # Upper boundary: use ϕuR stencil
    @loop (F_conv[I,j,i] = inv_Δx * ϕuR(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ);
           F_diff[I,j,i] = -ν_Δx2 * ∂(j,CI(I,i),u)) over I ∈ slice(N,N[j],j,2)
end

# Periodic boundary flux
function compute_boundary_flux!(F_conv,F_diff,u,inv_Δx,ν_Δx2,i,j,N,λ,::Val{true})
    # Lower boundary: use ϕuP stencil with wrapped index
    @loop (F_conv[I,j,i] = inv_Δx * ϕuP(j,CIj(j,CI(I,i),N[j]-2),CI(I,i),u,ϕ(i,CI(I,j),u),λ);
           F_diff[I,j,i] = -ν_Δx2 * ∂(j,CI(I,i),u)) over I ∈ slice(N,2,j,2)
    # Upper boundary: copy lower boundary flux (periodic wrap)
    @loop (F_conv[I,j,i] = F_conv[CIj(j,I,2),j,i];
           F_diff[I,j,i] = F_diff[CIj(j,I,2),j,i]) over I ∈ slice(N,N[j],j,2)
end

"""
    apply_fluxes!(r, F_conv, F_diff)

Apply stored fluxes to RHS using conservative finite volume formulation.

Each flux contributes to exactly two cells with opposite signs:
    r[I,i] += F[I,j,i]       (flux enters cell I)
    r[I-δ,i] -= F[I,j,i]     (flux leaves cell I-δ)

This ensures global conservation: sum of all fluxes = 0.
"""
function apply_fluxes!(r,F_conv,F_diff)
    N,n = size_u(r)  # Use r (velocity RHS) not F_conv (flux tensor) for dimensions
    T = eltype(r)
    r .= zero(T)
    for i ∈ 1:n, j ∈ 1:n
        # Lower boundary: flux enters cell at index 2 only (no neighbor outside domain)
        @loop r[I,i] += F_conv[I,j,i] + F_diff[I,j,i] over I ∈ slice(N,2,j,2)
        # Interior faces: flux enters cell I, leaves cell I-δ
        @loop r[I,i] += F_conv[I,j,i] + F_diff[I,j,i] over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= F_conv[I,j,i] + F_diff[I,j,i] over I ∈ inside_u(N,j)
        # Upper boundary: flux leaves cell at index N[j]-1 only
        @loop r[I-δ(j,I),i] -= F_conv[I,j,i] + F_diff[I,j,i] over I ∈ slice(N,N[j],j,2)
    end
end

"""
    conv_diff_fvm!(r, u, F_conv, F_diff, λ; ν, Δx, perdir)

Finite Volume Method for convection-diffusion with explicit flux storage.

Computes fluxes, stores them in F_conv/F_diff, and applies them conservatively.
This is the FVM alternative to conv_diff! for use when store_fluxes=true.
"""
function conv_diff_fvm!(r,u,F_conv,F_diff,λ::F;ν=0.1f0,Δx=(1,1),perdir=()) where {F}
    compute_face_flux!(F_conv,F_diff,u,λ;ν,Δx,perdir)
    apply_fluxes!(r,F_conv,F_diff)
end

"""
    accelerate!(r,t,g,U)

Accounts for applied and reference-frame acceleration using `rᵢ += g(i,x,t)+dU(i,x,t)/dt`
"""
accelerate!(r,t,::Nothing,::Union{Nothing,Tuple}) = nothing
accelerate!(r,t,f::Function) = @loop r[Ii] += f(last(Ii),loc(Ii,eltype(r)),t) over Ii ∈ CartesianIndices(r)
accelerate!(r,t,g::Function,::Union{Nothing,Tuple}) = accelerate!(r,t,g)
accelerate!(r,t,::Nothing,U::Function) = accelerate!(r,t,(i,x,t)->ForwardDiff.derivative(τ->U(i,x,τ),t))
accelerate!(r,t,g::Function,U::Function) = accelerate!(r,t,(i,x,t)->g(i,x,t)+ForwardDiff.derivative(τ->U(i,x,τ),t))
"""
    Flow{D::Int, T::Float, Sf<:AbstractArray{T,D}, Vf<:AbstractArray{T,D+1}, Tf<:AbstractArray{T,D+2}}

Composite type for a multidimensional immersed boundary flow simulation.

Flow solves the unsteady incompressible [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) on a Cartesian grid.
Solid boundaries are modelled using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/).
The primary variables are the scalar pressure `p` (an array of dimension `D`)
and the velocity vector field `u` (an array of dimension `D+1`).

The equations solved are the dimensional incompressible Navier-Stokes:
    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + g
    ∇·u = 0

where `Δx` is the uniform grid spacing (m), `ν` is kinematic viscosity (m²/s),
and all spatial derivatives are properly scaled by `Δx`.
"""
struct Flow{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Tf<:AbstractArray{T}}
    # Fluid fields
    u :: Vf # velocity vector field (m/s)
    u⁰:: Vf # previous velocity (m/s)
    f :: Vf # force/RHS vector (m/s²)
    p :: Sf # pressure scalar field (Pa = kg/(m·s²))
    σ :: Sf # divergence scalar (work array)
    # BDIM fields
    V :: Vf # body velocity vector (m/s)
    μ₀:: Vf # zeroth-moment vector (dimensionless)
    μ₁:: Tf # first-moment tensor field (dimensionless)
    # FVM flux storage (optional)
    F_conv :: Union{Tf, Nothing} # convective flux tensor F[I,j,i] = flux of u_i through face j
    F_diff :: Union{Tf, Nothing} # diffusive flux tensor
    store_fluxes :: Bool # flag to enable FVM flux storage
    # Non-fields
    inletBC :: Union{NTuple{D,Number},Function} # inlet boundary velocity (m/s)
    Δt:: Vector{T} # time step history (s)
    ν :: T # kinematic viscosity (m²/s)
    ρ :: T # fluid density (kg/m³), default 1000 for water
    Δx :: NTuple{D,T} # grid spacing per direction (m) - can be anisotropic
    g :: Union{Function,Nothing} # acceleration field (m/s²)
    outletBC :: Bool # convective outlet BC flag
    perdir :: NTuple # periodic directions tuple
    fixed_Δt :: Union{T,Nothing} # fixed time step (nothing = adaptive CFL)
    """
        Flow(N; L, inletBC=nothing, ν=0, ρ=1000, Δt=0.25, fixed_Δt=nothing, ...)

    Construct a Flow on grid of size `N` with domain size `L`.

    # Required Arguments
    - `N::NTuple{D}`: Number of grid cells, e.g., `(nx, nz)` or `(nx, ny, nz)`
    - `L::NTuple{D}`: Physical domain size (m), e.g., `(2.0, 1.0)` for 2m × 1m
      Grid spacing: `Δx[d] = L[d]/N[d]` for each direction d
      Supports anisotropic grids (Δx ≠ Δy ≠ Δz)

    # Optional Arguments
    - `inletBC`: Inlet boundary velocity (m/s). Tuple or `Function(i,x,t)`.
      Default: unit velocity in x-direction `(1, 0, ...)`.
    - `ν=0.`: Kinematic viscosity (m²/s). Water ≈ 1e-6, air ≈ 1.5e-5
    - `ρ=1000.`: Fluid density (kg/m³). Water = 1000, air ≈ 1.2
    - `Δt=0.25`: Initial time step (s). Used as first step, then adaptive CFL unless `fixed_Δt` is set.
    - `fixed_Δt=nothing`: Fixed time step (s). If specified, disables adaptive CFL time stepping.
    - `g=nothing`: Body acceleration function `g(i,x,t)` returning m/s²
    - `uλ=nothing`: Initial velocity. Tuple or `Function(i,x)`
    - `perdir=()`: Periodic directions, e.g., `(2,)` for y-periodic
    - `outletBC=false`: Convective outlet BC in direction 1
    - `store_fluxes=false`: Enable FVM flux storage for conservation analysis
    - `T=Float32`: Numeric type
    - `f=Array`: Memory backend

    # Example
    ```julia
    # Water flow (default density)
    flow = Flow((200, 100); L=(2.0, 1.0), inletBC=(1.0, 0.0), ν=1e-6)

    # Air flow
    flow = Flow((200, 100); L=(2.0, 1.0), inletBC=(1.0, 0.0), ν=1.5e-5, ρ=1.2)
    ```
    """
    function Flow(N::NTuple{D}; L::NTuple{D}, inletBC=nothing, f=Array, Δt=0.25f0, ν=0f0, ρ=1000f0, g=nothing,
            uλ=nothing, perdir=(), outletBC=false, store_fluxes=false, T=Float32, fixed_Δt=nothing) where D
        # Default inletBC: unit velocity in x-direction
        if isnothing(inletBC)
            inletBC = ntuple(i -> i==1 ? one(T) : zero(T), D)
        end
        # Compute grid spacing for each direction (supports anisotropic grids)
        Δx = ntuple(d -> T(L[d] / N[d]), D)
        Ng = N .+ 2
        Nd = (Ng..., D)
        isnothing(uλ) && (uλ = ic_function(inletBC))
        u = Array{T}(undef, Nd...) |> f
        isa(uλ, Function) ? apply!(uλ, u) : apply!((i,x)->uλ[i], u)
        BC!(u,inletBC,outletBC,perdir); exitBC!(u,u,0.)
        u⁰ = copy(u)
        fv, p, σ = zeros(T, Nd) |> f, zeros(T, Ng) |> f, zeros(T, Ng) |> f
        V, μ₀, μ₁ = zeros(T, Nd) |> f, ones(T, Nd) |> f, zeros(T, Ng..., D, D) |> f
        BC!(μ₀,ntuple(zero, D),false,perdir)
        # Initialize FVM flux storage if requested
        if store_fluxes
            F_conv = zeros(T, Ng..., D, D) |> f
            F_diff = zeros(T, Ng..., D, D) |> f
        else
            F_conv = nothing
            F_diff = nothing
        end
        # Convert fixed_Δt to correct type if specified
        fixed_dt = isnothing(fixed_Δt) ? nothing : T(fixed_Δt)
        new{D,T,typeof(p),typeof(u),typeof(μ₁)}(u,u⁰,fv,p,σ,V,μ₀,μ₁,F_conv,F_diff,store_fluxes,inletBC,T[Δt],T(ν),T(ρ),Δx,g,outletBC,perdir,fixed_dt)
    end
end

"""
    time(a::Flow)

Current flow time.
"""
time(a::Flow) = sum(@view(a.Δt[1:end-1]))

# =============================================================================
# BOUNDARY DATA IMMERSION METHOD (BDIM)
# =============================================================================
# BDIM enforces no-slip/no-penetration at immersed boundaries by smoothly
# blending the fluid velocity with the body velocity using moment fields:
#   μ₀: volume fraction (1 = fluid, 0 = solid)
#   μ₁: directional moments for gradient correction
#   V:  body velocity at each point
#
# The update formula interpolates between predicted velocity and body velocity:
#   u = μ₀*(u_predicted) + (1-μ₀)*V + μ₁·∇(correction)
# =============================================================================
function BDIM!(a::Flow)
    dt = a.Δt[end]
    # Compute correction field: f = u⁰ + Δt*RHS - V
    @loop a.f[Ii] = a.u⁰[Ii]+dt*a.f[Ii]-a.V[Ii] over Ii in CartesianIndices(a.f)
    # Apply BDIM blending: u += μ₁·∇f + V + μ₀*f
    @loop a.u[Ii] += μddn(Ii,a.μ₁,a.f)+a.V[Ii]+a.μ₀[Ii]*a.f[Ii] over Ii ∈ inside_u(size(a.p))
end

"""
    project!(a::Flow, b::AbstractPoisson, w=1)

Project velocity onto divergence-free space using pressure Poisson equation.

Solves: ∇²p = ρ·∇·u*/Δt
Updates: u = u* - (Δt/ρ)·∇p

The pressure field `p` has physical units of Pa (kg/(m·s²)).

The physical grid spacing Δx = L/N is properly incorporated:
- RHS: ρ * Δx * div_unit(u) balances the unit-spacing Laplacian (which gives Δx² * ∇²p)
- Correction: u -= L * ∂p / (ρ * Δx[i]) applies physical gradient (∇p = ∂p/Δx)

For isotropic grids (Δx = Δy = Δz = h), the resulting pressure has physical units.
For anisotropic grids, the minimum grid spacing is used for consistency.

The physical Δx is used throughout:
- CFL time step computation (Flow.jl CFL function)
- Convection-diffusion operators (conv_diff! uses physical Δx)
- Pressure projection (this function)
- Force computations (Metrics.jl integrates over physical domain)
"""
function project!(a::Flow{n},b::AbstractPoisson,w=1) where n
    dt = w*a.Δt[end]
    ρ = a.ρ
    Δx = a.Δx
    # For isotropic grids, use Δx[1]. For anisotropic, use minimum for stability.
    h = minimum(Δx)
    # Physical Poisson: ∇²p = ρ * ∇·u
    # With unit-spacing Laplacian (gives Δx² * ∇²p), RHS must be scaled:
    # Δ²p = h² * ∇²p = h² * ρ * ∇·u = h * ρ * (h * ∇·u) = h * ρ * div_unit
    @inside b.z[I] = ρ * h * div(I, a.u)
    b.x .*= dt  # Scale initial guess for warm start
    solver!(b)
    # Physical velocity correction: u -= (1/ρ) * ∇p = (1/ρ) * (1/Δx) * ∂p
    # With unit-spacing difference ∂, physical gradient is ∂p/Δx[i]
    for i ∈ 1:n
        inv_ρΔx = inv(ρ * Δx[i])
        @loop a.u[I,i] -= b.L[I,i] * ∂(i, I, b.x) * inv_ρΔx over I ∈ inside(b.x)
    end
    b.x ./= dt  # Unscale to recover actual pressure
end

"""
    mom_step!(a::Flow,b::AbstractPoisson;λ=quick,udf=nothing,kwargs...)

Integrate the `Flow` one time step using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/)
and the `AbstractPoisson` pressure solver to project the velocity onto an incompressible flow.

Solves the dimensional incompressible Navier-Stokes equations:
    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + g
    ∇·u = 0

Uses predictor-corrector time integration with proper Δx scaling.
"""
@fastmath function mom_step!(a::Flow{N,T},b::AbstractPoisson;λ=quick,udf=nothing,kwargs...) where {N,T}
    a.u⁰ .= a.u; scale_u!(a,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
    # predictor u → u'
    @log "p"
    if a.store_fluxes
        conv_diff_fvm!(a.f,a.u⁰,a.F_conv,a.F_diff,λ;ν=a.ν,Δx=a.Δx,perdir=a.perdir)
    else
        conv_diff!(a.f,a.u⁰,a.σ,λ;ν=a.ν,Δx=a.Δx,perdir=a.perdir)
    end
    udf!(a,udf,t₀; kwargs...)
    accelerate!(a.f,t₀,a.g,a.inletBC)
    BDIM!(a); BC!(a.u,a.inletBC,a.outletBC,a.perdir,t₁) # BC MUST be at t₁
    a.outletBC && exitBC!(a.u,a.u⁰,a.Δt[end]) # convective outlet
    project!(a,b); BC!(a.u,a.inletBC,a.outletBC,a.perdir,t₁)
    # corrector u → u¹
    @log "c"
    if a.store_fluxes
        conv_diff_fvm!(a.f,a.u,a.F_conv,a.F_diff,λ;ν=a.ν,Δx=a.Δx,perdir=a.perdir)
    else
        conv_diff!(a.f,a.u,a.σ,λ;ν=a.ν,Δx=a.Δx,perdir=a.perdir)
    end
    udf!(a,udf,t₁; kwargs...)
    accelerate!(a.f,t₁,a.g,a.inletBC)
    BDIM!(a); scale_u!(a,T(0.5)); BC!(a.u,a.inletBC,a.outletBC,a.perdir,t₁)
    project!(a,b,T(0.5)); BC!(a.u,a.inletBC,a.outletBC,a.perdir,t₁)
    # Use fixed time step if specified, otherwise adaptive CFL
    next_dt = isnothing(a.fixed_Δt) ? CFL(a) : a.fixed_Δt
    push!(a.Δt, next_dt)
end
scale_u!(a,scale) = @loop a.u[Ii] *= scale over Ii ∈ inside_u(size(a.p))

"""
    CFL(a::Flow; Δt_max=10)

Compute CFL-stable time step for dimensional Navier-Stokes with anisotropic grids.

For anisotropic grids, the CFL condition considers each direction:
- Convective: Δt ≤ 1 / Σ_d (u_d / Δx[d])
- Diffusive: Δt ≤ 1 / Σ_d (2ν / Δx[d]²)

Combined: Δt ≤ 1 / (Σ_d u_d/Δx[d] + Σ_d 2ν/Δx[d]²)
"""
function CFL(a::Flow{D};Δt_max=10) where D
    @inside a.σ[I] = flux_out_aniso(I,a.u,a.Δx)
    max_flux = maximum(a.σ)
    Δx = a.Δx
    # Diffusive CFL: Σ_d 2ν/Δx[d]²
    diffusive = sum(d -> 2*a.ν/Δx[d]^2, 1:D)
    min(Δt_max, inv(max_flux + diffusive))
end

# Anisotropic flux out: Σ_d (max(0,u[I+d,d])/Δx[d] + max(0,-u[I,d])/Δx[d])
@fastmath @inline function flux_out_aniso(I::CartesianIndex{d},u,Δx) where {d}
    T = eltype(u)
    s = zero(T)
    for i in 1:d
        s += @inbounds((max(zero(T),u[I+δ(i,I),i])+max(zero(T),-u[I,i])) / Δx[i])
    end
    return s
end
@fastmath @inline function flux_out(I::CartesianIndex{d},u) where {d}
    T = eltype(u)
    s = zero(T)
    for i in 1:d
        s += @inbounds(max(zero(T),u[I+δ(i,I),i])+max(zero(T),-u[I,i]))
    end
    return s
end

"""
    udf!(flow::Flow,udf::Function,t)

User defined function using `udf::Function` to operate on `flow::Flow` during the predictor and corrector step, in sync with time `t`.
Keyword arguments must be passed to `sim_step!` for them to be carried over the actual function call.
"""
udf!(flow,::Nothing,t; kwargs...) = nothing
udf!(flow,force!::Function,t; kwargs...) = force!(flow,t; kwargs...)
