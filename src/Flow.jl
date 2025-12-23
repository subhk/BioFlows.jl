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

Compute convective and diffusive fluxes for the momentum equation.

The dimensional form is:
    r = -∇·(u⊗u) + ν∇²u

With proper scaling for anisotropic grids:
- Convective flux in direction j: (u·∇)u scaled by 1/Δx[j]
- Diffusive flux in direction j: ν∂²u/∂xⱼ² scaled by ν/Δx[j]²

# Arguments
- `r`: RHS accumulator (output)
- `u`: Velocity field
- `Φ`: Flux work array
- `λ`: Convection scheme (quick, vanLeer, cds)
- `ν`: Kinematic viscosity (m²/s)
- `Δx`: Grid spacing tuple (m), e.g., `(Δx, Δy)` or `(Δx, Δy, Δz)`
- `perdir`: Tuple of periodic directions
"""
function conv_diff!(r,u,Φ,λ::F;ν=0.1,Δx=(1,1),perdir=()) where {F}
    r .= zero(eltype(r))
    N,n = size_u(u)
    T = eltype(r)
    for i ∈ 1:n, j ∈ 1:n
        # Direction-specific scaling for anisotropic grids
        inv_Δxj = T(1/Δx[j])        # For convective term in direction j
        ν_over_Δxj = T(ν/Δx[j])     # For diffusive term in direction j
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
lowerBoundary!(r,u,Φ,ν_Δx,inv_Δx,i,j,N,λ,::Val{false}) = @loop r[I,i] += inv_Δx*ϕuL(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ) - ν_Δx*∂(j,CI(I,i),u) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,Φ,ν_Δx,inv_Δx,i,j,N,λ,::Val{false}) = @loop r[I-δ(j,I),i] += -inv_Δx*ϕuR(j,CI(I,i),u,ϕ(i,CI(I,j),u),λ) + ν_Δx*∂(j,CI(I,i),u) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block (dimensional form)
lowerBoundary!(r,u,Φ,ν_Δx,inv_Δx,i,j,N,λ,::Val{true}) = @loop (
    Φ[I] = inv_Δx*ϕuP(j,CIj(j,CI(I,i),N[j]-2),CI(I,i),u,ϕ(i,CI(I,j),u),λ) - ν_Δx*∂(j,CI(I,i),u); r[I,i] += Φ[I]) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,Φ,ν_Δx,inv_Δx,i,j,N,λ,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)

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
    p :: Sf # pressure scalar field (m²/s²)
    σ :: Sf # divergence scalar (work array)
    # BDIM fields
    V :: Vf # body velocity vector (m/s)
    μ₀:: Vf # zeroth-moment vector (dimensionless)
    μ₁:: Tf # first-moment tensor field (dimensionless)
    # Non-fields
    inletBC :: Union{NTuple{D,Number},Function} # inlet boundary velocity (m/s)
    Δt:: Vector{T} # time step history (s)
    ν :: T # kinematic viscosity (m²/s)
    Δx :: NTuple{D,T} # grid spacing per direction (m) - can be anisotropic
    g :: Union{Function,Nothing} # acceleration field (m/s²)
    outletBC :: Bool # convective outlet BC flag
    perdir :: NTuple # periodic directions tuple
    """
        Flow(N, inletBC; L, ν=0, Δt=0.25, ...)

    Construct a Flow on grid of size `N` with domain size `L`.

    # Required Arguments
    - `N::NTuple{D}`: Number of grid cells, e.g., `(nx, nz)` or `(nx, ny, nz)`
    - `inletBC`: Inlet boundary velocity (m/s). Tuple or `Function(i,x,t)`
    - `L::NTuple{D}`: Physical domain size (m), e.g., `(2.0, 1.0)` for 2m × 1m
      Grid spacing: `Δx[d] = L[d]/N[d]` for each direction d
      Supports anisotropic grids (Δx ≠ Δy ≠ Δz)

    # Optional Arguments
    - `ν=0.`: Kinematic viscosity (m²/s). Water ≈ 1e-6, air ≈ 1.5e-5
    - `Δt=0.25`: Initial time step (s)
    - `g=nothing`: Body acceleration function `g(i,x,t)` returning m/s²
    - `uλ=nothing`: Initial velocity. Tuple or `Function(i,x)`
    - `perdir=()`: Periodic directions, e.g., `(2,)` for y-periodic
    - `outletBC=false`: Convective outlet BC in direction 1
    - `T=Float32`: Numeric type
    - `f=Array`: Memory backend

    # Example
    ```julia
    # 2m × 1m domain, 200×100 cells → Δx = 0.01m, Δz = 0.01m (uniform)
    flow = Flow((200, 100), (1.0, 0.0); L=(2.0, 1.0), ν=1e-6)

    # Anisotropic: 4m × 1m domain, 200×100 cells → Δx = 0.02m, Δz = 0.01m
    flow = Flow((200, 100), (1.0, 0.0); L=(4.0, 1.0), ν=1e-6)
    ```
    """
    function Flow(N::NTuple{D}, inletBC; L::NTuple{D}, f=Array, Δt=0.25, ν=0., g=nothing,
            uλ=nothing, perdir=(), outletBC=false, T=Float32) where D
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
        new{D,T,typeof(p),typeof(u),typeof(μ₁)}(u,u⁰,fv,p,σ,V,μ₀,μ₁,inletBC,T[Δt],T(ν),Δx,g,outletBC,perdir)
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

Note: The Poisson solver uses unit spacing internally (Δx=1 in computational coordinates).
For isotropic grids (Δx = Δy = Δz), this works correctly because all operators use
the same implicit scaling. For anisotropic grids, uniform grid spacing is required
in each direction (Δx can differ from Δy, but must be uniform within each direction).
"""
function project!(a::Flow{n},b::AbstractPoisson,w=1) where n
    dt = w*a.Δt[end]
    @inside b.z[I] = div(I,a.u)
    b.x .*= dt  # Scale initial guess for warm start
    solver!(b)
    for i ∈ 1:n
        @loop a.u[I,i] -= b.L[I,i]*∂(i,I,b.x) over I ∈ inside(b.x)
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
@fastmath function mom_step!(a::Flow{N},b::AbstractPoisson;λ=quick,udf=nothing,kwargs...) where N
    a.u⁰ .= a.u; scale_u!(a,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
    # predictor u → u'
    @log "p"
    conv_diff!(a.f,a.u⁰,a.σ,λ;ν=a.ν,Δx=a.Δx,perdir=a.perdir)
    udf!(a,udf,t₀; kwargs...)
    accelerate!(a.f,t₀,a.g,a.inletBC)
    BDIM!(a); BC!(a.u,a.inletBC,a.outletBC,a.perdir,t₁) # BC MUST be at t₁
    a.outletBC && exitBC!(a.u,a.u⁰,a.Δt[end]) # convective outlet
    project!(a,b); BC!(a.u,a.inletBC,a.outletBC,a.perdir,t₁)
    # corrector u → u¹
    @log "c"
    conv_diff!(a.f,a.u,a.σ,λ;ν=a.ν,Δx=a.Δx,perdir=a.perdir)
    udf!(a,udf,t₁; kwargs...)
    accelerate!(a.f,t₁,a.g,a.inletBC)
    BDIM!(a); scale_u!(a,0.5); BC!(a.u,a.inletBC,a.outletBC,a.perdir,t₁)
    project!(a,b,0.5); BC!(a.u,a.inletBC,a.outletBC,a.perdir,t₁)
    push!(a.Δt,CFL(a))
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
    s = zero(eltype(u))
    for i in 1:d
        s += @inbounds((max(0.,u[I+δ(i,I),i])+max(0.,-u[I,i])) / Δx[i])
    end
    return s
end
@fastmath @inline function flux_out(I::CartesianIndex{d},u) where {d}
    s = zero(eltype(u))
    for i in 1:d
        s += @inbounds(max(0.,u[I+δ(i,I),i])+max(0.,-u[I,i]))
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
