# =============================================================================
# BIOFLOWS UTILITY FUNCTIONS
# =============================================================================
# This module provides core utility functions for the BioFlows CFD solver:
#
# - Index operations: δ, CI, inside, slice
# - GPU/parallel execution: @loop macro with KernelAbstractions backend
# - Boundary conditions: BC!, perBC!, exitBC!
# - Spatial operations: loc (coordinate mapping), interp (interpolation)
# - Logging: @log macro for pressure solver diagnostics
# =============================================================================

using KernelAbstractions: get_backend, @index, @kernel
using LoggingExtras

# =============================================================================
# LOGGING FOR PRESSURE SOLVER DIAGNOSTICS
# =============================================================================

# Custom log level for pressure solver convergence tracking
_psolver = Logging.LogLevel(-123)  # Negative = lower priority than Debug

# Log pressure solver iterations and residuals
macro log(exs...)
    quote
        @logmsg _psolver $(map(x -> esc(x), exs)...)
    end
end
"""
    logger(fname="BioFlows")

Set up a logger to write the pressure solver data to a logging file named `BioFlows.log`.
"""
function logger(fname::String="BioFlows")
    ENV["JULIA_DEBUG"] = all
    logname = endswith(fname, ".log") ? fname : fname * ".log"
    logger = FormatLogger(logname; append=false) do io, args
        args.level == _psolver && print(io, args.message)
    end;
    global_logger(logger);
    # put header in file
    @log "p/c, iter, r∞, r₂\n"
end

# =============================================================================
# CARTESIAN INDEX UTILITIES
# =============================================================================
# These functions manipulate CartesianIndex objects for staggered grid operations.
# The staggered grid has N dimensions but indices are manipulated as tuples.
# =============================================================================

# Shorthand constructor for CartesianIndex
@inline CI(a...) = CartesianIndex(a...)

"""
    CIj(j,I,jj)

Replace jᵗʰ component of CartesianIndex with k.
Useful for implementing periodic boundary conditions.
"""
CIj(j,I::CartesianIndex{d},k) where d = CI(ntuple(i -> i==j ? k : I[i], d))

"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
Used for offsetting indices in a specific direction.

Example: δ(1, CartesianIndex(2,3)) returns CartesianIndex(1,0)
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())

"""
    inside(a)

Return CartesianIndices range excluding a single layer of cells on all boundaries.
"""
@inline inside(a::AbstractArray;buff=1) = CartesianIndices(map(ax->first(ax)+buff:last(ax)-buff,axes(a)))

"""
    inside_u(dims,j)

Return CartesianIndices range excluding the ghost-cells on the boundaries of
a _vector_ array on face `j` with size `dims`.
"""
function inside_u(dims::NTuple{N},j) where {N}
    CartesianIndices(ntuple( i-> i==j ? (3:dims[i]-1) : (2:dims[i]), N))
end
@inline inside_u(dims::NTuple{N}) where N = CartesianIndices((map(i->(2:i-1),dims)...,1:N))
@inline inside_u(u::AbstractArray) = CartesianIndices(map(i->(2:i-1),size(u)[1:end-1]))
splitn(n) = Base.front(n),last(n)
size_u(u) = splitn(size(u))

"""
    L₂(a)

L₂ norm of array `a` excluding ghosts.
"""
L₂(a) = sum(abs2,@inbounds(a[I]) for I ∈ inside(a))

"""
    @inside <arr[I] = value>

Simple macro to automate efficient loops over cells excluding ghosts.
The expression must be an assignment with an indexed array on the left side.

For example

    @inside p[I] = sum(loc(0,I))

becomes

    @loop p[I] = sum(loc(0,I)) over I ∈ inside(p)

See [`@loop`](@ref).
"""
macro inside(ex)
    # Make sure it's a single assignment
    @assert ex.head == :(=) && ex.args[1].head == :(ref)
    a,I = ex.args[1].args[1:2]
    return quote # loop over the size of the reference
        BioFlows.@loop $ex over $I ∈ inside($a)
    end |> esc
end

# Could also use ScopedValues in Julia 1.11+
using Preferences
const backend = @load_preference("backend", "KernelAbstractions")
function set_backend(new_backend::String)
    if !(new_backend in ("SIMD", "KernelAbstractions"))
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("backend" => new_backend)
    @info("New backend set; restart your Julia session for this change to take effect!")
end

"""
    @loop <expr> over <I ∈ R>

Macro to automate fast loops using @simd when running in serial,
or KernelAbstractions when running multi-threaded CPU or GPU.

For example

    @loop a[I,i] += sum(loc(i,I)) over I ∈ R

becomes

    @simd for I ∈ R
        @fastmath @inbounds a[I,i] += sum(loc(i,I))
    end

on serial execution, or

    @kernel function kern(a,i,@Const(I0))
        I ∈ @index(Global,Cartesian)+I0
        @fastmath @inbounds a[I,i] += sum(loc(i,I))
    end
    kern(get_backend(a),64)(a,i,R[1]-oneunit(R[1]),ndrange=size(R))

when multi-threading on CPU or using CuArrays.
Note that `get_backend` is used on the _first_ variable in `expr` (`a` in this example).
"""
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args
    sym = []
    grab!(sym,ex)     # get arguments and replace composites in `ex`
    setdiff!(sym,[I]) # don't want to pass I as an argument
    symT = [gensym() for _ in 1:length(sym)] # generate a list of types for each symbol
    symWtypes = joinsymtype(rep.(sym),symT) # symbols with types: [a::A, b::B, ...]
    @gensym(kern, kern_) # generate unique kernel function names for serial and KA execution
    @static if backend == "KernelAbstractions"
        return quote
            @kernel function $kern_($(symWtypes...),@Const(I0)) where {$(symT...)} # replace composite arguments
                $I = @index(Global,Cartesian)
                $I += I0
                @fastmath @inbounds $ex
            end
            function $kern($(symWtypes...)) where {$(symT...)}
                $kern_(get_backend($(sym[1])),64)($(sym...),$R[1]-oneunit($R[1]),ndrange=size($R))
            end
            $kern($(sym...))
        end |> esc
    else # backend == "SIMD"
        return quote
            function $kern($(symWtypes...)) where {$(symT...)}
                @simd for $I ∈ $R
                    @fastmath @inbounds $ex
                end
            end
            $kern($(sym...))
        end |> esc
    end
end
# Collect local variables (symbols assigned to within the expression)
function collect_locals!(locals::Set{Symbol}, ex::Expr)
    if ex.head == :(=) && ex.args[1] isa Symbol
        push!(locals, ex.args[1])
    end
    foreach(a -> collect_locals!(locals, a), ex.args)
end
collect_locals!(locals::Set{Symbol}, ex) = nothing

# Main grab! entry point: first collect locals, then grab non-local symbols
function grab!(sym, ex)
    locals = Set{Symbol}()
    collect_locals!(locals, ex)
    _grab!(sym, ex, locals)
end

function _grab!(sym, ex::Expr, locals::Set{Symbol})
    ex.head == :. && return union!(sym,[ex])      # grab composite name and return
    if ex.head == :(=) && ex.args[1] isa Symbol
        # Simple assignment: only process RHS
        _grab!(sym, ex.args[2], locals)
        ex.args[2] = rep(ex.args[2])
    else
        start = ex.head==:(call) ? 2 : 1          # don't grab function names
        foreach(a->_grab!(sym, a, locals), ex.args[start:end])   # recurse into args
        ex.args[start:end] = rep.(ex.args[start:end]) # replace composites in args
    end
end
_grab!(sym, ex::Symbol, locals::Set{Symbol}) = ex ∉ locals && union!(sym,[ex])  # grab non-local symbols
_grab!(sym, ex, locals::Set{Symbol}) = nothing
rep(ex) = ex
rep(ex::Expr) = ex.head == :. ? Symbol(ex.args[1], "_", ex.args[2].value) : ex
joinsymtype(sym::Symbol,symT::Symbol) = Expr(:(::), sym, symT)
joinsymtype(sym,symT) = zip(sym,symT) .|> x->joinsymtype(x...)

using StaticArrays
"""
    loc(i,I) = loc(Ii)

Location in space of the cell at CartesianIndex `I` at face `i` in **grid cell units**.
Using `i=0` returns the cell center s.t. `loc ≈ I - 1.5`.

Note: This returns coordinates in grid index space (0, 1, 2, ...), not physical coordinates.
For physical coordinates, use `loc_physical(i, I, Δx)`.
"""
@inline loc(i,I::CartesianIndex{N},T=Float32) where N = SVector{N,T}(I.I .- 1.5 .- 0.5 .* δ(i,I).I)
@inline loc(Ii::CartesianIndex,T=Float32) = loc(last(Ii),Base.front(Ii),T)

"""
    loc_physical(i, I, Δx, T=Float32)

Location in **physical coordinates** of the cell at CartesianIndex `I` at face `i`.
Converts grid indices to physical coordinates using grid spacing Δx.

Physical coordinate: x_physical = (grid_index - 1.5) * Δx

For anisotropic grids, `Δx` should be a tuple/vector of grid spacings.
For isotropic grids, `Δx` can be a scalar.
"""
@inline function loc_physical(i,I::CartesianIndex{N},Δx::NTuple{N},T=Float32) where N
    grid_loc = SVector{N,T}(I.I .- 1.5 .- 0.5 .* δ(i,I).I)
    SVector{N,T}(ntuple(d -> grid_loc[d] * Δx[d], N))
end
@inline loc_physical(i,I::CartesianIndex{N},Δx::Real,T=Float32) where N = loc(i,I,T) .* T(Δx)
# Combined index version: loc_physical(Ii, Δx, T) extracts face index from last dimension
@inline loc_physical(Ii::CartesianIndex,Δx,T=Float32) = loc_physical(last(Ii),Base.front(Ii),Δx,T)
Base.last(I::CartesianIndex) = last(I.I)
Base.front(I::CartesianIndex) = CI(Base.front(I.I))
"""
    apply!(f, c; Δx=nothing)

Apply a vector function `f(i,x)` to the faces of a uniform staggered array `c` or
a function `f(x)` to the center of a uniform array `c`.

If `Δx` is provided (as a tuple of grid spacings), the function receives physical
coordinates (in meters). Otherwise, it receives grid cell coordinates (legacy behavior).

# Examples
```julia
# With physical coordinates (recommended)
apply!(u; Δx=(0.01, 0.01)) do i, x
    i == 1 ? sin(2π * x[1]) : 0.0  # x is in meters
end

# Legacy: grid cell coordinates
apply!(u) do i, x
    i == 1 ? sin(2π * x[1] / N) : 0.0  # x is grid index
end
```
"""
apply!(f,c;Δx=nothing) = hasmethod(f,Tuple{Int,CartesianIndex}) ? applyV!(f,c,Δx) : applyS!(f,c,Δx)
applyV!(f,c,::Nothing) = @loop c[Ii] = f(last(Ii),loc(Ii,eltype(c))) over Ii ∈ CartesianIndices(c)
applyV!(f,c,Δx) = @loop c[Ii] = f(last(Ii),loc_physical(Ii,Δx,eltype(c))) over Ii ∈ CartesianIndices(c)
applyS!(f,c,::Nothing) = @loop c[I] = f(loc(0,I,eltype(c))) over I ∈ CartesianIndices(c)
applyS!(f,c,Δx) = @loop c[I] = f(loc_physical(0,I,Δx,eltype(c))) over I ∈ CartesianIndices(c)
"""
    slice(dims,i,j,low=1)

Return `CartesianIndices` range slicing through an array of size `dims` in
dimension `j` at index `i`. `low` optionally sets the lower extent of the range
in the other dimensions.
"""
function slice(dims::NTuple{N},i,j,low=1) where N
    CartesianIndices(ntuple( k-> k==j ? (i:i) : (low:dims[k]), N))
end

# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================
# These functions apply boundary conditions to the ghost cells of arrays.
# The staggered grid requires different treatment for:
#   - Normal velocity components: Dirichlet (specified value)
#   - Tangential velocity components: Neumann (zero gradient)
#   - Periodic directions: Copy from opposite boundary
# =============================================================================

"""
    BC!(a,A)

Apply boundary conditions to the ghost cells of a _vector_ field. A Dirichlet
condition `a[I,i]=A[i]` is applied to the vector component _normal_ to the domain
boundary. For example `aₓ(x)=Aₓ ∀ x ∈ minmax(X)`. A zero Neumann condition
is applied to the tangential components.

When `Δx` is provided as a keyword argument, coordinates are converted to physical
units using `loc_physical()`. Otherwise, grid cell coordinates are used (legacy behavior).
"""
# Coordinate helper - dispatches based on Δx
@inline _bc_loc(i,I,::Nothing,T) = loc(i,I,T)
@inline _bc_loc(i,I,Δx,T) = loc_physical(i,I,Δx,T)

BC!(a,U,saveoutlet=false,perdir=(),t=0;Δx=nothing) = BC!(a,(i,x,t)->U[i],saveoutlet,perdir,t;Δx)
function BC!(a,inletBC::Function,saveoutlet=false,perdir=(),t=0;Δx=nothing)
    N,n = size_u(a)
    T = eltype(a)
    for i ∈ 1:n, j ∈ 1:n  # i = velocity component, j = boundary direction
        if j in perdir
            # PERIODIC: Copy from opposite boundary
            @loop a[I,i] = a[CIj(j,I,N[j]-1),i] over I ∈ slice(N,1,j)
            @loop a[I,i] = a[CIj(j,I,2),i] over I ∈ slice(N,N[j],j)
        else
            if i==j  # NORMAL component: Dirichlet BC
                for s ∈ (1,2)  # Both ghost layers at inlet
                    @loop a[I,i] = inletBC(i,_bc_loc(i,I,Δx,T),t) over I ∈ slice(N,s,j)
                end
                # Outlet: apply BC unless saveoutlet and x-direction
                (!saveoutlet || i>1) && (@loop a[I,i] = inletBC(i,_bc_loc(i,I,Δx,T),t) over I ∈ slice(N,N[j],j))
            else  # TANGENTIAL component: Neumann BC (zero gradient)
                # u_ghost = u_BC + (u_interior - u_BC) = u_interior
                @loop a[I,i] = inletBC(i,_bc_loc(i,I,Δx,T),t)+a[I+δ(j,I),i]-inletBC(i,_bc_loc(i,I+δ(j,I),Δx,T),t) over I ∈ slice(N,1,j)
                @loop a[I,i] = inletBC(i,_bc_loc(i,I,Δx,T),t)+a[I-δ(j,I),i]-inletBC(i,_bc_loc(i,I-δ(j,I),Δx,T),t) over I ∈ slice(N,N[j],j)
            end
        end
    end
end

"""
    exitBC!(u,u⁰,Δt)

Apply a 1D convection scheme to fill the ghost cell on the outlet of the domain.
"""
function exitBC!(u,u⁰,Δt)
    N,_ = size_u(u)
    exitR = slice(N.-1,N[1],1,2)              # exit slice excluding ghosts
    U = sum(@view(u[slice(N.-1,2,1,2),1]))/length(exitR) # inflow mass flux
    @loop u[I,1] = u⁰[I,1]-U*Δt*(u⁰[I,1]-u⁰[I-δ(1,I),1]) over I ∈ exitR
    ∮u = sum(@view(u[exitR,1]))/length(exitR)-U   # mass flux imbalance
    @loop u[I,1] -= ∮u over I ∈ exitR         # correct flux
end
"""
    perBC!(a,perdir)
Apply periodic conditions to the ghost cells of a _scalar_ field.
"""
perBC!(a,::Tuple{}) = nothing
perBC!(a, perdir, N = size(a)) = for j ∈ perdir
    @loop a[I] = a[CIj(j,I,N[j]-1)] over I ∈ slice(N,1,j)
    @loop a[I] = a[CIj(j,I,2)] over I ∈ slice(N,N[j],j)
end
"""
    interp(x::SVector, arr::AbstractArray)

    Linear interpolation from array `arr` at Cartesian-coordinate `x`.
    Note: This routine works for any number of dimensions.
"""
function interp(x::SVector{D,T}, arr::AbstractArray{T,D}) where {D,T}
    # Index below the interpolation coordinate and the difference
    x = x .+ 1.5f0; i = floor.(Int,x); y = x.-i

    # CartesianIndices around x
    I = CartesianIndex(i...); R = I:I+oneunit(I)

    # Linearly weighted sum over arr[R] (in serial)
    s = zero(T)
    @fastmath @inbounds @simd for J in R
        weight = prod(@. ifelse(J.I==I.I,1-y,y))
        s += arr[J]*weight
    end
    return s
end
using EllipsisNotation
function interp(x::SVector{D,T}, varr::AbstractArray{T}) where {D,T}
    # Shift to align with each staggered grid component and interpolate
    @inline shift(i) = SVector{D,T}(ifelse(i==j,0.5,0.) for j in 1:D)
    return SVector{D,T}(interp(x+shift(i),@view(varr[..,i])) for i in 1:D)
end

"""
    interp_physical(x_phys::SVector, arr::AbstractArray, Δx)

Linear interpolation from array `arr` at **physical coordinate** `x_phys`.
Converts physical coordinates to grid cell coordinates using `Δx`.
"""
function interp_physical(x_phys::SVector{D,T}, arr::AbstractArray{T,D}, Δx) where {D,T}
    # Convert physical coordinates to grid cell coordinates
    x_grid = SVector{D,T}(x_phys[i] / Δx[i] for i in 1:D)
    interp(x_grid, arr)
end
function interp_physical(x_phys::SVector{D,T}, varr::AbstractArray{T}, Δx) where {D,T}
    # Convert physical coordinates to grid cell coordinates
    x_grid = SVector{D,T}(x_phys[i] / Δx[i] for i in 1:D)
    interp(x_grid, varr)
end

"""
    sgs!(flow, t; νₜ, S, Cs, Δ)

Implements a user-defined function `udf` to model subgrid-scale LES stresses based on the Boussinesq approximation
    τᵃᵢⱼ = τʳᵢⱼ - (1/3)τʳₖₖδᵢⱼ = -2νₜS̅ᵢⱼ
where
            ▁▁▁▁
    τʳᵢⱼ =  uᵢuⱼ - u̅ᵢu̅ⱼ

and we add -∂ⱼ(τᵃᵢⱼ) to the RHS as a body force (the isotropic part of the tensor is automatically modelled by the pressure gradient term).
Users need to define the turbulent viscosity function `νₜ` and pass it as a keyword argument to this function together with rate-of-strain
tensor array buffer `S`, Smagorinsky constant `Cs`, and filter width `Δ`.
For example, the standard Smagorinsky–Lilly model for the sub-grid scale stresses is

    νₜ = (CₛΔ)²|S̅ᵢⱼ|=(CₛΔ)²√(2S̅ᵢⱼS̅ᵢⱼ)

It can be implemented as
    `smagorinsky(I::CartesianIndex{m} where m; S, Cs, Δ) = @views (Cs*Δ)^2*sqrt(dot(S[I,:,:],S[I,:,:]))`
and passed into `sim_step!` as a keyword argument together with the varibles than the function needs (`S`, `Cs`, and `Δ`):
    `sim_step!(sim, ...; udf=sgs, νₜ=smagorinsky, S, Cs, Δ)`
"""
function sgs!(flow, t; νₜ, S, Cs, Δ)
    N,n = size_u(flow.u)
    Δx = flow.Δx
    # Compute physical strain rate S for correct turbulent viscosity
    @loop S[I,:,:] .= BioFlows.S(I,flow.u,Δx) over I ∈ inside(flow.σ)
    for i ∈ 1:n, j ∈ 1:n
        inv_Δxj = inv(Δx[j])  # Physical gradient scaling
        BioFlows.@loop (
            # SGS stress divergence: ∂τ_ij/∂x_j with physical gradient
            # νₜ computed from physical S, gradient scaled by 1/Δx_j
            flow.σ[I] = -νₜ(I;S,Cs,Δ)*∂(j,CI(I,i),flow.u)*inv_Δxj;
            flow.f[I,i] += flow.σ[I];
        ) over I ∈ inside_u(N,j)
        BioFlows.@loop flow.f[I-δ(j,I),i] -= flow.σ[I] over I ∈ BioFlows.inside_u(N,j)
    end
end

check_fn(f,N,T,nargs) = nothing
function check_fn(f::Function,N,T,nargs)
    @assert first(methods(f)).nargs==nargs+1 "$f signature needs $nargs arguments"
    @assert all(typeof.(ntuple(i->f(i,xtargs(Val{}(nargs),N,T)...),N)).==T) "$f is not type stable"
end
xtargs(::Val{2},N,T) = (zeros(SVector{N,T}),)
xtargs(::Val{3},N,T) = (zeros(SVector{N,T}),zero(T))

ic_function(inletBC::Function) = (i,x)->inletBC(i,x,0)
ic_function(inletBC::Tuple) = (i,x)->inletBC[i]

squeeze(a::AbstractArray) = dropdims(a, dims = tuple(findall(size(a) .== 1)...))
