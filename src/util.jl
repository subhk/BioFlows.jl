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

using KernelAbstractions: get_backend, @index, @kernel, CPU
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

# =============================================================================
# GPU-SAFE REDUCTION HELPERS
# =============================================================================
# CartesianIndices views on CuArray can cause scalar indexing with
# CUDA.allowscalar(false). These helpers transfer small boundary slices
# to CPU for reduction, which is safe and efficient for boundary operations.
# =============================================================================

"""
    _safe_sum(v)

GPU-safe sum that handles CartesianIndices views by transferring to CPU.
For boundary slices, the transfer overhead is negligible compared to the
GPU kernel launch overhead for small reductions.
"""
_safe_sum(v::AbstractArray) = sum(Array(v))
_safe_sum(v::Array) = sum(v)  # Already on CPU, no transfer needed

"""
    _safe_maximum(f, v)

GPU-safe maximum with function `f` applied element-wise.
Transfers to CPU to avoid scalar indexing issues with CartesianIndices views.
"""
_safe_maximum(f, v::AbstractArray) = maximum(f, Array(v))
_safe_maximum(f, v::Array) = maximum(f, v)  # Already on CPU

"""
    _safe_sum_abs2(v)

GPU-safe sum of squared absolute values.
"""
_safe_sum_abs2(v::AbstractArray) = sum(abs2, Array(v))
_safe_sum_abs2(v::Array) = sum(abs2, v)

"""
    L₂(a)

L₂ norm of array `a` excluding ghosts.
Uses GPU-safe reduction that transfers the interior view to CPU.
"""
L₂(a) = _safe_sum_abs2(@view a[inside(a)])

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
const cpu_threads = @load_preference("cpu_threads", true)  # Enable CPU multithreading by default

function set_backend(new_backend::String)
    if !(new_backend in ("SIMD", "KernelAbstractions"))
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end
    @set_preferences!("backend" => new_backend)
    @info("New backend set; restart your Julia session for this change to take effect!")
end

"""
    set_cpu_threads(enabled::Bool)

Enable or disable CPU multithreading for KernelAbstractions backend.
When enabled, CPU loops use all available Julia threads.

Start Julia with multiple threads for this to have effect:
    julia -t auto          # Use all available cores
    julia -t 4             # Use 4 threads
    julia --threads=auto   # Alternative syntax

# Example
```julia
using BioFlows
BioFlows.set_cpu_threads(true)   # Enable (default)
BioFlows.set_cpu_threads(false)  # Disable for debugging
# Restart Julia for changes to take effect
```
"""
function set_cpu_threads(enabled::Bool)
    @set_preferences!("cpu_threads" => enabled)
    @info("CPU threading $(enabled ? "enabled" : "disabled"); restart Julia for this change to take effect!")
end

# Create CPU backend with threading support
# KernelAbstractions.CPU() automatically uses Julia threads if available
const _cpu_backend = if cpu_threads && Threads.nthreads() > 1
    @info "BioFlows: CPU multithreading enabled with $(Threads.nthreads()) threads"
    CPU()
else
    if cpu_threads && Threads.nthreads() == 1
        @warn "BioFlows: CPU multithreading requested but Julia started with 1 thread. " *
              "Start Julia with `julia -t auto` or `julia -t N` for multithreading."
    end
    CPU()
end

"""
    _get_backend(arr)

Get the appropriate KernelAbstractions backend for the given array.
- For CPU Arrays: returns the configured CPU backend (with threading if enabled)
- For GPU Arrays (CuArray): returns the GPU backend via KernelAbstractions.get_backend
"""
@inline _get_backend(arr::Array) = _cpu_backend
@inline _get_backend(arr) = get_backend(arr)  # GPU arrays use their native backend

"""
    @loop <expr> over <I ∈ R>

Macro to automate fast loops using @simd when running in serial,
or KernelAbstractions when running multi-threaded CPU or GPU.

## Backend Selection

The backend is determined at **compile time** by the `backend` preference:
- `"KernelAbstractions"` (default): Uses KernelAbstractions.jl for GPU/parallel execution
- `"SIMD"`: Uses @simd loops for serial CPU execution

To change the backend:
```julia
using BioFlows
BioFlows.set_backend("KernelAbstractions")  # or "SIMD"
# Restart Julia for changes to take effect
```

**Important:** The backend must match your array type. Using `mem=CuArray` with
`backend="SIMD"` will cause scalar indexing errors. The `Simulation` constructor
validates this automatically.

## CPU Multithreading

When using `backend="KernelAbstractions"` with CPU arrays, loops automatically use
all available Julia threads. To enable multithreading:

1. Start Julia with multiple threads:
   ```bash
   julia -t auto          # Use all available cores
   julia -t 4             # Use 4 threads
   ```

2. CPU threading is enabled by default. To disable:
   ```julia
   BioFlows.set_cpu_threads(false)
   # Restart Julia
   ```

## Example

    @loop a[I,i] += sum(loc(i,I)) over I ∈ R

becomes

    @simd for I ∈ R
        @fastmath @inbounds a[I,i] += sum(loc(i,I))
    end

on serial execution (backend="SIMD"), or

    @kernel function kern(a,i,@Const(I0))
        I ∈ @index(Global,Cartesian)+I0
        @fastmath @inbounds a[I,i] += sum(loc(i,I))
    end
    kern(_get_backend(a),64)(a,i,R[1]-oneunit(R[1]),ndrange=size(R))

when using KernelAbstractions (backend="KernelAbstractions").
Note that `_get_backend` is used on the _first_ variable in `expr` (`a` in this example),
which returns a multithreaded CPU backend for Arrays or the native GPU backend for CuArrays.
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
                # Use _get_backend for CPU multithreading support
                event = $kern_(_get_backend($(sym[1])),64)($(sym...),$R[1]-oneunit($R[1]),ndrange=size($R))
                event !== nothing && wait(event)
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
function grab!(sym,ex::Expr)
    ex.head == :. && return union!(sym,[ex])      # grab composite name and return
    start = ex.head==:(call) ? 2 : 1              # don't grab function names
    foreach(a->grab!(sym,a),ex.args[start:end])   # recurse into args
    ex.args[start:end] = rep.(ex.args[start:end]) # replace composites in args
end
grab!(sym,ex::Symbol) = union!(sym,[ex])          # grab symbol name
grab!(sym,ex) = nothing
rep(ex) = ex
rep(ex::Expr) = ex.head == :. ? Symbol(ex.args[2].value) : ex
joinsymtype(sym::Symbol,symT::Symbol) = Expr(:(::), sym, symT)
joinsymtype(sym,symT) = zip(sym,symT) .|> x->joinsymtype(x...)

using StaticArrays
"""
    loc(i,I) = loc(Ii)

Location in space of the cell at CartesianIndex `I` at face `i`.
Using `i=0` returns the cell center s.t. `loc = I`.
"""
@inline loc(i,I::CartesianIndex{N},T=Float32) where N = SVector{N,T}(T.(I.I) .- T(1.5) .- T(0.5) .* T.(δ(i,I).I))
@inline loc(Ii::CartesianIndex,T=Float32) = loc(last(Ii),Base.front(Ii),T)
Base.last(I::CartesianIndex) = last(I.I)
Base.front(I::CartesianIndex) = CI(Base.front(I.I))
"""
    apply!(f, c)

Apply a vector function `f(i,x)` to the faces of a uniform staggered array `c` or
a function `f(x)` to the center of a uniform array `c`.
"""
apply!(f,c) = hasmethod(f,Tuple{Int,CartesianIndex}) ? applyV!(f,c) : applyS!(f,c)
applyV!(f,c) = @loop c[Ii] = f(last(Ii),loc(Ii,eltype(c))) over Ii ∈ CartesianIndices(c)
applyS!(f,c) = @loop c[I] = f(loc(0,I,eltype(c))) over I ∈ CartesianIndices(c)
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
"""
BC!(a,U,saveoutlet=false,perdir=(),t=0) = BC!(a,(i,x,t)->U[i],saveoutlet,perdir,t)
function BC!(a,inletBC::Function,saveoutlet=false,perdir=(),t=0)
    N,n = size_u(a)
    for i ∈ 1:n, j ∈ 1:n  # i = velocity component, j = boundary direction
        if j in perdir
            # PERIODIC: Copy from opposite boundary
            @loop a[I,i] = a[CIj(j,I,N[j]-1),i] over I ∈ slice(N,1,j)
            @loop a[I,i] = a[CIj(j,I,2),i] over I ∈ slice(N,N[j],j)
        else
            if i==j  # NORMAL component: Dirichlet BC
                for s ∈ (1,2)  # Both ghost layers at inlet
                    @loop a[I,i] = inletBC(i,loc(i,I),t) over I ∈ slice(N,s,j)
                end
                # Outlet: apply BC unless saveoutlet and x-direction
                (!saveoutlet || i>1) && (@loop a[I,i] = inletBC(i,loc(i,I),t) over I ∈ slice(N,N[j],j))
            else  # TANGENTIAL component: Neumann BC (zero gradient)
                # u_ghost = u_BC + (u_interior - u_BC) = u_interior
                @loop a[I,i] = inletBC(i,loc(i,I),t)+a[I+δ(j,I),i]-inletBC(i,loc(i,I+δ(j,I)),t) over I ∈ slice(N,1,j)
                @loop a[I,i] = inletBC(i,loc(i,I),t)+a[I-δ(j,I),i]-inletBC(i,loc(i,I-δ(j,I)),t) over I ∈ slice(N,N[j],j)
            end
        end
    end
end

"""
    exitBC!(u,u⁰,Δt)

Apply a 1D convection scheme to fill the ghost cell on the outlet of the domain.
Uses GPU-safe reductions that transfer boundary slices to CPU to avoid
scalar indexing issues with CartesianIndices views on CuArray.
"""
function exitBC!(u,u⁰,Δt)
    N,_ = size_u(u)
    exitR = slice(N.-1,N[1],1,2)              # exit slice excluding ghosts
    # GPU-safe sum: transfer boundary slice to CPU for reduction
    inletV = @view(u[slice(N.-1,2,1,2),1])
    U = _safe_sum(inletV)/length(exitR)       # inflow mass flux
    @loop u[I,1] = u⁰[I,1]-U*Δt*(u⁰[I,1]-u⁰[I-δ(1,I),1]) over I ∈ exitR
    exitV = @view(u[exitR,1])
    ∮u = _safe_sum(exitV)/length(exitR)-U     # mass flux imbalance
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

Warning: This function uses scalar indexing and is intended for CPU arrays.
On GPU arrays (CuArray), this will cause scalar indexing which is extremely
slow or may error with `CUDA.allowscalar(false)`. For GPU usage, transfer
the array to CPU first: `interp(x, Array(arr))`.
"""
function interp(x::SVector{D,T}, arr::AbstractArray{T,D}) where {D,T}
    # Index below the interpolation coordinate and the difference
    x = x .+ T(1.5); i = floor.(Int,x); y = x.-i

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
    @inline shift(i) = SVector{D,T}(ifelse(i==j,T(0.5),zero(T)) for j in 1:D)
    return SVector{D,T}(interp(x+shift(i),@view(varr[..,i])) for i in 1:D)
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
    @loop S[I,:,:] .= BioFlows.S(I,flow.u) over I ∈ inside(flow.σ)
    for i ∈ 1:n, j ∈ 1:n
        BioFlows.@loop (
            flow.σ[I] = -νₜ(I;S,Cs,Δ)*∂(j,CI(I,i),flow.u);
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
