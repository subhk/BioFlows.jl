"""
BioFlows.jl - A Julia package for computational fluid dynamics simulations
with immersed boundary methods based on the WaterLily.jl methodology.
"""
module BioFlows

using DocStringExtensions
using LinearAlgebra
using Statistics
using Random
using Logging
using ForwardDiff
using MPI

const _suppress_warnings = get(ENV, "BIOFLOWS_SUPPRESS_WARNINGS", "true") in ("1","true","TRUE")
const _silent_logger = Logging.SimpleLogger(stderr, Logging.Error)

_silent_include(path::AbstractString) = _suppress_warnings ?
    Logging.with_logger(_silent_logger) do
        include(path)
    end :
    include(path)

const _has_pencilarrays = let
    try
        @eval begin
            _suppress_warnings ? Logging.with_logger(_silent_logger) do
                using PencilArrays
            end : using PencilArrays
        end
        true
    catch err
        _suppress_warnings || @warn "PencilArrays not available; distributed helpers will use no-op stubs." exception=(err, catch_backtrace())
        @eval module PencilArrays
            module Pencils
                struct MPITopology end
                MPITopology(args...; kwargs...) = MPITopology()
                struct Pencil end
                Pencil(args...; kwargs...) = Pencil()
            end
            function exchange_halo!(field, args...; kwargs...)
                field
            end
            exchange_halo!(args...; kwargs...) = nothing
        end
        false
    end
end

# Core utilities and macros
_silent_include("util.jl")
export L₂,BC!,@inside,inside,δ,apply!,loc,@log,set_backend,backend

using Reexport
@reexport using KernelAbstractions: @kernel,@index,get_backend

# Pressure solver
_silent_include("Poisson.jl")
export AbstractPoisson,Poisson,solver!,mult!

# Multigrid pressure solver
_silent_include("MultiLevelPoisson.jl")
export MultiLevelPoisson

# Flow solver
_silent_include("Flow.jl")
export Flow,mom_step!,quick,cds

# Body definitions
_silent_include("Body.jl")
export AbstractBody,measure_sdf!

# Auto body (implicit geometry)
_silent_include("AutoBody.jl")
export AutoBody,Bodies,measure,sdf,+,-

# Metrics and diagnostics
_silent_include("Metrics.jl")
export MeanFlow,update!,uu!,uu

# AMR (Adaptive Mesh Refinement) functionality
_silent_include("amr/amr_types.jl")
_silent_include("amr/bioflows_amr_adapter.jl")
_silent_include("amr/body_refinement_indicator.jl")
export StaggeredGrid, SolutionState, RefinedGrid, GridType, TwoDimensional, ThreeDimensional
export FlowToGridAdapter, flow_to_staggered_grid, flow_to_solution_state
export compute_body_refinement_indicator, compute_velocity_gradient_indicator
export compute_vorticity_indicator, compute_combined_indicator

# Simulation container
abstract type AbstractSimulation end
"""
    Simulation(dims::NTuple, uBC::Union{NTuple,Function}, L::Number;
               U=norm2(Uλ), Δt=0.25, ν=0., ϵ=1, g=nothing,
               perdir=(), exitBC=false,
               body::AbstractBody=NoBody(),
               T=Float32, mem=Array)

Constructor for a BioFlows simulation:

  - `dims`: Simulation domain dimensions.
  - `uBC`: Velocity field applied to boundary and acceleration conditions.
        Define a `Tuple` for constant BCs, or a `Function` for space and time varying BCs `uBC(i,x,t)`.
  - `L`: Simulation length scale.
  - `U`: Simulation velocity scale. Required if using `Uλ::Function`.
  - `Δt`: Initial time step.
  - `ν`: Scaled viscosity (`Re=UL/ν`).
  - `g`: Domain acceleration, `g(i,x,t)=duᵢ/dt`
  - `ϵ`: BDIM kernel width.
  - `perdir`: Domain periodic boundary condition in the `(i,)` direction.
  - `uλ`: Velocity field applied to the initial condition.
        Define a Tuple for homogeneous (per direction) IC, or a `Function` for space varying IC `uλ(i,x)`.
  - `exitBC`: Convective exit boundary condition in the `i=1` direction.
  - `body`: Immersed geometry.
  - `T`: Array element type.
  - `mem`: memory location. `Array`, `CuArray`, `ROCm` to run on CPU, NVIDIA, or AMD devices, respectively.

See files in `examples` folder for examples.
"""
mutable struct Simulation <: AbstractSimulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: Flow
    body :: AbstractBody
    pois :: AbstractPoisson
    function Simulation(dims::NTuple{N}, uBC, L::Number;
                        Δt=0.25, ν=0., g=nothing, U=nothing, ϵ=1, perdir=(),
                        uλ=nothing, exitBC=false, body::AbstractBody=NoBody(),
                        T=Float32, mem=Array) where N
        @assert !(isnothing(U) && isa(uBC,Function)) "`U` (velocity scale) must be specified if boundary conditions `uBC` is a `Function`"
        isnothing(U) && (U = √sum(abs2,uBC))
        check_fn(uBC,N,T,3); check_fn(g,N,T,3); check_fn(uλ,N,T,2)
        flow = Flow(dims,uBC;uλ,Δt,ν,g,T,f=mem,perdir,exitBC)
        measure!(flow,body;ϵ)
        new(U,L,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir))
    end
end

time(sim::AbstractSimulation) = time(sim.flow)
"""
    sim_time(sim::Simulation)

Return the current dimensionless time of the simulation `tU/L`
where `t=sum(Δt)`, and `U`,`L` are the simulation velocity and length
scales.
"""
sim_time(sim::AbstractSimulation) = time(sim)*sim.U/sim.L

"""
    sim_step!(sim::AbstractSimulation,t_end;remeasure=true,λ=quick,max_steps=typemax(Int),verbose=false,
        udf=nothing,kwargs...)

Integrate the simulation `sim` up to dimensionless time `t_end`.
If `remeasure=true`, the body is remeasured at every time step. Can be set to `false` for static geometries to speed up simulation.
A user-defined function `udf` can be passed to arbitrarily modify the `::Flow` during the predictor and corrector steps.
If the `udf` user keyword arguments, these needs to be included in the `sim_step!` call as well.
A `λ::Function` function can be passed as a custom convective scheme, following the interface of `λ(u,c,d)` (for upstream, central,
downstream points).
"""
function sim_step!(sim::AbstractSimulation,t_end;remeasure=true,λ=quick,max_steps=typemax(Int),verbose=false,
        udf=nothing,kwargs...)
    steps₀ = length(sim.flow.Δt)
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        sim_step!(sim; remeasure, λ, udf, kwargs...)
        verbose && sim_info(sim)
    end
end
function sim_step!(sim::AbstractSimulation;remeasure=true,λ=quick,udf=nothing,kwargs...)
    remeasure && measure!(sim)
    mom_step!(sim.flow, sim.pois; λ, udf, kwargs...)
end

"""
    measure!(sim::Simulation,t=timeNext(sim))

Measure a dynamic `body` to update the `flow` and `pois` coefficients.
"""
function measure!(sim::AbstractSimulation,t=sum(sim.flow.Δt))
    measure!(sim.flow,sim.body;t,ϵ=sim.ϵ)
    update!(sim.pois)
end

"""
    sim_info(sim::AbstractSimulation)
Prints information on the current state of a simulation.
"""
sim_info(sim::AbstractSimulation) = println("tU/L=",round(sim_time(sim),digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))

"""
    perturb!(sim; noise=0.1)
Perturb the velocity field of a simulation with `noise` level with respect to velocity scale `U`.
"""
perturb!(sim::AbstractSimulation; noise=0.1) = sim.flow.u .+= randn(size(sim.flow.u))*sim.U*noise |> typeof(sim.flow.u).name.wrapper

export AbstractSimulation,Simulation,sim_step!,sim_time,measure!,sim_info,perturb!

# defaults JLD2 and VTK I/O functions
function load!(sim::AbstractSimulation; kwargs...)
    fname = get(Dict(kwargs), :fname, "BioFlows.jld2")
    ext = split(fname, ".")[end] |> Symbol
    vtk_loaded = !isnothing(Base.get_extension(BioFlows, :BioFlowsReadVTKExt))
    jld2_loaded = !isnothing(Base.get_extension(BioFlows, :BioFlowsJLD2Ext))
    ext == :pvd && (@assert vtk_loaded "WriteVTK must be loaded to save .pvd data.")
    ext == :jdl2 && (@assert jld2_loaded "JLD2 must be loaded to save .jld2 data.")
    load!(sim, Val{ext}(); kwargs...)
end
function save! end
function vtkWriter end
function default_attrib end
function pvd_collection end
export load!, save!, vtkWriter, default_attrib

# default Plots functions
function flood end
function addbody end
function body_plot! end
function sim_gif! end
function plot_logger end
export flood,addbody,body_plot!,sim_gif!,plot_logger

# default Makie functions
function viz! end
function get_body end
function plot_body_obs! end
export viz!, get_body, plot_body_obs!

# Check number of threads when loading BioFlows
"""
    check_nthreads()

Check the number of threads available for the Julia session that loads BioFlows.
A warning is shown when running in serial (JULIA_NUM_THREADS=1) with KernelAbstractions enabled.
"""
function check_nthreads()
    if backend == "KernelAbstractions" && Threads.nthreads() == 1
        @warn """
        Using BioFlows in serial (ie. JULIA_NUM_THREADS=1) is not recommended because it defaults to serial CPU execution.
        Use JULIA_NUM_THREADS=auto, or any number of threads greater than 1, to allow multi-threading in CPU backends.
        For a low-overhead single-threaded CPU only backend set: BioFlows.set_backend("SIMD")
        """
    end
end
check_nthreads()

# BioFlows-specific extensions
_silent_include("Diagnostics.jl")
_silent_include("Output.jl")

# Export diagnostics functions
export pressure_force, viscous_force, total_force, curl, ω, ω_mag
export vorticity_component, vorticity_magnitude
export cell_center_velocity, cell_center_vorticity
export CenterFieldWriter, maybe_save!
export force_components, force_coefficients, record_force!

end # module
