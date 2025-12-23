"""
BioFlows.jl - A Julia package for computational fluid dynamics simulations
with immersed boundary methods using the Boundary Data Immersion Method (BDIM).
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
export AutoBody,measure,sdf,+,-

# Metrics and diagnostics
_silent_include("Metrics.jl")
export MeanFlow,update!,uu!,uu

# AMR (Adaptive Mesh Refinement) functionality
_silent_include("amr/amr_types.jl")
_silent_include("amr/bioflows_amr_adapter.jl")
_silent_include("amr/body_refinement_indicator.jl")
export StaggeredGrid, SolutionState, RefinedGrid, GridType, TwoDimensional, ThreeDimensional
export is_2d, is_3d, num_refined_cells, refinement_level, domain_size, cell_volume
export FlowToGridAdapter, flow_to_staggered_grid, flow_to_solution_state, create_refined_grid
export compute_body_refinement_indicator, compute_velocity_gradient_indicator
export compute_vorticity_indicator, compute_combined_indicator
export mark_cells_for_refinement, apply_buffer_zone!

# AMR Composite Grid Poisson Solver
_silent_include("amr/refined_fields.jl")
_silent_include("amr/patch_poisson.jl")
_silent_include("amr/interface_operators.jl")
_silent_include("amr/composite_poisson.jl")
_silent_include("amr/composite_solver.jl")
_silent_include("amr/patch_creation.jl")
_silent_include("amr/amr_project.jl")
export CompositePoisson, PatchPoisson, RefinedVelocityField, RefinedVelocityPatch
export add_patch!, remove_patch!, get_patch, clear_patches!, has_patches, num_patches
export create_patches!, update_patches!, ensure_proper_nesting!
export amr_project!, amr_mom_step!, check_amr_divergence, regrid_amr!
export amr_cfl, synchronize_base_and_patches!, interpolate_velocity_to_patches!

# Simulation container
abstract type AbstractSimulation end
"""
    Simulation(dims::NTuple{N}, uBC, L::NTuple{N}; L_char=nothing, kwargs...)

Constructor for a BioFlows simulation solving the dimensional incompressible Navier-Stokes equations:

    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + g
    ∇·u = 0

# Arguments

## Required
- `dims::NTuple{N,Int}`: Number of grid cells in each direction, e.g., `(nx, nz)` or `(nx, ny, nz)`
- `uBC`: Boundary velocity. Either a `Tuple` for constant BCs, or `Function(i,x,t)` for space/time varying
- `L::NTuple{N}`: Physical domain size in each direction (e.g., `(Lx, Lz)` in meters)
  - Grid spacing is computed as `Δx = L[1]/dims[1]` (must be uniform)

## Optional (keyword arguments)
- `L_char`: Characteristic length scale for dimensionless time and force coefficients
  - Defaults to `L[1]` if not specified
  - For cylinder flows, typically use the diameter (2*radius)
- `U`: Velocity scale for dimensionless time reporting. Required if `uBC` is a `Function`
- `Δt=0.25`: Initial time step (seconds)
- `ν=0.`: Kinematic viscosity (m²/s)
- `g=nothing`: Body acceleration function `g(i,x,t)` (m/s²)
- `ϵ=1`: BDIM kernel width (in grid cells)
- `perdir=()`: Periodic directions, e.g., `(2,)` for y-periodic
- `uλ=nothing`: Initial velocity condition. Tuple or `Function(i,x)`
- `exitBC=false`: Enable convective exit BC in direction 1
- `body=NoBody()`: Immersed body geometry
- `T=Float32`: Numeric type
- `mem=Array`: Memory backend (`Array`, `CuArray`, etc.)

# Example
```julia
# 2D channel: 2m × 1m domain with 256 × 128 cells
# Δx = 2.0/256 = 0.0078125 m
# Inlet velocity 1 m/s, water viscosity
sim = Simulation((256, 128), (1.0, 0.0), (2.0, 1.0); ν=1e-6)

# With immersed cylinder of diameter 0.2m
diameter = 0.2
cylinder = AutoBody((x,t) -> √(x[1]^2 + x[2]^2) - diameter/2)
sim = Simulation((256, 128), (1.0, 0.0), (2.0, 1.0); ν=1e-6, body=cylinder, L_char=diameter)
```

See files in `examples` folder for more examples.
"""
mutable struct Simulation <: AbstractSimulation
    U :: Number # velocity scale (for dimensionless time)
    L :: Number # characteristic length scale (for dimensionless time/forces)
    ϵ :: Number # kernel width
    flow :: Flow
    body :: AbstractBody
    pois :: AbstractPoisson

    function Simulation(dims::NTuple{N}, uBC, L::NTuple{N};
                        L_char=nothing, Δt=0.25, ν=0., g=nothing, U=nothing, ϵ=1, perdir=(),
                        uλ=nothing, exitBC=false, body::AbstractBody=NoBody(),
                        T=Float32, mem=Array) where N
        @assert !(isnothing(U) && isa(uBC,Function)) "`U` (velocity scale) must be specified if boundary conditions `uBC` is a `Function`"
        isnothing(U) && (U = √sum(abs2,uBC))
        check_fn(uBC,N,T,3); check_fn(g,N,T,3); check_fn(uλ,N,T,2)
        # Pass domain size L to Flow for dimensional Δx computation
        flow = Flow(dims,uBC;L=L,uλ,Δt,ν,g,T,f=mem,perdir,exitBC)
        measure!(flow,body;ϵ)
        # Use L_char for dimensionless time/forces, default to L[1]
        char_length = isnothing(L_char) ? L[1] : L_char
        new(U,char_length,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir))
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

# ============================================================================
# AMR-enabled Simulation
# ============================================================================

"""
    AMRConfig

Configuration for adaptive mesh refinement.

# Fields
- `max_level`: Maximum refinement level (1 = 2x finer, 2 = 4x finer, etc.)
- `body_distance_threshold`: Refine within this distance (in grid cells) from body
- `velocity_gradient_threshold`: Threshold for velocity gradient refinement
- `vorticity_threshold`: Threshold for vorticity-based refinement
- `regrid_interval`: Number of time steps between regridding checks
- `buffer_size`: Number of buffer cells around refined regions
- `body_weight`: Weight for body proximity in combined indicator
- `gradient_weight`: Weight for velocity gradient in combined indicator
- `vorticity_weight`: Weight for vorticity in combined indicator
"""
struct AMRConfig
    max_level::Int
    body_distance_threshold::Float64
    velocity_gradient_threshold::Float64
    vorticity_threshold::Float64
    regrid_interval::Int
    buffer_size::Int
    body_weight::Float64
    gradient_weight::Float64
    vorticity_weight::Float64
end

"""
    AMRConfig(; max_level=2, body_distance_threshold=3.0, ...)

Create an AMR configuration with keyword arguments.
"""
function AMRConfig(; max_level::Int=2,
                    body_distance_threshold::Real=3.0,
                    velocity_gradient_threshold::Real=1.0,
                    vorticity_threshold::Real=1.0,
                    regrid_interval::Int=10,
                    buffer_size::Int=1,
                    body_weight::Real=0.5,
                    gradient_weight::Real=0.3,
                    vorticity_weight::Real=0.2)
    AMRConfig(max_level, Float64(body_distance_threshold),
              Float64(velocity_gradient_threshold), Float64(vorticity_threshold),
              regrid_interval, buffer_size,
              Float64(body_weight), Float64(gradient_weight), Float64(vorticity_weight))
end

"""
    AMRSimulation

Simulation with adaptive mesh refinement near immersed bodies.

Wraps a standard `Simulation` and adds AMR capability that automatically
refines the mesh near the body and in regions of high gradients.

# Fields
- `sim`: The underlying Simulation
- `config`: AMR configuration parameters
- `refined_grid`: Current refined grid state
- `composite_pois`: CompositePoisson solver for AMR (manages base + patches)
- `adapter`: Flow-to-grid adapter
- `last_regrid_step`: Step count at last regrid
- `amr_active`: Whether AMR is currently active

# Example
```julia
config = AMRConfig(max_level=3, body_distance_threshold=4.0, regrid_interval=5)
sim = AMRSimulation((128, 128), (1.0, 0.0), 16.0;
                    ν=0.01, body=AutoBody(sdf), amr_config=config)
for _ in 1:1000
    sim_step!(sim; remeasure=true)  # AMR regridding happens automatically
end
```
"""
mutable struct AMRSimulation <: AbstractSimulation
    sim::Simulation
    config::AMRConfig
    refined_grid::RefinedGrid
    composite_pois::CompositePoisson
    adapter::FlowToGridAdapter
    last_regrid_step::Int
    amr_active::Bool
end

"""
    AMRSimulation(dims, uBC, L::NTuple{N}; amr_config=AMRConfig(), kwargs...)

Create an AMR-enabled simulation.

# Arguments
- `dims`: Grid dimensions
- `uBC`: Boundary conditions
- `L::NTuple{N}`: Physical domain size tuple
- `amr_config`: AMR configuration (default: AMRConfig())
- `kwargs...`: Additional arguments passed to Simulation constructor (including `L_char`)
"""
function AMRSimulation(dims::NTuple{N}, uBC, L::NTuple{N};
                       amr_config::AMRConfig=AMRConfig(),
                       kwargs...) where N
    # Create base simulation
    sim = Simulation(dims, uBC, L; kwargs...)

    # Create adapter and refined grid using domain size L[1]
    adapter = FlowToGridAdapter(sim.flow, L[1])
    refined_grid = create_refined_grid(adapter)

    # Create composite Poisson solver wrapping the base MultiLevelPoisson
    composite_pois = CompositePoisson(sim.pois; max_level=amr_config.max_level)

    AMRSimulation(sim, amr_config, refined_grid, composite_pois, adapter, 0, true)
end

# Forward basic properties to underlying simulation
time(amr::AMRSimulation) = time(amr.sim)
sim_time(amr::AMRSimulation) = sim_time(amr.sim)
sim_info(amr::AMRSimulation) = begin
    base_info = "tU/L=$(round(sim_time(amr),digits=4)), Δt=$(round(amr.sim.flow.Δt[end],digits=3))"
    amr_info = ", AMR: $(num_refined_cells(amr.refined_grid)) refined cells"
    println(base_info * amr_info)
end

# Access underlying fields
Base.getproperty(amr::AMRSimulation, s::Symbol) = begin
    if s in (:sim, :config, :refined_grid, :composite_pois, :adapter, :last_regrid_step, :amr_active)
        getfield(amr, s)
    elseif s in (:U, :L, :ϵ, :flow, :body, :pois)
        getproperty(getfield(amr, :sim), s)
    else
        getfield(amr, s)
    end
end

"""
    sim_step!(sim::AMRSimulation; remeasure=true, λ=quick, kwargs...)

Advance AMR simulation by one time step with automatic regridding.
Uses CompositePoisson for pressure solve when AMR has refined patches.
"""
function sim_step!(amr::AMRSimulation; remeasure=true, λ=quick, udf=nothing, kwargs...)
    step_count = length(amr.sim.flow.Δt)

    # Check if regridding is needed
    if amr.amr_active && (step_count - amr.last_regrid_step) >= amr.config.regrid_interval
        amr_regrid!(amr)
        amr.last_regrid_step = step_count
    end

    # Perform simulation step
    remeasure && measure!(amr.sim)

    # Use AMR solver if patches exist, otherwise fall back to base solver
    if amr.amr_active && has_patches(amr.composite_pois)
        amr_mom_step!(amr.sim.flow, amr.composite_pois; λ)
    else
        mom_step!(amr.sim.flow, amr.sim.pois; λ, udf, kwargs...)
    end
end

"""
    sim_step!(sim::AMRSimulation, t_end; kwargs...)

Advance AMR simulation up to dimensionless time `t_end`.
"""
function sim_step!(amr::AMRSimulation, t_end; remeasure=true, λ=quick, max_steps=typemax(Int),
                   verbose=false, udf=nothing, kwargs...)
    steps₀ = length(amr.sim.flow.Δt)
    while sim_time(amr) < t_end && length(amr.sim.flow.Δt) - steps₀ < max_steps
        sim_step!(amr; remeasure, λ, udf, kwargs...)
        verbose && sim_info(amr)
    end
end

"""
    amr_regrid!(amr::AMRSimulation)

Perform AMR regridding based on current flow state and body position.
Updates both the RefinedGrid cell tracking and creates PatchPoisson solvers.
"""
function amr_regrid!(amr::AMRSimulation)
    flow = amr.sim.flow
    body = amr.sim.body
    config = amr.config
    t = time(amr.sim)

    # Compute combined refinement indicator
    indicator = compute_combined_indicator(flow, body;
        body_threshold=config.body_distance_threshold,
        gradient_threshold=config.velocity_gradient_threshold,
        vorticity_threshold=config.vorticity_threshold,
        t=t,
        body_weight=config.body_weight,
        gradient_weight=config.gradient_weight,
        vorticity_weight=config.vorticity_weight
    )

    # Apply buffer zone
    if config.buffer_size > 0
        apply_buffer_zone!(indicator; buffer_size=config.buffer_size)
    end

    # Mark cells for refinement
    cells_to_refine = mark_cells_for_refinement(indicator; threshold=0.5)

    # Update refined grid tracking
    update_refined_cells!(amr.refined_grid, cells_to_refine, config.max_level)

    # Create PatchPoisson solvers from marked cells
    create_patches!(amr.composite_pois, amr.refined_grid, flow.μ₀)

    # Synchronize solution data between base and patches
    if has_patches(amr.composite_pois)
        synchronize_base_and_patches!(flow, amr.composite_pois)
    end

    return amr
end

"""
    update_refined_cells!(rg::RefinedGrid, cells::Vector{CartesianIndex}, max_level::Int)

Update the refined grid with new cells to refine.
"""
function update_refined_cells!(rg::RefinedGrid, cells::Vector{CartesianIndex{N}},
                                max_level::Int) where N
    # Clear existing refinement
    if N == 2
        empty!(rg.refined_cells_2d)
        for I in cells
            i, j = I[1], I[2]
            if 1 <= i <= rg.base_grid.nx && 1 <= j <= rg.base_grid.nz
                rg.refined_cells_2d[(i, j)] = min(max_level, 1)
            end
        end
    else
        empty!(rg.refined_cells_3d)
        for I in cells
            i, j, k = I[1], I[2], I[3]
            if 1 <= i <= rg.base_grid.nx && 1 <= j <= rg.base_grid.ny && 1 <= k <= rg.base_grid.nz
                rg.refined_cells_3d[(i, j, k)] = min(max_level, 1)
            end
        end
    end
    return rg
end

"""
    set_amr_active!(amr::AMRSimulation, active::Bool)

Enable or disable AMR regridding.
"""
set_amr_active!(amr::AMRSimulation, active::Bool) = (amr.amr_active = active; amr)

"""
    get_refinement_indicator(amr::AMRSimulation)

Compute and return the current refinement indicator without applying regridding.
Useful for visualization and debugging.
"""
function get_refinement_indicator(amr::AMRSimulation)
    compute_combined_indicator(amr.sim.flow, amr.sim.body;
        body_threshold=amr.config.body_distance_threshold,
        gradient_threshold=amr.config.velocity_gradient_threshold,
        vorticity_threshold=amr.config.vorticity_threshold,
        t=time(amr.sim),
        body_weight=amr.config.body_weight,
        gradient_weight=amr.config.gradient_weight,
        vorticity_weight=amr.config.vorticity_weight
    )
end

perturb!(amr::AMRSimulation; noise=0.1) = perturb!(amr.sim; noise)

function measure!(amr::AMRSimulation, t=sum(amr.sim.flow.Δt))
    measure!(amr.sim, t)
    # Update composite Poisson after remeasure (coefficients may change)
    update!(amr.composite_pois)
end

"""
    amr_info(amr::AMRSimulation)

Print detailed AMR status information.
"""
function amr_info(amr::AMRSimulation)
    println("AMR Status:")
    println("  Active: ", amr.amr_active)
    println("  Refined cells: ", num_refined_cells(amr.refined_grid))
    println("  Number of patches: ", num_patches(amr.composite_pois))
    if has_patches(amr.composite_pois)
        for (anchor, patch) in amr.composite_pois.patches
            println("    Patch at ", anchor, ": level=", patch.level,
                    ", fine dims=", patch.fine_dims)
        end
    end
end

"""
    check_divergence(amr::AMRSimulation; verbose=false)

Check divergence at all refinement levels.
"""
check_divergence(amr::AMRSimulation; verbose::Bool=false) =
    check_amr_divergence(amr.sim.flow, amr.composite_pois; verbose)

export AMRConfig, AMRSimulation, amr_regrid!, set_amr_active!, get_refinement_indicator
export amr_info, check_divergence

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
export cell_center_velocity, cell_center_vorticity, cell_center_pressure
export CenterFieldWriter, ForceWriter, maybe_save!
export force_components, force_coefficients, record_force!
export compute_diagnostics, summarize_force_history

end # module
