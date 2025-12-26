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
export compute_face_flux!,apply_fluxes!,conv_diff_fvm!

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
    Simulation(dims::NTuple{N}, L::NTuple{N}; inletBC=(1,0,...), kwargs...)

Constructor for a BioFlows simulation solving the dimensional incompressible Navier-Stokes equations:

    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + g
    ∇·u = 0

# Arguments

## Required
- `dims::NTuple{N,Int}`: Number of grid cells in each direction, e.g., `(nx, nz)` or `(nx, ny, nz)`
- `L::NTuple{N}`: Physical domain size in each direction (e.g., `(Lx, Lz)` in meters)

## Optional (keyword arguments)
- `inletBC`: Inlet boundary velocity (default: unit velocity in x-direction)
  - `Tuple`: Constant velocity, e.g., `(1.0, 0.0)` for uniform flow
  - `Function(i,x,t)`: Spatially/temporally varying (requires `U` to be specified)
- `outletBC=false`: Enable convective outlet BC in direction 1
- `ν=0.`: Kinematic viscosity (m²/s)
- `ρ=1000.`: Fluid density (kg/m³). Water = 1000, air ≈ 1.2
- `body=NoBody()`: Immersed body geometry
- `L_char`: Characteristic length for force coefficients (default: `L[1]`)
- `U`: Velocity scale. Auto-computed from `inletBC` if constant, required if function
- `Δt=0.25`: Initial time step (seconds)
- `fixed_Δt=nothing`: Fixed time step (seconds). If specified, disables adaptive CFL time stepping.
- `g=nothing`: Body acceleration function `g(i,x,t)` (m/s²)
- `ϵ=1`: BDIM kernel width (in grid cells)
- `perdir=()`: Periodic directions, e.g., `(2,)` for z-periodic
- `uλ=nothing`: Initial velocity condition. Tuple or `Function(i,x)`
- `T=Float32`: Numeric type
- `mem=Array`: Memory backend (`Array`, `CuArray`, etc.)
- `store_fluxes=false`: Enable FVM flux storage for conservation analysis

# Examples

## Constant inlet velocity
```julia
# 2D channel with uniform inlet
sim = Simulation((256, 128), (2.0, 1.0); inletBC=(1.0, 0.0), ν=1e-6)

# With immersed cylinder
diameter = 0.2
cylinder = AutoBody((x,t) -> √(x[1]^2 + x[2]^2) - diameter/2)
sim = Simulation((256, 128), (2.0, 1.0);
                 inletBC = (1.0, 0.0),
                 ν = 1e-6,
                 body = cylinder,
                 L_char = diameter)
```

## Spatially-varying inlet (parabolic profile)
```julia
# Parabolic inlet: u(z) = U_max * (1 - (z - H)²/H²)
Lx, Lz = 2.0, 1.0
H = Lz / 2
U_max = 1.5
inletBC(i, x, t) = i == 1 ? U_max * (1 - ((x[2] - H) / H)^2) : 0.0

sim = Simulation((256, 128), (Lx, Lz);
                 inletBC = inletBC,
                 U = U_max,
                 ν = 1e-6,
                 outletBC = true)
```

## Time-varying inlet
```julia
U₀, ω = 1.0, 2π
inletBC(i, x, t) = i == 1 ? U₀ * (1 + 0.1*sin(ω*t)) : 0.0

sim = Simulation((256, 128), (2.0, 1.0); inletBC=inletBC, U=U₀, ν=1e-6)
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

    function Simulation(dims::NTuple{N}, L::NTuple{N};
                        inletBC=nothing, L_char=nothing, Δt=0.25, ν=0., ρ=1000., g=nothing, U=nothing, ϵ=1, perdir=(),
                        uλ=nothing, outletBC=false, body::AbstractBody=NoBody(),
                        T=Float32, mem=Array, fixed_Δt=nothing, store_fluxes=false) where N
        # Default inletBC: unit velocity in x-direction
        if isnothing(inletBC)
            inletBC = ntuple(i -> i==1 ? one(T) : zero(T), N)
        end
        @assert !(isnothing(U) && isa(inletBC,Function)) "`U` (velocity scale) must be specified if boundary conditions `inletBC` is a `Function`"
        isnothing(U) && (U = √sum(abs2,inletBC))
        check_fn(inletBC,N,T,3); check_fn(g,N,T,3); check_fn(uλ,N,T,2)
        # Pass domain size L to Flow for dimensional Δx computation
        flow = Flow(dims;L=L,inletBC=inletBC,uλ,Δt,ν,ρ,g,T,f=mem,perdir,outletBC,fixed_Δt=fixed_Δt,store_fluxes=store_fluxes)
        measure!(flow,body;ϵ)
        # Use L_char for dimensionless time/forces, default to L[1]
        char_length = isnothing(L_char) ? L[1] : L_char

        # Check for anisotropic grids - not supported
        Δx = flow.Δx
        if !all(isapprox.(Δx, Δx[1], rtol=1e-6))
            error("Anisotropic grids (Δx ≠ Δy) are not supported. " *
                  "The pressure solver requires uniform grid spacing in all directions. " *
                  "Got Δx = $Δx. Adjust grid resolution or domain size to ensure Δx = Δy = Δz.")
        end

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

## Flexible Body Support (for moving/deforming bodies)
- `flexible_body`: Enable adaptive regridding for time-varying bodies
- `indicator_change_threshold`: Regrid when indicator changes by this fraction (0-1)
- `regrid_on_measure`: Always regrid after body remeasurement (most accurate but expensive)
- `min_regrid_interval`: Minimum steps between regrids (prevents excessive regridding)
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
    # Flexible body support
    flexible_body::Bool
    indicator_change_threshold::Float64
    regrid_on_measure::Bool
    min_regrid_interval::Int
end

"""
    AMRConfig(; max_level=2, body_distance_threshold=3.0, ...)

Create an AMR configuration with keyword arguments.

# Flexible Body Options
For moving or deforming bodies (e.g., swimming fish), use these options:
- `flexible_body=true`: Enable motion-adaptive regridding
- `indicator_change_threshold=0.1`: Regrid when 10% of cells change refinement status
- `regrid_on_measure=false`: Set `true` for fastest bodies (expensive but most accurate)
- `min_regrid_interval=1`: Allow regridding every step if needed

# Example for Swimming Fish
```julia
config = AMRConfig(
    max_level=2,
    body_distance_threshold=3.0,
    flexible_body=true,
    indicator_change_threshold=0.1,
    min_regrid_interval=2
)
```
"""
function AMRConfig(; max_level::Int=2,
                    body_distance_threshold::Real=3.0,
                    velocity_gradient_threshold::Real=1.0,
                    vorticity_threshold::Real=1.0,
                    regrid_interval::Int=10,
                    buffer_size::Int=1,
                    body_weight::Real=0.5,
                    gradient_weight::Real=0.3,
                    vorticity_weight::Real=0.2,
                    # Flexible body options
                    flexible_body::Bool=false,
                    indicator_change_threshold::Real=0.1,
                    regrid_on_measure::Bool=false,
                    min_regrid_interval::Int=1)
    AMRConfig(max_level, Float64(body_distance_threshold),
              Float64(velocity_gradient_threshold), Float64(vorticity_threshold),
              regrid_interval, buffer_size,
              Float64(body_weight), Float64(gradient_weight), Float64(vorticity_weight),
              flexible_body, Float64(indicator_change_threshold),
              regrid_on_measure, min_regrid_interval)
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
    # Flexible body tracking
    last_body_indicator::Union{Nothing, AbstractArray}  # Previous body indicator for change detection
end

"""
    AMRSimulation(dims, L::NTuple{N}; inletBC=nothing, amr_config=AMRConfig(), kwargs...)

Create an AMR-enabled simulation.

# Arguments
- `dims`: Grid dimensions
- `L::NTuple{N}`: Physical domain size tuple
- `inletBC`: Inlet boundary conditions (default: unit velocity in x-direction)
- `amr_config`: AMR configuration (default: AMRConfig())
- `kwargs...`: Additional arguments passed to Simulation constructor (including `L_char`)
"""
function AMRSimulation(dims::NTuple{N}, L::NTuple{N};
                       inletBC=nothing,
                       amr_config::AMRConfig=AMRConfig(),
                       kwargs...) where N
    # Create base simulation
    sim = Simulation(dims, L; inletBC=inletBC, kwargs...)

    # Create adapter and refined grid using domain size L[1]
    adapter = FlowToGridAdapter(sim.flow, L[1])
    refined_grid = create_refined_grid(adapter)

    # Create composite Poisson solver wrapping the base MultiLevelPoisson
    composite_pois = CompositePoisson(sim.pois; max_level=amr_config.max_level)

    # Set μ₀ reference for flexible body coefficient updates
    set_μ₀_reference!(composite_pois, sim.flow.μ₀)

    # Initialize with nothing for last_body_indicator (will be set on first regrid)
    AMRSimulation(sim, amr_config, refined_grid, composite_pois, adapter, 0, true, nothing)
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

For flexible bodies (moving/deforming), the function:
1. First remeasures the body position (if remeasure=true)
2. Checks if regridding is needed based on:
   - Fixed interval (regrid_interval)
   - Body motion detection (if flexible_body=true)
   - Always-regrid mode (if regrid_on_measure=true)
3. Updates patch coefficients after body remeasurement
"""
function sim_step!(amr::AMRSimulation; remeasure=true, λ=quick, udf=nothing, kwargs...)
    step_count = length(amr.sim.flow.Δt)
    config = amr.config
    steps_since_regrid = step_count - amr.last_regrid_step

    # IMPORTANT: Remeasure body FIRST (before regridding check)
    # This ensures the body position is current when computing refinement indicators
    if remeasure
        measure!(amr.sim)
        # Update composite Poisson coefficients after body remeasurement
        update!(amr.composite_pois)
    end

    # Determine if regridding is needed
    need_regrid = false

    if amr.amr_active
        # Check minimum interval constraint
        can_regrid = steps_since_regrid >= config.min_regrid_interval

        if can_regrid
            # Standard interval-based regridding
            if steps_since_regrid >= config.regrid_interval
                need_regrid = true
            end

            # Flexible body: motion-triggered regridding
            if config.flexible_body && remeasure
                if config.regrid_on_measure
                    # Always regrid after remeasure (most accurate, most expensive)
                    need_regrid = true
                else
                    # Check if body indicator has changed significantly
                    need_regrid = need_regrid || should_regrid_for_body_motion(amr)
                end
            end
        end
    end

    # Perform regridding if needed
    if need_regrid
        amr_regrid!(amr)
        amr.last_regrid_step = step_count
    end

    # Use AMR solver if patches exist, otherwise fall back to base solver
    if amr.amr_active && has_patches(amr.composite_pois)
        amr_mom_step!(amr.sim.flow, amr.composite_pois; λ)
    else
        mom_step!(amr.sim.flow, amr.sim.pois; λ, udf, kwargs...)
    end
end

"""
    should_regrid_for_body_motion(amr::AMRSimulation)

Check if the body has moved enough to warrant regridding.
Computes the current body proximity indicator and compares with the last one.
Returns true if the fraction of changed cells exceeds the threshold.
"""
function should_regrid_for_body_motion(amr::AMRSimulation)
    config = amr.config
    flow = amr.sim.flow
    body = amr.sim.body

    # Compute current body indicator
    current_indicator = compute_body_refinement_indicator(flow, body;
        distance_threshold=config.body_distance_threshold,
        t=time(amr.sim))

    # If no previous indicator, store current and return false (first step)
    if isnothing(amr.last_body_indicator)
        amr.last_body_indicator = copy(current_indicator)
        return false
    end

    # Compare indicators: count cells that changed refinement status
    n_total = length(current_indicator)
    n_changed = 0
    for I in eachindex(current_indicator)
        # A cell "changed" if it went from refined to unrefined or vice versa
        was_refined = amr.last_body_indicator[I] > 0.5
        is_refined = current_indicator[I] > 0.5
        if was_refined != is_refined
            n_changed += 1
        end
    end

    # Compute change fraction
    change_fraction = n_changed / n_total

    # Update stored indicator if we'll regrid, or if change is significant
    if change_fraction > config.indicator_change_threshold
        amr.last_body_indicator = copy(current_indicator)
        return true
    end

    return false
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

For flexible bodies, this function is called:
- At regular intervals (regrid_interval)
- When body motion is detected (if flexible_body=true)
- After every remeasure (if regrid_on_measure=true)
"""
function amr_regrid!(amr::AMRSimulation)
    flow = amr.sim.flow
    body = amr.sim.body
    config = amr.config
    t = time(amr.sim)

    # Compute body indicator first (needed for flexible body tracking)
    body_indicator = compute_body_refinement_indicator(flow, body;
        distance_threshold=config.body_distance_threshold, t=t)

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

    # Update last_body_indicator for flexible body tracking
    if config.flexible_body
        amr.last_body_indicator = copy(body_indicator)
    end

    return amr
end

"""
    update_refined_cells!(rg::RefinedGrid, cells::Vector{CartesianIndex}, max_level::Int)

Update the refined grid with new cells to refine.
"""
function update_refined_cells!(rg::RefinedGrid, cells::Vector{CartesianIndex{N}},
                                max_level::Int) where N
    max_level < 1 && return rg
    level = max_level
    # Clear existing refinement
    if N == 2
        i_max = rg.base_grid.nx + 1
        j_max = rg.base_grid.nz + 1
        empty!(rg.refined_cells_2d)
        for I in cells
            i, j = I[1], I[2]
            if 2 <= i <= i_max && 2 <= j <= j_max
                rg.refined_cells_2d[(i, j)] = level
            end
        end
    else
        i_max = rg.base_grid.nx + 1
        j_max = rg.base_grid.ny + 1
        k_max = rg.base_grid.nz + 1
        empty!(rg.refined_cells_3d)
        for I in cells
            i, j, k = I[1], I[2], I[3]
            if 2 <= i <= i_max && 2 <= j <= j_max && 2 <= k <= k_max
                rg.refined_cells_3d[(i, j, k)] = level
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
    amr_info(amr::AMRSimulation; verbose=true)

Get AMR status information. Optionally prints detailed status.

# Returns
NamedTuple with:
- `active`: Whether AMR is active
- `flexible_body`: Whether flexible body mode is enabled
- `refined_cells`: Number of refined cells
- `num_patches`: Number of active patches
- `steps_since_regrid`: Steps since last regridding
"""
function amr_info(amr::AMRSimulation; verbose::Bool=true)
    n_refined = num_refined_cells(amr.refined_grid)
    n_patches = num_patches(amr.composite_pois)
    steps_since = length(amr.sim.flow.Δt) - amr.last_regrid_step

    if verbose
        println("AMR Status:")
        println("  Active: ", amr.amr_active)
        println("  Flexible body: ", amr.config.flexible_body)
        println("  Refined cells: ", n_refined)
        println("  Number of patches: ", n_patches)
        println("  Steps since last regrid: ", steps_since)
        if has_patches(amr.composite_pois)
            for (anchor, patch) in amr.composite_pois.patches
                println("    Patch at ", anchor, ": level=", patch.level,
                        ", fine dims=", patch.fine_dims)
            end
        end
    end

    return (
        active = amr.amr_active,
        flexible_body = amr.config.flexible_body,
        refined_cells = n_refined,
        num_patches = n_patches,
        steps_since_regrid = steps_since
    )
end

# =============================================================================
# FLEXIBLE BODY AMR HELPER FUNCTIONS
# =============================================================================

"""
    FlexibleBodyAMRConfig(; kwargs...)

Convenience constructor for AMRConfig optimized for flexible/deforming bodies
such as swimming fish.

# Default Settings
- `flexible_body=true`: Enable motion-adaptive regridding
- `indicator_change_threshold=0.05`: 5% cell change triggers regrid
- `min_regrid_interval=2`: Allow regridding every 2 steps
- `regrid_interval=5`: Check regridding at least every 5 steps
- `body_distance_threshold=4.0`: Larger refinement region for moving bodies

# Example
```julia
config = FlexibleBodyAMRConfig(max_level=2)
sim = AMRSimulation((256, 128), (1.0, 0.5); body=fish, amr_config=config)
for _ in 1:1000
    sim_step!(sim; remeasure=true)  # Patches follow the fish automatically
end
```
"""
function FlexibleBodyAMRConfig(; max_level::Int=2,
                                 body_distance_threshold::Real=4.0,
                                 velocity_gradient_threshold::Real=1.0,
                                 vorticity_threshold::Real=1.0,
                                 regrid_interval::Int=5,
                                 buffer_size::Int=2,
                                 body_weight::Real=0.6,
                                 gradient_weight::Real=0.2,
                                 vorticity_weight::Real=0.2,
                                 indicator_change_threshold::Real=0.05,
                                 regrid_on_measure::Bool=false,
                                 min_regrid_interval::Int=2)
    AMRConfig(;
        max_level=max_level,
        body_distance_threshold=body_distance_threshold,
        velocity_gradient_threshold=velocity_gradient_threshold,
        vorticity_threshold=vorticity_threshold,
        regrid_interval=regrid_interval,
        buffer_size=buffer_size,
        body_weight=body_weight,
        gradient_weight=gradient_weight,
        vorticity_weight=vorticity_weight,
        flexible_body=true,
        indicator_change_threshold=indicator_change_threshold,
        regrid_on_measure=regrid_on_measure,
        min_regrid_interval=min_regrid_interval
    )
end

"""
    RigidBodyAMRConfig(; kwargs...)

Convenience constructor for AMRConfig optimized for rigid moving bodies
such as oscillating cylinders, rotating ellipses, or translating objects.

Rigid bodies move without deformation, so the motion pattern is typically
more predictable than flexible bodies. This configuration uses slightly
less aggressive regridding than FlexibleBodyAMRConfig.

# Default Settings
- `flexible_body=true`: Enable motion-adaptive regridding
- `indicator_change_threshold=0.08`: 8% cell change triggers regrid
- `min_regrid_interval=3`: Allow regridding every 3 steps
- `regrid_interval=8`: Check regridding at least every 8 steps
- `body_distance_threshold=3.0`: Standard refinement region

# Supported Motion Types
- Translation (e.g., oscillating cylinder)
- Rotation (e.g., rotating ellipse)
- Combined translation + rotation
- Prescribed motion paths

# Example
```julia
# Oscillating cylinder with AMR
sdf(x, t) = norm(x .- center) - radius
map(x, t) = x .- [0, A*sin(ω*t)]  # Vertical oscillation
body = AutoBody(sdf, map)

config = RigidBodyAMRConfig(max_level=2)
sim = AMRSimulation((128, 128), (L, L); body=body, amr_config=config)

for _ in 1:1000
    sim_step!(sim; remeasure=true)  # Patches follow the cylinder
end
```
"""
function RigidBodyAMRConfig(; max_level::Int=2,
                              body_distance_threshold::Real=3.0,
                              velocity_gradient_threshold::Real=1.0,
                              vorticity_threshold::Real=1.0,
                              regrid_interval::Int=8,
                              buffer_size::Int=2,
                              body_weight::Real=0.5,
                              gradient_weight::Real=0.3,
                              vorticity_weight::Real=0.2,
                              indicator_change_threshold::Real=0.08,
                              regrid_on_measure::Bool=false,
                              min_regrid_interval::Int=3)
    AMRConfig(;
        max_level=max_level,
        body_distance_threshold=body_distance_threshold,
        velocity_gradient_threshold=velocity_gradient_threshold,
        vorticity_threshold=vorticity_threshold,
        regrid_interval=regrid_interval,
        buffer_size=buffer_size,
        body_weight=body_weight,
        gradient_weight=gradient_weight,
        vorticity_weight=vorticity_weight,
        flexible_body=true,  # Uses same mechanism as flexible bodies
        indicator_change_threshold=indicator_change_threshold,
        regrid_on_measure=regrid_on_measure,
        min_regrid_interval=min_regrid_interval
    )
end

"""
    force_regrid!(amr::AMRSimulation)

Force an immediate regridding operation, regardless of step count or motion detection.
Useful when you've made significant changes to the body or flow field.
"""
function force_regrid!(amr::AMRSimulation)
    amr_regrid!(amr)
    amr.last_regrid_step = length(amr.sim.flow.Δt)
    return amr
end

"""
    reset_body_tracking!(amr::AMRSimulation)

Reset the body indicator tracking for flexible bodies.
Call this after making sudden changes to the body position/shape.
"""
function reset_body_tracking!(amr::AMRSimulation)
    amr.last_body_indicator = nothing
    return amr
end

"""
    get_body_motion_stats(amr::AMRSimulation)

Get statistics about body motion for debugging and tuning.

# Returns
NamedTuple with:
- `n_refined_cells`: Number of currently refined cells
- `indicator_stored`: Whether body indicator is being tracked
- `steps_since_regrid`: Steps since last regridding
- `n_patches`: Number of active patches
"""
function get_body_motion_stats(amr::AMRSimulation)
    (
        n_refined_cells = num_refined_cells(amr.refined_grid),
        indicator_stored = !isnothing(amr.last_body_indicator),
        steps_since_regrid = length(amr.sim.flow.Δt) - amr.last_regrid_step,
        n_patches = num_patches(amr.composite_pois),
        flexible_body_enabled = amr.config.flexible_body
    )
end

"""
    estimate_body_displacement(amr::AMRSimulation)

Estimate how much the body has moved since the last regridding.
Useful for tuning `indicator_change_threshold`.

Returns the fraction of cells that have changed refinement status.
"""
function estimate_body_displacement(amr::AMRSimulation)
    if !amr.config.flexible_body || isnothing(amr.last_body_indicator)
        return 0.0
    end

    flow = amr.sim.flow
    body = amr.sim.body
    config = amr.config

    # Compute current body indicator
    current_indicator = compute_body_refinement_indicator(flow, body;
        distance_threshold=config.body_distance_threshold,
        t=time(amr.sim))

    # Count changed cells
    n_total = length(current_indicator)
    n_changed = 0
    for I in eachindex(current_indicator)
        was_refined = amr.last_body_indicator[I] > 0.5
        is_refined = current_indicator[I] > 0.5
        if was_refined != is_refined
            n_changed += 1
        end
    end

    return n_changed / n_total
end

"""
    check_divergence(amr::AMRSimulation; verbose=false)

Check divergence at all refinement levels.
"""
check_divergence(amr::AMRSimulation; verbose::Bool=false) =
    check_amr_divergence(amr.sim.flow, amr.composite_pois; verbose)

export AMRConfig, AMRSimulation, amr_regrid!, set_amr_active!, get_refinement_indicator
export amr_info, check_divergence
# Moving body AMR exports (flexible and rigid bodies)
export FlexibleBodyAMRConfig, RigidBodyAMRConfig
export force_regrid!, reset_body_tracking!
export get_body_motion_stats, estimate_body_displacement, should_regrid_for_body_motion

# =============================================================================
# FLUID-STRUCTURE INTERACTION (FSI) MODULE
# =============================================================================
# Euler-Bernoulli beam coupled with incompressible Navier-Stokes
# =============================================================================

_silent_include("fsi/EulerBernoulliBeam.jl")
_silent_include("fsi/FluidStructureCoupling.jl")
_silent_include("fsi/BeamAMR.jl")

# FSI exports
export BeamBoundaryCondition, CLAMPED, FREE, PINNED, PRESCRIBED
export BeamMaterial, BeamGeometry, EulerBernoulliBeam
export fish_thickness_profile
export FlexibleBodyFSI, FSISimulation
export traveling_wave_forcing, heave_pitch_forcing
export get_beam, get_displacement, get_velocity, get_curvature
export get_bending_moment, kinetic_energy, potential_energy, total_energy
export set_fluid_load!, set_active_forcing!, get_fluid_load, reset!, step!
export BeamStateWriter, BeamStateWriterGroup, close!

# FSI AMR exports
export FlexibleBodySDF, BeamAMRConfig, BeamAMRTracker
export compute_beam_refinement_indicator, compute_beam_combined_indicator
export create_beam_body, regrid_for_beam!, should_regrid, mark_regrid!
export get_beam_bounding_box, count_refined_cells_near_beam

# BeamAMRSimulation exports
export BeamAMRSimulation, set_forcing!, beam_info, get_flow, get_beam
export perform_beam_regrid!, swimming_fish_simulation

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
export CenterFieldWriter, ForceWriter, file_save!
export force_components, force_coefficients, record_force!
export compute_diagnostics, summarize_force_history

end # module
