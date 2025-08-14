abstract type AbstractGrid end
abstract type AbstractBody end
abstract type AbstractBoundaryCondition end
abstract type AbstractTimeStepping end
abstract type AbstractSolver end

abstract type DensityModel end
struct ConstantDensity <: DensityModel
    ρ::Float64
end

struct VariableDensity <: DensityModel
    ρ_func::Function
end

@enum GridType TwoDimensional ThreeDimensional

@enum BoundaryType NoSlip FreeSlip Periodic Inlet Outlet

struct FluidProperties
    μ::Float64  # Dynamic viscosity
    ρ::DensityModel  # Density model
    Re::Float64  # Reynolds number
    function FluidProperties(μ, ρ, Re)
        new(μ, ρ, Re)
    end
end

struct SimulationParameters
    dt::Float64  # Time step
    T_final::Float64  # Final time
    CFL::Float64  # CFL number
    save_interval::Int  # Save every N steps
    adaptive_dt::Bool  # Adaptive time stepping
end

mutable struct SolutionState{T<:Real}
    u::Array{T}  # x-velocity
    v::Array{T}  # y-velocity  
    w::Array{T}  # z-velocity (3D only)
    p::Array{T}  # pressure
    t::T         # current time
    step::Int    # current step
end

function SolutionState2D(nx, nz, T=Float64)
    SolutionState{T}(
        zeros(T, nx+1, nz),    # u (staggered in x)
        zeros(T, nx, nz+1),    # v (staggered in z, represents w-velocity for XZ plane)
        zeros(T, 0, 0),        # w (not used in 2D)
        zeros(T, nx, nz),      # p (cell-centered)
        zero(T),               # t
        0                      # step
    )
end

function SolutionState3D(nx, ny, nz, T=Float64)
    SolutionState{T}(
        zeros(T, nx+1, ny, nz),    # u (staggered)
        zeros(T, nx, ny+1, nz),    # v (staggered)
        zeros(T, nx, ny, nz+1),    # w (staggered)
        zeros(T, nx, ny, nz),      # p (cell-centered)
        zero(T),                   # t
        0                          # step
    )
end

# MPI-aware solution state with ghost cells
mutable struct MPISolutionState2D{T<:Real}
    u::Matrix{T}     # u-velocity with ghost cells (nx_g+1, nz_g) - XZ plane
    v::Matrix{T}     # v-velocity with ghost cells (nx_g, nz_g+1) - represents w-velocity in XZ plane
    p::Matrix{T}     # pressure with ghost cells (nx_g, nz_g)
    t::T             # current time
    step::Int        # current step
    decomp::Union{Nothing, Any}  # MPI decomposition info
end

function MPISolutionState2D(decomp, T=Float64)
    nx_g = decomp.nx_local_with_ghosts
    nz_g = decomp.nz_local_with_ghosts  # nz_local for XZ plane
    
    MPISolutionState2D{T}(
        zeros(T, nx_g + 1, nz_g),  # u (staggered in x)
        zeros(T, nx_g, nz_g + 1),  # v (staggered in z, represents w-velocity for XZ plane)
        zeros(T, nx_g, nz_g),      # p (cell-centered)
        zero(T),                   # t
        0,                         # step
        decomp                     # MPI decomposition
    )
end

mutable struct MPISolutionState3D{T<:Real}
    u::Array{T,3}    # u-velocity with ghost cells
    v::Array{T,3}    # v-velocity with ghost cells
    w::Array{T,3}    # w-velocity with ghost cells
    p::Array{T,3}    # pressure with ghost cells
    t::T             # current time
    step::Int        # current step
    decomp::Union{Nothing, Any}  # MPI decomposition info
end