struct Grid2D
    nx::Int
    ny::Int
    Lx::Float64
    Ly::Float64
    dx::Float64
    dy::Float64
end

struct Grid3D
    nx::Int
    ny::Int
    nz::Int
    Lx::Float64
    Ly::Float64
    Lz::Float64
    dx::Float64
    dy::Float64
    dz::Float64
end

Grid2D(nx,ny; Lx=1.0, Ly=1.0) = Grid2D(nx,ny,Lx,Ly,Lx/nx,Ly/ny)
Grid3D(nx,ny,nz; Lx=1.0, Ly=1.0, Lz=1.0) = Grid3D(nx,ny,nz,Lx,Ly,Lz,Lx/nx,Ly/ny,Lz/nz)

struct Params
    dt::Float64
    ν::Float64
    ρ::Float64
    penalty::Float64   # strength for Brinkman penalization
    accel::NTuple{3,Float64}  # uniform acceleration forcing (g)
end

Params(;dt=1e-3, ν=1e-3, ρ=1.0, penalty=1e3, accel=(0.0,0.0,0.0)) = Params(dt,ν,ρ,penalty,accel)

"""
State on a MAC grid in 2D:
 - `u`: size (nx+1, ny) face-x velocities
 - `v`: size (nx, ny+1) face-y velocities
 - `p`: size (nx, ny) cell-center pressure
"""
struct State2D
    u::Array{Float64,2}
    v::Array{Float64,2}
    p::Array{Float64,2}
end

"""
State on a MAC grid in 3D:
 - `u`: size (nx+1, ny, nz)
 - `v`: size (nx, ny+1, nz)
 - `w`: size (nx, ny, nz+1)
 - `p`: size (nx, ny, nz)
"""
struct State3D
    u::Array{Float64,3}
    v::Array{Float64,3}
    w::Array{Float64,3}
    p::Array{Float64,3}
end

function zeros_state2D(g::Grid2D)
    State2D(zeros(g.nx+1, g.ny), zeros(g.nx, g.ny+1), zeros(g.nx, g.ny))
end

function zeros_state3D(g::Grid3D)
    State3D(zeros(g.nx+1, g.ny, g.nz), zeros(g.nx, g.ny+1, g.nz), zeros(g.nx, g.ny, g.nz+1), zeros(g.nx, g.ny, g.nz))
end

