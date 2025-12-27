# =============================================================================
# SIMULATION OUTPUT UTILITIES
# =============================================================================
# This module provides tools for saving simulation results to disk:
#
# - CenterFieldWriter: Periodic snapshot writer to JLD2 files
# - ForceWriter: Periodic writer for lift/drag coefficients to JLD2 files
# - file_save!: Conditional snapshot saving based on time intervals
#
# Output files use JLD2 format (Julia Data Format v2) which can be read back
# using the JLD2.jl package:
#   using JLD2
#   data = load("center_fields.jld2")
#   vel = data["snapshot_1/velocity"]
#
# Force coefficient files use JLD2 format with arrays:
#   time, Cd, Cl, Cd_pressure, Cd_viscous, Cl_pressure, Cl_viscous
#
# Snapshots include:
# - Velocity (cell-centered, [nx, nz, 2] for 2D or [nx, ny, nz, 3] for 3D)
# - Vorticity (cell-centered, scalar for 2D or vector for 3D)
# - Pressure (cell-centered)
# =============================================================================

using JLD2

_to_host(a) = a isa Array ? a : Array(a)

function _load_centerfield_state(filename::AbstractString, interval::Float64)
    samples = 0
    next_time = interval
    try
        jldopen(filename, "r") do file
            groups = String[]
            for key in keys(file)
                if startswith(key, "snapshot_")
                    push!(groups, key)
                elseif occursin("/", key)
                    head = split(key, '/')[1]
                    startswith(head, "snapshot_") && push!(groups, head)
                end
            end
            groups = unique(groups)
            isempty(groups) && return
            nums = Int[]
            for group in groups
                m = match(r"^snapshot_(\d+)$", group)
                m === nothing && continue
                push!(nums, parse(Int, m.captures[1]))
            end
            isempty(nums) && return
            samples = maximum(nums)
            time_key = "snapshot_$(samples)/time"
            if haskey(file, time_key)
                next_time = float(file[time_key]) + interval
            else
                next_time = (samples + 1) * interval
            end
        end
    catch
        samples = 0
        next_time = interval
    end
    return samples, next_time
end

"""
    CenterFieldWriter(filename::AbstractString="center_fields.jld2";
                      interval::Real=0.1,
                      overwrite::Bool=true,
                      strip_ghosts::Bool=true,
                      save_grid::Bool=true,
                      grid_filename::AbstractString="")

Helper that saves cell-centred velocity, vorticity, and pressure fields to a
JLD2 file at fixed convective-time intervals. Call [`file_save!`](@ref) after
each `sim_step!` to trigger writes.

# Arguments
- `filename`: Output JLD2 file path for field data
- `interval`: Time interval between saves (default: 0.1)
- `overwrite`: If true, overwrite existing file (default: true)
- `strip_ghosts`: If true, exclude ghost cells from output (default: true)
- `save_grid`: If true, save grid coordinates with first snapshot (default: true)
- `grid_filename`: Custom grid file name (default: same as filename with "_grid" suffix)

# Grid Output
When `save_grid=true`, a separate grid file is created containing:
- Cell-center coordinates (x, z for 2D; x, y, z for 3D)
- Grid spacing (dx, dz for 2D; dx, dy, dz for 3D)
- Grid dimensions (nx, nz for 2D; nx, ny, nz for 3D)
"""
mutable struct CenterFieldWriter
    filename::String
    interval::Float64
    next_time::Float64
    samples::Int
    strip_ghosts::Bool
    save_grid::Bool
    grid_filename::String
    grid_saved::Bool
    function CenterFieldWriter(filename::AbstractString="center_fields.jld2";
                               interval::Real=0.1,
                               overwrite::Bool=true,
                               strip_ghosts::Bool=true,
                               save_grid::Bool=true,
                               grid_filename::AbstractString="")
        interval > 0 || throw(ArgumentError("interval must be positive"))
        if overwrite && isfile(filename)
            rm(filename)
        end
        samples = 0
        next_time = float(interval)
        if !overwrite && isfile(filename)
            samples, next_time = _load_centerfield_state(filename, float(interval))
        end
        # Default grid filename: replace .jld2 with _grid.jld2
        if isempty(grid_filename)
            base = replace(filename, r"\.jld2$" => "")
            grid_filename = base * "_grid.jld2"
        end
        grid_saved = !save_grid  # If not saving grid, mark as already saved
        return new(String(filename), float(interval), next_time, samples,
                   strip_ghosts, save_grid, String(grid_filename), grid_saved)
    end
end

"""
    file_save!(writer, sim)

Check the simulation time and, if the configured interval has elapsed, append a
snapshot with cell-centred velocity, vorticity, and pressure to the writer's
JLD2 file.
"""
function file_save!(writer::CenterFieldWriter, sim::AbstractSimulation)
    t = sim_time(sim)
    if t + eps(writer.interval) < writer.next_time
        return writer
    end
    while t + eps(writer.interval) >= writer.next_time
        _write_snapshot!(writer, sim)
        writer.samples += 1
        writer.next_time += writer.interval
    end
    return writer
end

function _write_snapshot!(writer::CenterFieldWriter, sim::AbstractSimulation)
    # Save grid file on first snapshot if requested
    if writer.save_grid && !writer.grid_saved
        save_grid(writer.grid_filename, sim; strip_ghosts=writer.strip_ghosts)
        writer.grid_saved = true
    end

    vel = _to_host(cell_center_velocity(sim; strip_ghosts=writer.strip_ghosts))
    vort = _to_host(cell_center_vorticity(sim; strip_ghosts=writer.strip_ghosts))
    pres = _to_host(cell_center_pressure(sim; strip_ghosts=writer.strip_ghosts))
    time = sim_time(sim)
    jldopen(writer.filename, writer.samples == 0 ? "w" : "a") do file
        group = "snapshot_$(writer.samples + 1)"
        file["$group/time"] = time
        file["$group/velocity"] = vel
        file["$group/vorticity"] = vort
        file["$group/pressure"] = pres
    end
end

# =============================================================================
# FORCE COEFFICIENT WRITER
# =============================================================================
# Writes lift and drag coefficients to a JLD2 file at specified intervals.
# Data is stored as arrays that grow with each save, making it easy to
# load and plot the entire time history.
# =============================================================================

"""
    ForceWriter(filename::AbstractString="force_coefficients.jld2";
                interval::Real=0.1,
                overwrite::Bool=true,
                reference_area::Real=1.0)

Helper that saves lift and drag coefficients to a JLD2 file at fixed
convective-time intervals. Call [`file_save!`](@ref) after each `sim_step!`
to trigger writes. Uses the simulation's density (sim.flow.ρ) for coefficients.

# Arguments
- `filename`: Output JLD2 file path (default: "force_coefficients.jld2")
- `interval`: Time interval between saves (default: 0.1)
- `overwrite`: If true, overwrite existing file; if false, append (default: true)
- `reference_area`: Reference area for coefficient calculation (default: 1.0,
  typically set to sim.L for 2D simulations)

# Output Format
JLD2 file containing:
- `time`: Vector of simulation times
- `Cd`: Vector of total drag coefficients
- `Cl`: Vector of total lift coefficients
- `Cd_pressure`: Vector of pressure drag coefficients
- `Cd_viscous`: Vector of viscous drag coefficients
- `Cl_pressure`: Vector of pressure lift coefficients
- `Cl_viscous`: Vector of viscous lift coefficients

# Example
```julia
# Create simulation
sim = Simulation((128, 128), (1.0, 0.0), 1.0; ν=0.001, body=AutoBody(sdf))

# Create force writer (saves every 0.1 time units)
force_writer = ForceWriter("forces.jld2"; interval=0.1, reference_area=sim.L)

# Time stepping loop
for _ in 1:1000
    sim_step!(sim)
    file_save!(force_writer, sim)
end
```

# Reading the Output
```julia
using JLD2
data = load("forces.jld2")
time = data["time"]
Cd = data["Cd"]
Cl = data["Cl"]

# Or load individual arrays
time = load("forces.jld2", "time")
```
"""
mutable struct ForceWriter
    filename::String
    interval::Float64
    next_time::Float64
    samples::Int
    reference_area::Float64
    # Internal storage for accumulating data before writing
    time_history::Vector{Float64}
    Cd_history::Vector{Float64}
    Cl_history::Vector{Float64}
    Cd_pressure_history::Vector{Float64}
    Cd_viscous_history::Vector{Float64}
    Cl_pressure_history::Vector{Float64}
    Cl_viscous_history::Vector{Float64}

    function ForceWriter(filename::AbstractString="force_coefficients.jld2";
                         interval::Real=0.1,
                         overwrite::Bool=true,
                         reference_area::Real=1.0)
        interval > 0 || throw(ArgumentError("interval must be positive"))
        reference_area > 0 || throw(ArgumentError("reference_area must be positive"))

        # Handle file creation/overwrite
        if overwrite && isfile(filename)
            rm(filename)
        end

        time_hist = Float64[]
        Cd_hist = Float64[]
        Cl_hist = Float64[]
        Cd_p_hist = Float64[]
        Cd_v_hist = Float64[]
        Cl_p_hist = Float64[]
        Cl_v_hist = Float64[]
        samples = 0
        next_time = float(interval)

        if !overwrite && isfile(filename)
            try
                jldopen(filename, "r") do file
                    required = ("time", "Cd", "Cl", "Cd_pressure", "Cd_viscous",
                                "Cl_pressure", "Cl_viscous")
                    all(haskey(file, key) for key in required) || return
                    time_hist = Float64.(file["time"])
                    Cd_hist = Float64.(file["Cd"])
                    Cl_hist = Float64.(file["Cl"])
                    Cd_p_hist = Float64.(file["Cd_pressure"])
                    Cd_v_hist = Float64.(file["Cd_viscous"])
                    Cl_p_hist = Float64.(file["Cl_pressure"])
                    Cl_v_hist = Float64.(file["Cl_viscous"])
                    lengths = (length(time_hist), length(Cd_hist), length(Cl_hist),
                               length(Cd_p_hist), length(Cd_v_hist), length(Cl_p_hist),
                               length(Cl_v_hist))
                    min_len = minimum(lengths)
                    if min_len == 0
                        time_hist = Float64[]
                        Cd_hist = Float64[]
                        Cl_hist = Float64[]
                        Cd_p_hist = Float64[]
                        Cd_v_hist = Float64[]
                        Cl_p_hist = Float64[]
                        Cl_v_hist = Float64[]
                        return
                    end
                    time_hist = time_hist[1:min_len]
                    Cd_hist = Cd_hist[1:min_len]
                    Cl_hist = Cl_hist[1:min_len]
                    Cd_p_hist = Cd_p_hist[1:min_len]
                    Cd_v_hist = Cd_v_hist[1:min_len]
                    Cl_p_hist = Cl_p_hist[1:min_len]
                    Cl_v_hist = Cl_v_hist[1:min_len]
                end
            catch
                time_hist = Float64[]
                Cd_hist = Float64[]
                Cl_hist = Float64[]
                Cd_p_hist = Float64[]
                Cd_v_hist = Float64[]
                Cl_p_hist = Float64[]
                Cl_v_hist = Float64[]
            end
        end

        samples = length(time_hist)
        next_time = samples == 0 ? float(interval) : time_hist[end] + float(interval)

        return new(String(filename), float(interval), next_time, samples,
                   float(reference_area),
                   time_hist, Cd_hist, Cl_hist, Cd_p_hist, Cd_v_hist, Cl_p_hist, Cl_v_hist)
    end
end

"""
    file_save!(writer::ForceWriter, sim)

Check the simulation time and, if the configured interval has elapsed, append
the current lift and drag coefficients to the writer's JLD2 file.

Returns the writer for chaining.
"""
function file_save!(writer::ForceWriter, sim::AbstractSimulation)
    t = sim_time(sim)
    if t + eps(writer.interval) < writer.next_time
        return writer
    end
    while t + eps(writer.interval) >= writer.next_time
        _write_forces!(writer, sim)
        writer.samples += 1
        writer.next_time += writer.interval
    end
    return writer
end

function _write_forces!(writer::ForceWriter, sim::AbstractSimulation)
    # Compute force coefficients using simulation's density (sim.flow.ρ)
    # Use sim.L as reference area if writer.reference_area is 1.0 (default)
    ref_area = writer.reference_area == 1.0 ? sim.L : writer.reference_area
    components = force_components(sim; reference_area=ref_area)

    t = sim_time(sim)

    # Extract coefficients (pressure, viscous, total)
    if components.coefficients !== nothing
        Cd_p = components.coefficients[1][1]
        Cl_p = length(components.coefficients[1]) >= 2 ? components.coefficients[1][2] : 0.0
        Cd_v = components.coefficients[2][1]
        Cl_v = length(components.coefficients[2]) >= 2 ? components.coefficients[2][2] : 0.0
        Cd = components.coefficients[3][1]
        Cl = length(components.coefficients[3]) >= 2 ? components.coefficients[3][2] : 0.0
    else
        # Fallback to raw forces if coefficients not available
        Cd_p = components.pressure[1]
        Cl_p = length(components.pressure) >= 2 ? components.pressure[2] : 0.0
        Cd_v = components.viscous[1]
        Cl_v = length(components.viscous) >= 2 ? components.viscous[2] : 0.0
        Cd = components.total[1]
        Cl = length(components.total) >= 2 ? components.total[2] : 0.0
    end

    # Append to internal history
    push!(writer.time_history, t)
    push!(writer.Cd_history, Cd)
    push!(writer.Cl_history, Cl)
    push!(writer.Cd_pressure_history, Cd_p)
    push!(writer.Cd_viscous_history, Cd_v)
    push!(writer.Cl_pressure_history, Cl_p)
    push!(writer.Cl_viscous_history, Cl_v)

    # Write to JLD2 file (overwrite with full history each time)
    jldopen(writer.filename, "w") do file
        file["time"] = writer.time_history
        file["Cd"] = writer.Cd_history
        file["Cl"] = writer.Cl_history
        file["Cd_pressure"] = writer.Cd_pressure_history
        file["Cd_viscous"] = writer.Cd_viscous_history
        file["Cl_pressure"] = writer.Cl_pressure_history
        file["Cl_viscous"] = writer.Cl_viscous_history
    end
end

"""
    reset!(writer::ForceWriter)

Reset the writer to start fresh. Does not delete the existing file.
"""
function reset!(writer::ForceWriter)
    writer.samples = 0
    writer.next_time = writer.interval
    empty!(writer.time_history)
    empty!(writer.Cd_history)
    empty!(writer.Cl_history)
    empty!(writer.Cd_pressure_history)
    empty!(writer.Cd_viscous_history)
    empty!(writer.Cl_pressure_history)
    empty!(writer.Cl_viscous_history)
end

# =============================================================================
# GRID COORDINATE UTILITIES
# =============================================================================
# Functions for computing and saving cell-center coordinates.
# Grid files are essential for visualization tools like ParaView, VisIt, etc.
# =============================================================================

"""
    cell_center_coordinates(sim; strip_ghosts=true)

Compute the cell-center coordinates for the simulation grid.

# Returns
For 2D simulations: `(x, z)` where each is a 1D array of coordinates
For 3D simulations: `(x, y, z)` where each is a 1D array of coordinates

# Example
```julia
x, z = cell_center_coordinates(sim)  # 2D
x, y, z = cell_center_coordinates(sim)  # 3D
```
"""
function cell_center_coordinates(sim::AbstractSimulation; strip_ghosts::Bool=true)
    flow = sim.flow
    Δx = flow.Δx
    p_size = size(flow.p)
    spatial_dims = ndims(flow.p)

    if strip_ghosts
        # Interior dimensions (excluding ghost cells)
        dims = ntuple(i -> p_size[i] - 2, spatial_dims)
        # Cell centers start at Δx/2 from origin
        coords = ntuple(spatial_dims) do i
            n = dims[i]
            dx = Δx[i]
            collect(range(dx/2, step=dx, length=n))
        end
    else
        # Full dimensions including ghost cells
        dims = p_size
        # Ghost cells extend before origin
        coords = ntuple(spatial_dims) do i
            n = dims[i]
            dx = Δx[i]
            # First ghost cell center at -Δx/2
            collect(range(-dx/2, step=dx, length=n))
        end
    end

    return coords
end

"""
    cell_center_coordinates_meshgrid(sim; strip_ghosts=true)

Compute the cell-center coordinates as full meshgrid arrays.
Each output array has the same shape as the pressure field.

# Returns
For 2D: `(X, Z)` where X[i,j] and Z[i,j] give coordinates of cell (i,j)
For 3D: `(X, Y, Z)` where X[i,j,k], Y[i,j,k], Z[i,j,k] give coordinates

# Example
```julia
X, Z = cell_center_coordinates_meshgrid(sim)  # 2D
```
"""
function cell_center_coordinates_meshgrid(sim::AbstractSimulation; strip_ghosts::Bool=true)
    coords = cell_center_coordinates(sim; strip_ghosts=strip_ghosts)
    spatial_dims = length(coords)

    if spatial_dims == 2
        x, z = coords
        nx, nz = length(x), length(z)
        X = [x[i] for i in 1:nx, j in 1:nz]
        Z = [z[j] for i in 1:nx, j in 1:nz]
        return (X, Z)
    else  # 3D
        x, y, z = coords
        nx, ny, nz = length(x), length(y), length(z)
        X = [x[i] for i in 1:nx, j in 1:ny, k in 1:nz]
        Y = [y[j] for i in 1:nx, j in 1:ny, k in 1:nz]
        Z = [z[k] for i in 1:nx, j in 1:ny, k in 1:nz]
        return (X, Y, Z)
    end
end

"""
    save_grid(filename::AbstractString, sim::AbstractSimulation;
              strip_ghosts::Bool=true, format::Symbol=:jld2)

Save the grid coordinates to a file.

# Arguments
- `filename`: Output file path
- `sim`: Simulation object
- `strip_ghosts`: If true, exclude ghost cells (default: true)
- `format`: Output format, `:jld2` (default) or `:vtk`

# JLD2 Output Format
The file contains:
- `x`, `z` (2D) or `x`, `y`, `z` (3D): 1D coordinate arrays
- `X`, `Z` (2D) or `X`, `Y`, `Z` (3D): Full meshgrid arrays
- `dx`, `dz` (2D) or `dx`, `dy`, `dz` (3D): Grid spacing
- `nx`, `nz` (2D) or `nx`, `ny`, `nz` (3D): Grid dimensions
- `ndims`: Number of spatial dimensions

# Example
```julia
save_grid("grid.jld2", sim)

# Load later
using JLD2
grid = load("grid.jld2")
x, z = grid["x"], grid["z"]
```
"""
function save_grid(filename::AbstractString, sim::AbstractSimulation;
                   strip_ghosts::Bool=true, format::Symbol=:jld2)
    if format == :jld2
        _save_grid_jld2(filename, sim; strip_ghosts=strip_ghosts)
    elseif format == :vtk
        _save_grid_vtk(filename, sim; strip_ghosts=strip_ghosts)
    else
        throw(ArgumentError("Unknown format: $format. Use :jld2 or :vtk"))
    end
end

function _save_grid_jld2(filename::AbstractString, sim::AbstractSimulation;
                          strip_ghosts::Bool=true)
    coords = cell_center_coordinates(sim; strip_ghosts=strip_ghosts)
    meshgrid = cell_center_coordinates_meshgrid(sim; strip_ghosts=strip_ghosts)
    Δx = sim.flow.Δx
    spatial_dims = length(coords)

    jldopen(filename, "w") do file
        file["ndims"] = spatial_dims

        if spatial_dims == 2
            file["x"] = coords[1]
            file["z"] = coords[2]
            file["X"] = meshgrid[1]
            file["Z"] = meshgrid[2]
            file["dx"] = Δx[1]
            file["dz"] = Δx[2]
            file["nx"] = length(coords[1])
            file["nz"] = length(coords[2])
        else  # 3D
            file["x"] = coords[1]
            file["y"] = coords[2]
            file["z"] = coords[3]
            file["X"] = meshgrid[1]
            file["Y"] = meshgrid[2]
            file["Z"] = meshgrid[3]
            file["dx"] = Δx[1]
            file["dy"] = Δx[2]
            file["dz"] = Δx[3]
            file["nx"] = length(coords[1])
            file["ny"] = length(coords[2])
            file["nz"] = length(coords[3])
        end
    end
end

function _save_grid_vtk(filename::AbstractString, sim::AbstractSimulation;
                         strip_ghosts::Bool=true)
    coords = cell_center_coordinates(sim; strip_ghosts=strip_ghosts)
    spatial_dims = length(coords)

    # Ensure .vtk extension
    if !endswith(filename, ".vtk")
        filename = filename * ".vtk"
    end

    open(filename, "w") do io
        # VTK legacy header
        println(io, "# vtk DataFile Version 3.0")
        println(io, "BioFlows Grid - Cell Centers")
        println(io, "ASCII")

        if spatial_dims == 2
            x, z = coords
            nx, nz = length(x), length(z)

            println(io, "DATASET RECTILINEAR_GRID")
            println(io, "DIMENSIONS $nx $nz 1")

            println(io, "X_COORDINATES $nx float")
            for xi in x
                println(io, xi)
            end

            println(io, "Y_COORDINATES $nz float")
            for zi in z
                println(io, zi)
            end

            println(io, "Z_COORDINATES 1 float")
            println(io, "0.0")
        else  # 3D
            x, y, z = coords
            nx, ny, nz = length(x), length(y), length(z)

            println(io, "DATASET RECTILINEAR_GRID")
            println(io, "DIMENSIONS $nx $ny $nz")

            println(io, "X_COORDINATES $nx float")
            for xi in x
                println(io, xi)
            end

            println(io, "Y_COORDINATES $ny float")
            for yi in y
                println(io, yi)
            end

            println(io, "Z_COORDINATES $nz float")
            for zi in z
                println(io, zi)
            end
        end
    end
end

"""
    GridWriter(filename::AbstractString="grid.jld2";
               strip_ghosts::Bool=true)

Helper that saves grid coordinates once at the beginning of a simulation.
Call `file_save!` to write the grid file.

# Example
```julia
grid_writer = GridWriter("grid.jld2")
file_save!(grid_writer, sim)  # Call once at start
```
"""
mutable struct GridWriter
    filename::String
    strip_ghosts::Bool
    saved::Bool

    function GridWriter(filename::AbstractString="grid.jld2";
                        strip_ghosts::Bool=true)
        return new(String(filename), strip_ghosts, false)
    end
end

"""
    file_save!(writer::GridWriter, sim)

Save the grid coordinates if not already saved. Returns the writer.
"""
function file_save!(writer::GridWriter, sim::AbstractSimulation)
    if !writer.saved
        save_grid(writer.filename, sim; strip_ghosts=writer.strip_ghosts)
        writer.saved = true
    end
    return writer
end
