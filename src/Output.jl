# =============================================================================
# SIMULATION OUTPUT UTILITIES
# =============================================================================
# This module provides tools for saving simulation results to disk:
#
# - CenterFieldWriter: Periodic snapshot writer to JLD2 files
# - ForceWriter: Periodic writer for lift/drag coefficients to CSV files
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

"""
    CenterFieldWriter(filename::AbstractString="center_fields.jld2";
                      interval::Real=0.1,
                      overwrite::Bool=true,
                      strip_ghosts::Bool=true)

Helper that saves cell-centred velocity, vorticity, and pressure fields to a
JLD2 file at fixed convective-time intervals. Call [`file_save!`](@ref) after
each `sim_step!` to trigger writes.
"""
mutable struct CenterFieldWriter
    filename::String
    interval::Float64
    next_time::Float64
    samples::Int
    strip_ghosts::Bool
    function CenterFieldWriter(filename::AbstractString="center_fields.jld2";
                               interval::Real=0.1,
                               overwrite::Bool=true,
                               strip_ghosts::Bool=true)
        interval > 0 || throw(ArgumentError("interval must be positive"))
        overwrite && isfile(filename) && rm(filename)
        return new(String(filename), float(interval), float(interval), 0, strip_ghosts)
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
    vel = cell_center_velocity(sim; strip_ghosts=writer.strip_ghosts)
    vort = cell_center_vorticity(sim; strip_ghosts=writer.strip_ghosts)
    pres = cell_center_pressure(sim; strip_ghosts=writer.strip_ghosts)
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
                ρ::Real=1000.0,
                reference_area::Real=1.0)

Helper that saves lift and drag coefficients to a JLD2 file at fixed
convective-time intervals. Call [`file_save!`](@ref) after each `sim_step!`
to trigger writes.

# Arguments
- `filename`: Output JLD2 file path (default: "force_coefficients.jld2")
- `interval`: Time interval between saves (default: 0.1)
- `overwrite`: If true, overwrite existing file; if false, append (default: true)
- `ρ`: Fluid density for coefficient calculation (default: 1000.0 kg/m³, water)
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
    ρ::Float64
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
                         ρ::Real=1000.0,
                         reference_area::Real=1.0)
        interval > 0 || throw(ArgumentError("interval must be positive"))
        ρ > 0 || throw(ArgumentError("density ρ must be positive"))
        reference_area > 0 || throw(ArgumentError("reference_area must be positive"))

        # Handle file creation/overwrite
        if overwrite && isfile(filename)
            rm(filename)
        end

        return new(String(filename), float(interval), float(interval), 0,
                   float(ρ), float(reference_area),
                   Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[])
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
    # Compute force coefficients
    # Use sim.L as reference area if writer.reference_area is 1.0 (default)
    ref_area = writer.reference_area == 1.0 ? sim.L : writer.reference_area
    components = force_components(sim; ρ=writer.ρ, reference_area=ref_area)

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
