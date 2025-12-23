# =============================================================================
# SIMULATION OUTPUT UTILITIES
# =============================================================================
# This module provides tools for saving simulation results to disk:
#
# - CenterFieldWriter: Periodic snapshot writer to JLD2 files
# - ForceWriter: Periodic writer for lift/drag coefficients to CSV files
# - maybe_save!: Conditional snapshot saving based on time intervals
#
# Output files use JLD2 format (Julia Data Format v2) which can be read back
# using the JLD2.jl package:
#   using JLD2
#   data = load("center_fields.jld2")
#   vel = data["snapshot_1/velocity"]
#
# Force coefficient files use CSV format for easy import into plotting tools:
#   time, Cd, Cl, Cd_pressure, Cd_viscous, Cl_pressure, Cl_viscous
#
# Snapshots include:
# - Velocity (cell-centered, [nx, nz, 2] for 2D or [nx, ny, nz, 3] for 3D)
# - Vorticity (cell-centered, scalar for 2D or vector for 3D)
# - Pressure (cell-centered)
# =============================================================================

using JLD2
using Printf

"""
    CenterFieldWriter(filename::AbstractString="center_fields.jld2";
                      interval::Real=0.1,
                      overwrite::Bool=true,
                      strip_ghosts::Bool=true)

Helper that saves cell-centred velocity, vorticity, and pressure fields to a
JLD2 file at fixed convective-time intervals. Call [`maybe_save!`](@ref) after
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
    maybe_save!(writer, sim)

Check the simulation time and, if the configured interval has elapsed, append a
snapshot with cell-centred velocity, vorticity, and pressure to the writer's
JLD2 file.
"""
function maybe_save!(writer::CenterFieldWriter, sim::AbstractSimulation)
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
# Writes lift and drag coefficients to a CSV file at specified intervals.
# The CSV format allows easy import into plotting tools (Python, MATLAB, Excel).
# =============================================================================

"""
    ForceWriter(filename::AbstractString="force_coefficients.csv";
                interval::Real=0.1,
                overwrite::Bool=true,
                ρ::Real=1.0,
                reference_area::Real=1.0)

Helper that saves lift and drag coefficients to a CSV file at fixed
convective-time intervals. Call [`maybe_save!`](@ref) after each `sim_step!`
to trigger writes.

# Arguments
- `filename`: Output CSV file path (default: "force_coefficients.csv")
- `interval`: Time interval between saves (default: 0.1)
- `overwrite`: If true, overwrite existing file; if false, append (default: true)
- `ρ`: Fluid density for coefficient calculation (default: 1.0)
- `reference_area`: Reference area for coefficient calculation (default: 1.0,
  typically set to sim.L for 2D simulations)

# Output Format
CSV file with columns:
- `time`: Simulation time
- `Cd`: Total drag coefficient
- `Cl`: Total lift coefficient
- `Cd_pressure`: Pressure drag coefficient
- `Cd_viscous`: Viscous drag coefficient
- `Cl_pressure`: Pressure lift coefficient
- `Cl_viscous`: Viscous lift coefficient

# Example
```julia
# Create simulation
sim = Simulation((128, 128), (1.0, 0.0), 1.0; ν=0.001, body=AutoBody(sdf))

# Create force writer (saves every 0.1 time units)
force_writer = ForceWriter("forces.csv"; interval=0.1, reference_area=sim.L)

# Time stepping loop
for _ in 1:1000
    sim_step!(sim)
    maybe_save!(force_writer, sim)
end
```

# Reading the Output
```julia
using DelimitedFiles
data = readdlm("forces.csv", ',', Float64; skipstart=1)
time, Cd, Cl = data[:,1], data[:,2], data[:,3]
```
"""
mutable struct ForceWriter
    filename::String
    interval::Float64
    next_time::Float64
    samples::Int
    ρ::Float64
    reference_area::Float64
    io::Union{IOStream, Nothing}

    function ForceWriter(filename::AbstractString="force_coefficients.csv";
                         interval::Real=0.1,
                         overwrite::Bool=true,
                         ρ::Real=1.0,
                         reference_area::Real=1.0)
        interval > 0 || throw(ArgumentError("interval must be positive"))
        ρ > 0 || throw(ArgumentError("density ρ must be positive"))
        reference_area > 0 || throw(ArgumentError("reference_area must be positive"))

        # Handle file creation/overwrite
        if overwrite && isfile(filename)
            rm(filename)
        end

        return new(String(filename), float(interval), float(interval), 0,
                   float(ρ), float(reference_area), nothing)
    end
end

"""
    maybe_save!(writer::ForceWriter, sim)

Check the simulation time and, if the configured interval has elapsed, append
the current lift and drag coefficients to the writer's CSV file.

Returns the writer for chaining.
"""
function maybe_save!(writer::ForceWriter, sim::AbstractSimulation)
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
        Cd_p, Cl_p = components.coefficients[1][1], length(components.coefficients[1]) >= 2 ? components.coefficients[1][2] : 0.0
        Cd_v, Cl_v = components.coefficients[2][1], length(components.coefficients[2]) >= 2 ? components.coefficients[2][2] : 0.0
        Cd, Cl = components.coefficients[3][1], length(components.coefficients[3]) >= 2 ? components.coefficients[3][2] : 0.0
    else
        # Fallback to raw forces if coefficients not available
        Cd_p, Cl_p = components.pressure[1], length(components.pressure) >= 2 ? components.pressure[2] : 0.0
        Cd_v, Cl_v = components.viscous[1], length(components.viscous) >= 2 ? components.viscous[2] : 0.0
        Cd, Cl = components.total[1], length(components.total) >= 2 ? components.total[2] : 0.0
    end

    # Open file and write header if first sample
    open(writer.filename, writer.samples == 0 ? "w" : "a") do io
        if writer.samples == 0
            println(io, "time,Cd,Cl,Cd_pressure,Cd_viscous,Cl_pressure,Cl_viscous")
        end
        # Write data with high precision
        @printf(io, "%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                t, Cd, Cl, Cd_p, Cd_v, Cl_p, Cl_v)
    end
end

"""
    close!(writer::ForceWriter)

Close the ForceWriter's file handle if open.
"""
function close!(writer::ForceWriter)
    if writer.io !== nothing
        close(writer.io)
        writer.io = nothing
    end
end

"""
    reset!(writer::ForceWriter)

Reset the writer to start fresh. Does not delete the existing file.
"""
function reset!(writer::ForceWriter)
    close!(writer)
    writer.samples = 0
    writer.next_time = writer.interval
end
