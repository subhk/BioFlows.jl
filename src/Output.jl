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
