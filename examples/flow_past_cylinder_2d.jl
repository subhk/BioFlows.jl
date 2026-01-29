using BioFlows
using Statistics

"""
    flow_past_cylinder_2d_sim(; nx=600, nz=240, Lx=10f0, Lz=4f0, ν=0.001f0, ...)

Construct the classic 2D cylinder benchmark using physical coordinates.
Returns `(sim, meta)`.

The cylinder is defined using an SDF in **physical coordinates** (meters).
The center position and radius are specified in physical units.
"""
function flow_past_cylinder_2d_sim(; nx::Int=600, nz::Int=240,
                                    Lx::Real=10f0,
                                    Lz::Real=4f0,
                                    ν::Real=0.001f0,
                                    U::Real=1f0,
                                    radius::Union{Nothing,Real}=nothing,
                                    dt=nothing,
                                    inletBC=nothing,
                                    perdir=(2,),
                                    outletBC::Bool=true)
    # Compute physical grid spacing
    dx = Lx / nx
    dz = Lz / nz
    @assert isapprox(dx, dz; atol=1e-8, rtol=1e-6) "Non-uniform cell spacing (Δx ≠ Δz) is not supported"

    # Cylinder parameters in PHYSICAL coordinates (meters)
    radius_phys = isnothing(radius) ? 0.1f0 : Float32(radius)
    center_x_phys = Lx / 12  # Physical x position of cylinder center
    center_z_phys = Lz / 2   # Physical z position of cylinder center (centered in z)

    # SDF using PHYSICAL coordinates
    # x is passed in physical units (meters) by the solver
    sdf(x, t) = √((x[1] - center_x_phys)^2 + (x[2] - center_z_phys)^2) - radius_phys

    boundary = isnothing(inletBC) ? (U, 0) : inletBC

    diameter = 2radius_phys
    base_kwargs = (; ν = ν, perdir = perdir,
                    outletBC = outletBC,
                    body = AutoBody(sdf),
                    L_char = diameter)

    sim = dt === nothing ?
        Simulation((nx, nz), (Lx, Lz); inletBC=boundary, base_kwargs...) :
        Simulation((nx, nz), (Lx, Lz); inletBC=boundary, Δt = dt, base_kwargs...)

    meta = (domain = (Lx, Lz), grid = (nx, nz), cell_size = (dx, dz),
            center = (center_x_phys, center_z_phys), radius = radius_phys,
            Δt = sim.flow.Δt[end], fixed_dt = dt,
            inletBC = boundary, perdir = perdir, outletBC = outletBC, ν = ν, U = U)

    return sim, meta
end

"""
    run_cylinder_simulation(; nx=150, nz=60, n_steps=1000, save_interval=0.1,
                             output_dir="output", kwargs...)

Run the flow past cylinder simulation with periodic output using BioFlows'
built-in CenterFieldWriter and ForceWriter.

# Arguments
- `nx, nz`: Grid resolution
- `n_steps`: Total number of time steps
- `save_interval`: Time interval between saves (default: 0.1 convective time units)
- `output_dir`: Directory for output files
- `kwargs...`: Additional arguments passed to `flow_past_cylinder_2d_sim`

# Output Files
- `output_dir/fields.jld2`: Velocity, pressure, vorticity snapshots
- `output_dir/fields_grid.jld2`: Grid coordinates
- `output_dir/forces.jld2`: Force coefficient history (Cd, Cl with pressure/viscous breakdown)
"""
function run_cylinder_simulation(; nx::Int=150, nz::Int=60,
                                   n_steps::Int=1000,
                                   save_interval::Real=0.1,
                                   output_dir::String="output",
                                   kwargs...)
    # Create output directory
    mkpath(output_dir)

    @info "Setting up flow past cylinder simulation" nx=nx nz=nz n_steps=n_steps save_interval=save_interval

    # Create simulation
    sim, meta = flow_past_cylinder_2d_sim(; nx=nx, nz=nz, kwargs...)

    # Save metadata and grid info as text
    grid_file = joinpath(output_dir, "grid_info.txt")
    open(grid_file, "w") do io
        println(io, "# Flow Past Cylinder 2D - Grid Information")
        println(io, "# Domain size: Lx=$(meta.domain[1]), Lz=$(meta.domain[2])")
        println(io, "# Grid: nx=$(meta.grid[1]), nz=$(meta.grid[2])")
        println(io, "# Cell size: dx=$(meta.cell_size[1]), dz=$(meta.cell_size[2])")
        println(io, "# Cylinder center: x=$(meta.center[1]), z=$(meta.center[2])")
        println(io, "# Cylinder radius: $(meta.radius)")
        println(io, "# Viscosity: $(meta.ν)")
        println(io, "# Inlet velocity: $(meta.inletBC)")
        println(io, "# Reynolds number: $(meta.inletBC[1] * 2 * meta.radius / meta.ν)")
    end

    # Set up BioFlows output writers
    field_writer = CenterFieldWriter(
        joinpath(output_dir, "fields.jld2");
        interval = save_interval,
        overwrite = true,
        save_grid = true
    )

    force_writer = ForceWriter(
        joinpath(output_dir, "forces.jld2");
        interval = save_interval,
        overwrite = true,
        reference_area = sim.L  # Use characteristic length (diameter)
    )

    @info "Starting simulation loop..."

    for iter in 1:n_steps
        # Advance simulation
        sim_step!(sim; remeasure=false)

        # Use BioFlows built-in writers (automatically save at time intervals)
        file_save!(field_writer, sim)
        file_save!(force_writer, sim)

        # Log progress periodically
        if iter % 100 == 0 || iter == n_steps
            forces = force_components(sim)
            Cd, Cl = forces.coefficients[3]
            @info "Iteration $iter / $n_steps" time=sim_time(sim) Cd=Cd Cl=Cl
        end
    end

    # Compute final statistics from force writer history
    n_samples = length(force_writer.Cd_history)
    discard = min(n_samples ÷ 2, 100)
    if n_samples > discard
        Cd_trimmed = force_writer.Cd_history[discard+1:end]
        Cl_trimmed = force_writer.Cl_history[discard+1:end]
        @info "Simulation complete" Cd_mean=mean(Cd_trimmed) Cd_std=std(Cd_trimmed) Cl_rms=sqrt(mean(abs2, Cl_trimmed))
    else
        @info "Simulation complete" samples=n_samples
    end

    return sim, meta, force_writer
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    @info "Running flow past cylinder 2D example" threads=Threads.nthreads() backend=BioFlows.backend

    # Run simulation with output
    # Note: nx/nz ratio must match Lx/Lz = 10/4 = 2.5 for uniform cell spacing
    sim, meta, force_writer = run_cylinder_simulation(
        nx = 250,
        nz = 100,
        n_steps = 500,
        save_interval = 0.1,  # Save every 0.1 convective time units
        output_dir = "cylinder_output"
    )

    @info "Output files saved to cylinder_output/"
    @info "  - fields.jld2: Velocity, vorticity, pressure snapshots"
    @info "  - fields_grid.jld2: Grid coordinates"
    @info "  - forces.jld2: Cd, Cl time history"
end
