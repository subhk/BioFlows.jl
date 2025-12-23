using Test
using BioFlows
using JLD2

@testset "BioFlows example constructors" begin
    include(joinpath(@__DIR__, "..", "examples", "circle_benchmark.jl"))
    circle = circle_sim(; n=3*2^4, m=2^5, ν=0.16)
    @test isa(circle, BioFlows.Simulation)
    sim_step!(circle; remeasure=false)
    @test sim_time(circle) > 0

    include(joinpath(@__DIR__, "..", "examples", "flow_past_cylinder_2d.jl"))
    cyl, _ = flow_past_cylinder_2d_sim(; nx=48, nz=48, ν=0.003)
    sim_step!(cyl; remeasure=false)
    history = NamedTuple[]
    record_force!(history, cyl)
    stats = summarize_force_history(history; discard=0)
    @test !isnan(stats.drag_mean)

    tmp = mktempdir()
    writer = CenterFieldWriter(joinpath(tmp, "snapshots.jld2"); interval=0.05)
    for _ in 1:3
        sim_step!(circle; remeasure=false)
        file_save!(writer, circle)
    end
    @test isfile(writer.filename)
    jldopen(writer.filename, "r") do f
        @test haskey(f, "snapshot_1/time")
        vel = f["snapshot_1/velocity"]
        vort = f["snapshot_1/vorticity"]
        @test ndims(vel) == ndims(circle.flow.p) + 1
        @test size(vort)[1:ndims(circle.flow.p)] == size(vel)[1:ndims(circle.flow.p)]
    end

    include(joinpath(@__DIR__, "..", "examples", "oscillating_cylinder.jl"))
    osc = oscillating_cylinder_sim(; n=3*2^4, m=2^5, ν=0.067, St=0.1, amplitude=0.15)
    @test isa(osc, BioFlows.Simulation)
    sim_step!(osc; remeasure=true)
    coeff = total_force(osc) ./ (0.5 * osc.L * osc.U^2)
    @test length(coeff) == 2

    include(joinpath(@__DIR__, "..", "examples", "torus_3d.jl"))
    donut = donut_sim(; n=2^4, ν=0.028, major_ratio=0.3, minor_ratio=0.08)  # Reduced from 2^5 for CI
    @test isa(donut, BioFlows.Simulation)
    sim_step!(donut; remeasure=false)
    coeff3d = total_force(donut)
    @test length(coeff3d) == 3

    include(joinpath(@__DIR__, "..", "examples", "sphere_3d.jl"))
    sphere = sphere_sim(; n=2^4, m=2^4, ℓ=2^4, ν=0.053)  # Reduced from 2^5 for CI
    @test isa(sphere, BioFlows.Simulation)
    sim_step!(sphere; remeasure=false)
    @test sim_time(sphere) > 0
end

# Include AMR tests
include("test_amr.jl")

# Include pressure solver tests
include("test_pressure_solver.jl")

# Include composite Poisson solver tests
include("test_composite_poisson.jl")
