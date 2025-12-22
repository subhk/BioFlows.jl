using Test
using BioFlows
using JLD2

@testset "BioFlows example constructors" begin
    include(joinpath(@__DIR__, "..", "examples", "waterlily_circle.jl"))
    circle = circle_sim(; n=3*2^4, m=2^5, Re=50)
    @test isa(circle, BioFlows.Simulation)
    sim_step!(circle; remeasure=false)
    @test sim_time(circle) > 0

    include(joinpath(@__DIR__, "..", "examples", "flow_past_cylinder_2d.jl"))
    cyl, _ = flow_past_cylinder_2d_sim(; nx=3*2^4, nz=2^5, Re=120)
    sim_step!(cyl; remeasure=false)
    history = NamedTuple[]
    record_force!(history, cyl)
    stats = summarize_force_history(history; discard=0)
    @test !isnan(stats.drag_mean)

    tmp = mktempdir()
    writer = CenterFieldWriter(joinpath(tmp, "snapshots.jld2"); interval=0.05)
    for _ in 1:3
        sim_step!(circle; remeasure=false)
        maybe_save!(writer, circle)
    end
    @test isfile(writer.filename)
    jldopen(writer.filename, "r") do f
        @test haskey(f, "snapshot_1/time")
        vel = f["snapshot_1/velocity"]
        vort = f["snapshot_1/vorticity"]
        @test ndims(vel) == ndims(circle.flow.p) + 1
        @test size(vort)[1:ndims(circle.flow.p)] == size(vel)[1:ndims(circle.flow.p)]
    end

    include(joinpath(@__DIR__, "..", "examples", "waterlily_oscillating_cylinder.jl"))
    osc = oscillating_cylinder_sim(; n=3*2^4, m=2^5, Re=120, St=0.1, amplitude=0.15)
    @test isa(osc, BioFlows.Simulation)
    sim_step!(osc; remeasure=true)
    coeff = total_force(osc) ./ (0.5 * osc.L * osc.U^2)
    @test length(coeff) == 2

    include(joinpath(@__DIR__, "..", "examples", "waterlily_donut.jl"))
    donut = donut_sim(; n=2^5, Re=800, major_ratio=0.3, minor_ratio=0.08)
    @test isa(donut, BioFlows.Simulation)
    sim_step!(donut; remeasure=false)
    coeff3d = total_force(donut)
    @test length(coeff3d) == 3

    include(joinpath(@__DIR__, "..", "examples", "waterlily_3d_sphere.jl"))
    sphere = sphere_sim(; n=2^5, m=2^5, ℓ=2^5, Re=150)
    @test isa(sphere, BioFlows.Simulation)
    sim_step!(sphere; remeasure=false)
    @test sim_time(sphere) > 0
end
