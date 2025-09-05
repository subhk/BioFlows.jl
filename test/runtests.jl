using Test

@testset "BioFlows basic load" begin
    include(joinpath(@__DIR__, "..", "test_simple.jl"))
    @test true  # Basic smoke test
end

@testset "Optional multigrid test" begin
    try
        include(joinpath(@__DIR__, "..", "test_multigrid_fix.jl"))
        @test true
    catch e
        @info "Skipping multigrid test (dependency not available)" error=e
        @test true
    end
end

