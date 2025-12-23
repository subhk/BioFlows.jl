using Test
using BioFlows

@testset "Finite Volume Method" begin

    @testset "Flux Storage Creation" begin
        # Test 1: Flux arrays are created when store_fluxes=true
        flow = Flow((32, 32); L=(32.0, 32.0), store_fluxes=true, T=Float64)
        @test !isnothing(flow.F_conv)
        @test !isnothing(flow.F_diff)
        @test flow.store_fluxes == true
        @test size(flow.F_conv) == (34, 34, 2, 2)  # Ng x Ng x D x D
        @test size(flow.F_diff) == (34, 34, 2, 2)

        # Test 2: Flux arrays are NOT created by default
        flow_default = Flow((32, 32); L=(32.0, 32.0), T=Float64)
        @test isnothing(flow_default.F_conv)
        @test isnothing(flow_default.F_diff)
        @test flow_default.store_fluxes == false
    end

    @testset "Simulation with FVM" begin
        # Test 3: Simulation accepts store_fluxes parameter
        sim = BioFlows.Simulation((32, 32), (32.0, 32.0);
                                   inletBC=(1.0, 0.0), ν=0.01, store_fluxes=true,
                                   T=Float64, L_char=8.0)
        @test sim.flow.store_fluxes == true
        @test !isnothing(sim.flow.F_conv)

        # Test 4: FVM simulation step works
        sim_step!(sim; remeasure=false)
        @test sim_time(sim) > 0

        # Test 5: Fluxes are populated after step
        @test maximum(abs, sim.flow.F_conv) > 0 || maximum(abs, sim.flow.F_diff) > 0
    end

    @testset "FVM vs Original Consistency" begin
        # Test 6: FVM gives similar results to original method
        # Create two identical simulations
        sim_orig = BioFlows.Simulation((32, 32), (32.0, 32.0);
                                        inletBC=(1.0, 0.0), ν=0.01, store_fluxes=false,
                                        T=Float64, L_char=8.0)
        sim_fvm = BioFlows.Simulation((32, 32), (32.0, 32.0);
                                       inletBC=(1.0, 0.0), ν=0.01, store_fluxes=true,
                                       T=Float64, L_char=8.0)

        # Take a few steps
        for _ in 1:3
            sim_step!(sim_orig; remeasure=false)
            sim_step!(sim_fvm; remeasure=false)
        end

        # Results should be very close (not identical due to different code paths)
        @test isapprox(sim_orig.flow.u, sim_fvm.flow.u, rtol=1e-10)
        @test isapprox(sim_orig.flow.p, sim_fvm.flow.p, rtol=1e-10)
    end

    @testset "3D FVM" begin
        # Test 7: 3D case works
        flow3d = Flow((16, 16, 16); L=(16.0, 16.0, 16.0), store_fluxes=true, T=Float64)
        @test size(flow3d.F_conv) == (18, 18, 18, 3, 3)  # Ng x Ng x Ng x D x D
        @test size(flow3d.F_diff) == (18, 18, 18, 3, 3)
    end

    @testset "Flux Computation Functions" begin
        # Test 8: Direct flux computation
        n = 16
        T = Float64
        Ng = n + 2
        u = ones(T, Ng, Ng, 2)
        F_conv = zeros(T, Ng, Ng, 2, 2)
        F_diff = zeros(T, Ng, Ng, 2, 2)

        # Compute fluxes
        BioFlows.compute_face_flux!(F_conv, F_diff, u, BioFlows.quick;
                                     ν=0.01, Δx=(1.0, 1.0), perdir=())

        # For uniform velocity, convective fluxes should be uniform
        # and diffusive fluxes should be near zero
        @test all(isfinite, F_conv)
        @test all(isfinite, F_diff)
    end

end
