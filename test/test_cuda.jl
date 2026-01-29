# =============================================================================
# CUDA GPU Tests for BioFlows.jl
# =============================================================================
# These tests verify GPU functionality when CUDA is available.
# Tests are skipped if CUDA is not functional.
# =============================================================================

using Test
using BioFlows

# Check if CUDA is available
cuda_available = false
try
    using CUDA
    cuda_available = CUDA.functional()
catch
    cuda_available = false
end

@testset "CUDA GPU Support" begin
    if !cuda_available
        @info "CUDA not available, skipping GPU tests"
        @test_skip "CUDA tests require functional CUDA"
    else
        @info "CUDA available, running GPU tests" device=CUDA.device()

        @testset "GPU Backend Detection" begin
            info = gpu_backend()
            @test info.available == true
            @test info.backend_name == "CUDA"
            @test info.array_type == CuArray
        end

        @testset "GPU Array Utilities" begin
            # Test gpu_sync! (should be no-op or synchronize)
            arr = CUDA.zeros(Float32, 10, 10)
            @test gpu_sync!(arr) === nothing

            # Test to_cpu
            gpu_arr = CUDA.ones(Float32, 5, 5)
            cpu_arr = to_cpu(gpu_arr)
            @test cpu_arr isa Array
            @test size(cpu_arr) == (5, 5)
            @test all(cpu_arr .== 1.0f0)
        end

        @testset "GPU Simulation Construction" begin
            # Small 2D simulation on GPU
            dims = (32, 32)
            L = (1.0, 1.0)
            sim = Simulation(dims, L;
                inletBC = (1.0f0, 0.0f0),
                ν = 0.01f0,
                mem = CuArray
            )
            @test sim isa BioFlows.Simulation
            @test sim.flow.u isa CuArray
            @test sim.flow.p isa CuArray
        end

        @testset "GPU Simulation Step" begin
            dims = (32, 32)
            L = (1.0, 1.0)
            sim = Simulation(dims, L;
                inletBC = (1.0f0, 0.0f0),
                ν = 0.01f0,
                mem = CuArray
            )

            # Run a few steps
            initial_time = sim_time(sim)
            sim_step!(sim; remeasure=false)
            @test sim_time(sim) > initial_time

            # Verify arrays are still on GPU
            @test sim.flow.u isa CuArray
            @test sim.flow.p isa CuArray
        end

        @testset "GPU Diagnostics" begin
            dims = (32, 32)
            L = (1.0, 1.0)
            sim = Simulation(dims, L;
                inletBC = (1.0f0, 0.0f0),
                ν = 0.01f0,
                mem = CuArray
            )
            sim_step!(sim; remeasure=false)

            # Test diagnostics return CPU arrays
            vel = cell_center_velocity(sim)
            @test vel isa Array  # Should be CPU array
            @test ndims(vel) == 3  # 2D + component dimension

            pressure = cell_center_pressure(sim)
            @test pressure isa Array
            @test ndims(pressure) == 2

            vort = cell_center_vorticity(sim)
            @test vort isa Array
            @test ndims(vort) == 2  # 2D scalar field
        end

        @testset "GPU with Immersed Body" begin
            dims = (48, 32)
            L = (1.5, 1.0)

            # Simple cylinder
            radius = 0.1f0
            center = (0.5f0, 0.5f0)
            sdf(x, t) = sqrt((x[1] - center[1])^2 + (x[2] - center[2])^2) - radius
            body = AutoBody(sdf)

            sim = Simulation(dims, L;
                inletBC = (1.0f0, 0.0f0),
                ν = 0.01f0,
                body = body,
                mem = CuArray
            )

            @test sim.flow.u isa CuArray
            @test sim.flow.μ₀ isa CuArray

            # Run with body
            sim_step!(sim; remeasure=true)
            @test sim_time(sim) > 0
        end

        @testset "GPU Force Computation" begin
            dims = (48, 32)
            L = (1.5, 1.0)

            radius = 0.1f0
            center = (0.5f0, 0.5f0)
            sdf(x, t) = sqrt((x[1] - center[1])^2 + (x[2] - center[2])^2) - radius
            body = AutoBody(sdf)

            sim = Simulation(dims, L;
                inletBC = (1.0f0, 0.0f0),
                ν = 0.01f0,
                body = body,
                L_char = 2*radius,
                mem = CuArray
            )

            sim_step!(sim; remeasure=true)

            # Force computation should work and return CPU values
            forces = force_components(sim)
            @test forces.total isa Tuple
            @test length(forces.total) == 2
            @test !isnan(forces.total[1])
        end

        @testset "GPU Scalar Indexing Check" begin
            # This test ensures no scalar indexing occurs during normal operation
            # Run with CUDA.allowscalar(false) to catch any issues
            dims = (32, 32)
            L = (1.0, 1.0)

            CUDA.allowscalar(false)
            try
                sim = Simulation(dims, L;
                    inletBC = (1.0f0, 0.0f0),
                    ν = 0.01f0,
                    mem = CuArray
                )
                sim_step!(sim; remeasure=false)
                @test true  # If we get here, no scalar indexing occurred
            catch e
                if e isa CUDA.ScalarIndexingException
                    @test false "Scalar indexing detected: $e"
                else
                    rethrow(e)
                end
            finally
                CUDA.allowscalar(true)  # Re-enable for other tests
            end
        end
    end
end
