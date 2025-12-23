using Test
using BioFlows
using Statistics: mean

# Helper function for correlation - must be defined before testset
function cor(x, y)
    mx, my = mean(x), mean(y)
    num = sum((x .- mx) .* (y .- my))
    den = sqrt(sum((x .- mx).^2) * sum((y .- my).^2))
    return den > 0 ? num / den : 0.0
end

@testset "Pressure Solver Verification" begin

    @testset "Poisson Solver Convergence" begin
        # Create a simple test case: solve ∇²p = f with known solution
        n = 32
        T = Float64

        # Create arrays for Poisson solver
        x = zeros(T, n+2, n+2)  # solution (with ghost cells)
        z = zeros(T, n+2, n+2)  # source term
        L = ones(T, n+2, n+2, 2)  # coefficient array (uniform = 1)

        # Set source term: must have zero mean for Neumann BCs
        for I in CartesianIndices((2:n+1, 2:n+1))
            z[I] = sin(2π * (I[1]-1.5) / n) * sin(2π * (I[2]-1.5) / n)
        end

        # Create Poisson solver
        pois = BioFlows.Poisson(x, L, z)

        # Verify diagonal is computed correctly
        # For interior cells with L=1: D = -(L[I,1]+L[I+1,1]+L[I,2]+L[I+1,2]) = -4
        @test pois.D[5,5] ≈ -4.0
        @test pois.iD[5,5] ≈ -0.25

        # Solve with tight tolerance
        BioFlows.solver!(pois; tol=1e-10, itmx=2000)

        # Check that solver ran (iterations recorded)
        @test length(pois.n) > 0
        @test pois.n[end] > 0

        # L₂ norm of residual should be very small (solver! uses L₂ internally)
        @test BioFlows.L₂(pois) < 1e-8
    end

    @testset "MultiLevel Poisson via Simulation" begin
        # Test multigrid solver through Simulation (proper BC setup)
        n = 64
        ν = 0.01

        # Domain size = (n, n) with Δx=1, L_char = n/4
        sim = BioFlows.Simulation((n, n), (1.0, 0.0), (Float64(n), Float64(n));
                                   ν=ν, T=Float64, L_char=Float64(n/4))

        # Simulation uses MultiLevelPoisson
        @test isa(sim.pois, BioFlows.MultiLevelPoisson)

        # Should have multiple levels
        @test length(sim.pois.levels) >= 3

        # Add divergence and project
        for I in BioFlows.inside(sim.flow.p)
            sim.flow.u[I, 1] += 0.1 * sin(2π * I[1]/n)
        end

        BioFlows.project!(sim.flow, sim.pois)

        # Solver should have run
        @test length(sim.pois.n) > 0
        @test sim.pois.n[end] > 0
    end

    @testset "Projection Makes Velocity Divergence-Free" begin
        # Create a flow and verify projection produces divergence-free field
        n = 64
        ν = 0.01  # kinematic viscosity

        # Simple simulation setup - Domain size = (n, n) with Δx=1, L_char = n/4
        sim = BioFlows.Simulation((n, n), (1.0, 0.0), (Float64(n), Float64(n));
                                   ν=ν, T=Float64, L_char=Float64(n/4))
        flow = sim.flow
        pois = sim.pois

        # Add some divergence to the velocity field
        for I in BioFlows.inside(flow.p)
            flow.u[I, 1] += 0.1 * sin(2π * I[1]/n)
            flow.u[I, 2] += 0.1 * cos(2π * I[2]/n)
        end

        # Compute initial divergence
        div_before = 0.0
        for I in BioFlows.inside(flow.p)
            div_before = max(div_before, abs(BioFlows.div(I, flow.u)))
        end

        # Project to make divergence-free
        BioFlows.project!(flow, pois)

        # Compute divergence after projection
        div_after = 0.0
        for I in BioFlows.inside(flow.p)
            div_after = max(div_after, abs(BioFlows.div(I, flow.u)))
        end

        # Divergence should be significantly reduced
        @test div_after < div_before * 0.1
        # Should be reasonably small (iterative solver with default tolerance)
        @test div_after < 1e-2
    end

    @testset "Matrix Symmetry" begin
        # Verify the Poisson matrix is symmetric
        n = 16
        T = Float64

        x = zeros(T, n+2, n+2)
        z = zeros(T, n+2, n+2)
        L = ones(T, n+2, n+2, 2)

        pois = BioFlows.Poisson(x, L, z)

        # Test symmetry: (Ax)·y = x·(Ay) for random vectors
        x1 = randn(T, n+2, n+2)
        x2 = randn(T, n+2, n+2)

        # Zero out ghost cells
        x1[1,:] .= 0; x1[end,:] .= 0; x1[:,1] .= 0; x1[:,end] .= 0
        x2[1,:] .= 0; x2[end,:] .= 0; x2[:,1] .= 0; x2[:,end] .= 0

        # Compute Ax1 and Ax2
        pois.x .= x1
        Ax1 = copy(BioFlows.mult!(pois, x1))

        pois.x .= x2
        Ax2 = copy(BioFlows.mult!(pois, x2))

        # Check symmetry: (Ax1)·x2 ≈ x1·(Ax2)
        dot1 = sum(Ax1 .* x2)
        dot2 = sum(x1 .* Ax2)
        @test isapprox(dot1, dot2, rtol=1e-10)
    end

    @testset "Jacobi Smoother Reduces Residual" begin
        n = 32
        T = Float64

        x = zeros(T, n+2, n+2)
        z = zeros(T, n+2, n+2)
        L = ones(T, n+2, n+2, 2)

        # Non-zero source with zero mean
        for I in CartesianIndices((2:n+1, 2:n+1))
            z[I] = sin(2π * (I[1]-1.5) / n)
        end

        pois = BioFlows.Poisson(x, L, z)

        # Compute initial residual
        BioFlows.residual!(pois)
        r0 = BioFlows.L₂(pois)

        # Apply Jacobi iterations - this updates r internally
        BioFlows.Jacobi!(pois; it=100)

        # L₂ is computed from r which is updated by increment!
        r1 = BioFlows.L₂(pois)

        # Residual should decrease after many Jacobi iterations
        @test r1 < r0
    end

    @testset "PCG Smoother Converges" begin
        # Test that full solver with PCG converges
        n = 32
        T = Float64

        x = zeros(T, n+2, n+2)
        z = zeros(T, n+2, n+2)
        L = ones(T, n+2, n+2, 2)

        # Non-zero source with zero mean
        for I in CartesianIndices((2:n+1, 2:n+1))
            z[I] = sin(2π * (I[1]-1.5) / n) * cos(2π * (I[2]-1.5) / n)
        end

        pois = BioFlows.Poisson(x, L, z)

        # Use full solver which handles PCG correctly
        BioFlows.solver!(pois; tol=1e-8, itmx=500)

        # Solver should converge
        @test BioFlows.L₂(pois) < 1e-6
        @test length(pois.n) > 0
    end

    @testset "Restriction and Prolongation Operators" begin
        n = 32
        T = Float64

        # Create fine grid data
        fine = zeros(T, n+2, n+2)
        for I in CartesianIndices((2:n+1, 2:n+1))
            fine[I] = sin(2π * I[1]/n) * sin(2π * I[2]/n)
        end

        # Restrict to coarse grid
        nc = n ÷ 2
        coarse = zeros(T, nc+2, nc+2)
        for I in CartesianIndices((2:nc+1, 2:nc+1))
            coarse[I] = BioFlows.restrict(I, fine)
        end

        # Coarse grid should capture smooth features
        @test maximum(abs, coarse) > 0

        # Prolongate back to fine grid
        fine2 = zeros(T, n+2, n+2)
        for I in CartesianIndices((2:n+1, 2:n+1))
            fine2[I] = coarse[BioFlows.down(I)]
        end

        # Prolongated field should be similar to original (for smooth data)
        # Note: piecewise constant prolongation, so not exact
        correlation = cor(vec(fine[2:n+1,2:n+1]), vec(fine2[2:n+1,2:n+1]))
        @test correlation > 0.9
    end

    @testset "Momentum Step Maintains Incompressibility" begin
        # Full simulation test: after mom_step!, velocity should be divergence-free
        n = 48
        ν = 0.01
        # Domain size = (n, n) with Δx=1, L_char = n/4
        sim = BioFlows.Simulation((n, n), (1.0, 0.0), (Float64(n), Float64(n));
                                   ν=ν, T=Float64, L_char=Float64(n/4))

        # Take a momentum step
        BioFlows.mom_step!(sim.flow, sim.pois)

        # Check divergence
        max_div = 0.0
        for I in BioFlows.inside(sim.flow.p)
            d = abs(BioFlows.div(I, sim.flow.u))
            max_div = max(max_div, d)
        end

        # Should be essentially divergence-free
        @test max_div < 1e-3
    end

    @testset "CFL Condition" begin
        n = 32
        ν = 0.01
        # Domain size = (n, n) with Δx=1, L_char = n/4
        sim = BioFlows.Simulation((n, n), (1.0, 0.0), (Float64(n), Float64(n));
                                   ν=ν, T=Float64, L_char=Float64(n/4))

        # CFL should give reasonable time step
        dt = BioFlows.CFL(sim.flow)
        @test dt > 0
        @test dt < 10  # Should be bounded
        @test isfinite(dt)
    end

    @testset "Predictor-Corrector Scheme" begin
        # Verify that the two-stage scheme produces stable results
        n = 32
        ν = 0.01
        # Domain size = (n, n) with Δx=1, L_char = n/4
        sim = BioFlows.Simulation((n, n), (1.0, 0.0), (Float64(n), Float64(n));
                                   ν=ν, T=Float64, L_char=Float64(n/4))

        # Take several momentum steps
        for _ in 1:5
            BioFlows.mom_step!(sim.flow, sim.pois)
        end

        # Time should have advanced
        @test BioFlows.time(sim.flow) > 0

        # Velocity should remain bounded
        @test maximum(abs, sim.flow.u) < 10
        @test all(isfinite, sim.flow.u)

        # Pressure should be computed
        @test all(isfinite, sim.flow.p)
    end

    @testset "Divergence Operator" begin
        # Test that div computes correctly
        n = 16
        T = Float64

        # Create uniform flow field u = (1, 0)
        u = ones(T, n+2, n+2, 2)
        u[:,:,1] .= 1.0  # u_x = 1
        u[:,:,2] .= 0.0  # u_z = 0

        # Divergence of uniform flow should be zero
        for I in BioFlows.inside(zeros(T, n+2, n+2))
            d = BioFlows.div(I, u)
            @test abs(d) < 1e-10
        end

        # Create linear flow u = (x, -z) which has div = 0
        for I in CartesianIndices((1:n+2, 1:n+2))
            u[I, 1] = Float64(I[1])
            u[I, 2] = -Float64(I[2])
        end

        # Divergence should be zero for incompressible flow
        for I in BioFlows.inside(zeros(T, n+2, n+2))
            d = BioFlows.div(I, u)
            @test abs(d) < 1e-10
        end
    end

    @testset "Diagonal Computation" begin
        # Verify diagonal entries are correct for different coefficients
        n = 8
        T = Float64

        # Non-uniform L
        L = zeros(T, n+2, n+2, 2)
        L[3:n, 3:n, 1] .= 2.0
        L[3:n, 3:n, 2] .= 3.0

        x = zeros(T, n+2, n+2)
        z = zeros(T, n+2, n+2)

        pois = BioFlows.Poisson(x, L, z)

        # Interior point with L[:,:,1]=2, L[:,:,2]=3
        # D[I] = -(L[I,1] + L[I+1,1] + L[I,2] + L[I+1,2]) = -(2+2+3+3) = -10
        @test pois.D[5,5] ≈ -10.0
    end
end
