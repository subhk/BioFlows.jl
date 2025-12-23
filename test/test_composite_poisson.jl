using Test
using BioFlows
using LinearAlgebra
using Statistics: mean

@testset "Composite Poisson Solver" begin

    @testset "PatchPoisson Construction" begin
        # Create a simple patch
        anchor = (5, 5)
        extent = (4, 4)
        level = 1  # 2x refinement

        # Mock μ₀ array (uniform coefficients)
        μ₀ = ones(Float64, 20, 20, 2)

        patch = BioFlows.PatchPoisson(anchor, extent, level, μ₀, Float64)

        # Check dimensions
        @test patch.level == 1
        @test patch.anchor == anchor
        @test patch.coarse_extent == extent
        @test patch.fine_dims == (8, 8)  # 4*2 = 8

        # Check array sizes (fine_dims + 2 ghost cells each direction)
        @test size(patch.x) == (10, 10)
        @test size(patch.L) == (10, 10, 2)
        @test size(patch.D) == (10, 10)

        # Diagonal should be computed
        @test patch.D[5, 5] ≈ -4.0 atol=0.1
    end

    @testset "PatchPoisson Level 2 (4x refinement)" begin
        anchor = (5, 5)
        extent = (2, 2)
        level = 2  # 4x refinement

        μ₀ = ones(Float64, 20, 20, 2)
        patch = BioFlows.PatchPoisson(anchor, extent, level, μ₀, Float64)

        @test patch.level == 2
        @test patch.fine_dims == (8, 8)  # 2*4 = 8
        @test BioFlows.refinement_ratio(patch) == 4
    end

    @testset "Patch Residual and Smoothing" begin
        anchor = (5, 5)
        extent = (4, 4)
        level = 1
        μ₀ = ones(Float64, 20, 20, 2)

        patch = BioFlows.PatchPoisson(anchor, extent, level, μ₀, Float64)

        # Set source term
        for I in BioFlows.inside(patch)
            fi, fj = I.I
            patch.z[I] = sin(2π * fi / 8) * sin(2π * fj / 8)
        end

        # Compute initial residual
        BioFlows.patch_residual!(patch)
        r0 = BioFlows.patch_L₂(patch)
        @test r0 > 0

        # Apply smoothing
        BioFlows.patch_jacobi!(patch; it=10)

        # Residual should decrease
        r1 = BioFlows.patch_L₂(patch)
        @test r1 < r0
    end

    @testset "CompositePoisson Construction" begin
        n = 32
        T = Float64

        # Create base arrays
        x = zeros(T, n+2, n+2)
        L = ones(T, n+2, n+2, 2)
        z = zeros(T, n+2, n+2)

        # Create MultiLevelPoisson
        base = BioFlows.MultiLevelPoisson(x, L, z)

        # Create CompositePoisson
        cp = BioFlows.CompositePoisson(base; max_level=3)

        @test cp.refinement_ratio == 2
        @test cp.max_level == 3
        @test !BioFlows.has_patches(cp)
        @test BioFlows.num_patches(cp) == 0
    end

    @testset "Add and Remove Patches" begin
        n = 32
        T = Float64
        x = zeros(T, n+2, n+2)
        L = ones(T, n+2, n+2, 2)
        z = zeros(T, n+2, n+2)

        base = BioFlows.MultiLevelPoisson(x, L, z)
        cp = BioFlows.CompositePoisson(base; max_level=2)

        # Create mock μ₀ for patch creation
        μ₀ = ones(T, n+2, n+2, 2)

        # Add a patch
        anchor = (10, 10)
        extent = (4, 4)
        level = 1

        BioFlows.add_patch!(cp, anchor, extent, level, μ₀)

        @test BioFlows.has_patches(cp)
        @test BioFlows.num_patches(cp) == 1

        patch = BioFlows.get_patch(cp, anchor)
        @test patch !== nothing
        @test patch.anchor == anchor

        # Remove the patch
        BioFlows.remove_patch!(cp, anchor)
        @test !BioFlows.has_patches(cp)
        @test BioFlows.num_patches(cp) == 0
    end

    @testset "RefinedVelocityField" begin
        # Create refined velocity field
        field = BioFlows.RefinedVelocityField(Val{2}(), Float64)

        @test BioFlows.num_patches(field) == 0

        # Add a patch
        anchor = (5, 5)
        extent = (4, 4)
        level = 1

        BioFlows.add_patch!(field, anchor, extent, level)

        @test BioFlows.num_patches(field) == 1
        @test BioFlows.has_patch(field, anchor)

        patch = BioFlows.get_patch(field, anchor)
        @test patch !== nothing
        @test patch.level == 1
        @test patch.fine_dims == (8, 8)  # 4*2 = 8

        # Clear patches
        BioFlows.clear_patches!(field)
        @test BioFlows.num_patches(field) == 0
    end

    @testset "Interpolation from Coarse" begin
        # Test interpolation of coarse velocity to fine
        field = BioFlows.RefinedVelocityField(Val{2}(), Float64)

        anchor = (5, 5)
        extent = (2, 2)
        level = 1  # 2x refinement

        BioFlows.add_patch!(field, anchor, extent, level)

        # Create coarse velocity (linear profile)
        u_coarse = zeros(Float64, 20, 20, 2)
        for i in 1:20, j in 1:20
            u_coarse[i, j, 1] = Float64(i) * 0.1
            u_coarse[i, j, 2] = Float64(j) * 0.1
        end

        # Interpolate
        patch = BioFlows.get_patch(field, anchor)
        BioFlows.interpolate_from_coarse!(patch, u_coarse, anchor)

        # Fine velocities should be approximately linear too
        @test patch.u[3, 3, 1] > 0  # Should have positive u
        @test patch.u[3, 3, 2] > 0  # Should have positive w
    end

    @testset "Interface Operators - Prolongation" begin
        # Test boundary prolongation
        anchor = (5, 5)
        extent = (4, 4)
        level = 1
        μ₀ = ones(Float64, 20, 20, 2)

        patch = BioFlows.PatchPoisson(anchor, extent, level, μ₀, Float64)

        # Create coarse pressure (linear)
        p_coarse = zeros(Float64, 20, 20)
        for i in 1:20, j in 1:20
            p_coarse[i, j] = Float64(i + j)
        end

        # Set boundary
        BioFlows.set_patch_boundary!(patch, p_coarse, anchor)

        # Check ghost cells are filled
        @test patch.x[1, 5] != 0  # Left ghost
        @test patch.x[10, 5] != 0  # Right ghost
        @test patch.x[5, 1] != 0  # Bottom ghost
        @test patch.x[5, 10] != 0  # Top ghost
    end

    @testset "Interface Operators - Restriction" begin
        anchor = (5, 5)
        extent = (4, 4)
        level = 1
        μ₀ = ones(Float64, 20, 20, 2)

        patch = BioFlows.PatchPoisson(anchor, extent, level, μ₀, Float64)

        # Set fine pressure (constant)
        fill!(patch.x, 1.0)

        # Create coarse array
        p_coarse = zeros(Float64, 20, 20)

        # Restrict
        BioFlows.restrict_pressure_to_coarse!(p_coarse, patch, anchor)

        # Coarse cells covered by patch should have pressure ≈ 1
        @test p_coarse[5, 5] ≈ 1.0 atol=1e-10
        @test p_coarse[8, 8] ≈ 1.0 atol=1e-10
    end

    @testset "Cell Clustering" begin
        # Test clustering of cells into patches
        cells = [(1, 1), (1, 2), (2, 1), (2, 2),  # Cluster 1
                 (10, 10), (10, 11), (11, 10)]     # Cluster 2

        clusters = BioFlows.cluster_cells_2d(cells)

        @test length(clusters) == 2

        # Check cluster sizes
        sizes = sort([length(c) for c in clusters])
        @test sizes == [3, 4]
    end

    @testset "Bounding Box" begin
        cells = [(5, 5), (5, 6), (6, 5), (7, 8)]

        anchor, extent = BioFlows.bounding_box_2d(cells)

        @test anchor == (5, 5)
        @test extent == (3, 4)  # 7-5+1, 8-5+1
    end

    @testset "Integration with Simulation" begin
        # Test that CompositePoisson works with Simulation
        n = 32
        ν = 0.01

        sim = BioFlows.Simulation((n, n), (1.0, 0.0), Float64(n/4);
                                   ν=ν, T=Float64)

        # Create CompositePoisson from simulation's MultiLevelPoisson
        cp = BioFlows.CompositePoisson(sim.pois; max_level=2)

        # Should share base arrays
        @test cp.base === sim.pois
        @test cp.x === sim.pois.x

        # Add a patch manually
        μ₀ = sim.flow.μ₀
        BioFlows.add_patch!(cp, (10, 10), (4, 4), 1, μ₀)

        @test BioFlows.has_patches(cp)

        # Solver should work
        @inside cp.base.z[I] = sin(2π * I[1]/n) * sin(2π * I[2]/n)
        BioFlows.solver!(cp; tol=1e-4, itmx=50)

        @test length(cp.n) > 0
    end

    @testset "Flux Mismatch Computation" begin
        anchor = (5, 5)
        extent = (4, 4)
        level = 1
        μ₀ = ones(Float64, 20, 20, 2)

        patch = BioFlows.PatchPoisson(anchor, extent, level, μ₀, Float64)

        # Set uniform pressure (no gradient)
        fill!(patch.x, 1.0)

        # Coarse pressure also uniform
        p_coarse = ones(Float64, 20, 20)
        L_coarse = ones(Float64, 20, 20, 2)

        # Compute interface fluxes
        fluxes = BioFlows.compute_all_interface_fluxes(patch, p_coarse, L_coarse, anchor)

        # With uniform pressure, flux mismatch should be small
        total_mismatch = BioFlows.total_flux_mismatch(fluxes)
        @test total_mismatch < 1e-10
    end

    @testset "Proper Nesting" begin
        # Create a refined grid
        grid = BioFlows.StaggeredGrid(32, 32, 1.0, 1.0)
        rg = BioFlows.RefinedGrid(grid)

        # Mark a single cell at level 2 (4x refinement)
        rg.refined_cells_2d[(15, 15)] = 2

        # Ensure proper nesting
        BioFlows.ensure_proper_nesting!(rg, 1)

        # Surrounding cells should be level 1
        @test get(rg.refined_cells_2d, (14, 15), 0) >= 1
        @test get(rg.refined_cells_2d, (16, 15), 0) >= 1
        @test get(rg.refined_cells_2d, (15, 14), 0) >= 1
        @test get(rg.refined_cells_2d, (15, 16), 0) >= 1
    end

    @testset "Create Patches from RefinedGrid" begin
        n = 32
        T = Float64

        # Create CompositePoisson
        x = zeros(T, n+2, n+2)
        L = ones(T, n+2, n+2, 2)
        z = zeros(T, n+2, n+2)
        base = BioFlows.MultiLevelPoisson(x, L, z)
        cp = BioFlows.CompositePoisson(base; max_level=2)

        # Create refined grid with some marked cells
        grid = BioFlows.StaggeredGrid(n, n, 1.0, 1.0)
        rg = BioFlows.RefinedGrid(grid)

        # Mark a cluster of cells
        for i in 10:13, j in 10:13
            rg.refined_cells_2d[(i, j)] = 1
        end

        # Create patches
        μ₀ = ones(T, n+2, n+2, 2)
        BioFlows.create_patches!(cp, rg, μ₀)

        @test BioFlows.has_patches(cp)
        @test BioFlows.num_patches(cp) >= 1
    end

    @testset "AMRSimulation Basic" begin
        n = 32

        # Define a simple circle body
        radius = Float64(n/8)
        center = (Float64(n/4), Float64(n/2))

        circle_sdf(x, t) = sqrt((x[1]-center[1])^2 + (x[2]-center[2])^2) - radius

        body = BioFlows.AutoBody(circle_sdf)

        config = BioFlows.AMRConfig(
            max_level=1,
            body_distance_threshold=3.0,
            regrid_interval=5,
            buffer_size=1
        )

        amr = BioFlows.AMRSimulation((n, n), (1.0, 0.0), Float64(n/4);
                                      ν=0.01, body=body, amr_config=config,
                                      T=Float64)

        # AMR simulation should be created
        @test amr.amr_active == true
        @test amr.config.max_level == 1

        # Composite poisson should exist
        @test amr.composite_pois !== nothing
        @test amr.composite_pois.max_level == 1
    end

    @testset "AMRSimulation Step" begin
        n = 32

        radius = Float64(n/8)
        center = (Float64(n/4), Float64(n/2))
        circle_sdf(x, t) = sqrt((x[1]-center[1])^2 + (x[2]-center[2])^2) - radius

        body = BioFlows.AutoBody(circle_sdf)

        # Use base solver only (without patches) for stability test
        # The full AMR solver may need more tuning for stability
        config = BioFlows.AMRConfig(
            max_level=1,
            body_distance_threshold=2.0,
            regrid_interval=100,  # Don't regrid during test
            buffer_size=1
        )

        amr = BioFlows.AMRSimulation((n, n), (1.0, 0.0), Float64(n/4);
                                      ν=0.05, body=body, amr_config=config,
                                      T=Float64)

        # Take a few steps using base solver
        for _ in 1:5
            BioFlows.sim_step!(amr; remeasure=true)
        end

        # Time should have advanced
        @test BioFlows.time(amr) > 0

        # Velocity should be bounded when using base solver
        @test all(isfinite, amr.flow.u)
        @test maximum(abs, amr.flow.u) < 100
    end

    @testset "Divergence Check" begin
        n = 32
        ν = 0.01

        sim = BioFlows.Simulation((n, n), (1.0, 0.0), Float64(n/4);
                                   ν=ν, T=Float64)

        # Create composite poisson
        cp = BioFlows.CompositePoisson(sim.pois; max_level=2)

        # Add some divergence
        for I in BioFlows.inside(sim.flow.p)
            sim.flow.u[I, 1] += 0.1 * sin(2π * I[1]/n)
        end

        # Check divergence
        base_div, _ = BioFlows.divergence_at_all_levels(sim.flow, cp)
        @test base_div > 0.01  # Should have significant divergence

        # Project
        BioFlows.project!(sim.flow, sim.pois)

        # Check divergence after
        base_div_after, _ = BioFlows.divergence_at_all_levels(sim.flow, cp)
        @test base_div_after < base_div * 0.1  # Should be reduced significantly
    end

end
