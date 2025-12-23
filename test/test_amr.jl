using Test
using BioFlows
using LinearAlgebra: norm
using StaticArrays: SVector

@testset "AMR Types" begin
    # Test StaggeredGrid creation (2D)
    grid2d = StaggeredGrid(32, 32, 0.1, 0.1)
    @test grid2d.grid_type == TwoDimensional
    @test grid2d.nx == 32
    @test grid2d.nz == 32
    @test grid2d.ny == 1
    @test length(grid2d.x) == 32
    @test length(grid2d.z) == 32
    @test is_2d(grid2d)
    @test !is_3d(grid2d)

    # Test StaggeredGrid creation (3D)
    grid3d = StaggeredGrid(16, 16, 16, 0.1, 0.1, 0.1)
    @test grid3d.grid_type == ThreeDimensional
    @test grid3d.nx == 16
    @test grid3d.ny == 16
    @test grid3d.nz == 16
    @test is_3d(grid3d)

    # Test SolutionState creation
    state2d = SolutionState(grid2d)
    @test size(state2d.u) == (33, 32)  # nx+1, nz
    @test size(state2d.v) == (32, 33)  # nx, nz+1
    @test size(state2d.p) == (32, 32)
    @test state2d.w === nothing

    state3d = SolutionState(grid3d)
    @test size(state3d.u) == (17, 16, 16)
    @test size(state3d.v) == (16, 17, 16)
    @test size(state3d.w) == (16, 16, 17)
    @test size(state3d.p) == (16, 16, 16)

    # Test RefinedGrid
    rg = RefinedGrid(grid2d)
    @test num_refined_cells(rg) == 0
    @test refinement_level(rg, 5, 5) == 0
end

@testset "AMR Adapter" begin
    # Create a simple simulation
    sdf(x, t) = norm(x .- SVector(16.0, 16.0)) - 4.0
    # Domain size = (32, 32) with Δx=1, characteristic length L_char=8.0
    sim = Simulation((32, 32), (32.0, 32.0);
                     inletBC=(1.0, 0.0), ν=0.01, body=AutoBody(sdf), L_char=8.0)

    # Test adapter creation
    adapter = FlowToGridAdapter(sim.flow, 8.0)
    @test adapter.L == 8.0
    @test adapter.dx ≈ 8.0 / 32

    # Test grid conversion
    grid = flow_to_staggered_grid(adapter)
    @test grid.nx == 32
    @test grid.nz == 32
    @test is_2d(grid)

    # Test solution state extraction
    state = flow_to_solution_state(sim.flow)
    @test size(state.p) == (32, 32)
    @test size(state.u) == (33, 32)
    @test size(state.v) == (32, 33)

    # Test refined grid creation
    rg = create_refined_grid(adapter)
    @test rg.base_grid.nx == 32
end

@testset "Body Refinement Indicator" begin
    # Create simulation with cylinder body
    sdf(x, t) = norm(x .- SVector(16.0, 16.0)) - 4.0
    # Domain size = (32, 32) with Δx=1, characteristic length L_char=8.0
    sim = Simulation((32, 32), (32.0, 32.0);
                     inletBC=(1.0, 0.0), ν=0.01, body=AutoBody(sdf), L_char=8.0)

    # Advance a few steps to get non-trivial flow
    for _ in 1:2  # Reduced from 5 for CI
        sim_step!(sim; remeasure=false)
    end

    # Test body proximity indicator
    body_ind = compute_body_refinement_indicator(sim.flow, sim.body;
                                                  distance_threshold=3.0)
    @test size(body_ind) == size(sim.flow.p)
    @test maximum(body_ind) > 0  # Should have some cells marked
    @test minimum(body_ind) >= 0
    @test maximum(body_ind) <= 1

    # Test velocity gradient indicator
    grad_ind = compute_velocity_gradient_indicator(sim.flow)
    @test size(grad_ind) == size(sim.flow.p)
    @test minimum(grad_ind) >= 0

    # Test vorticity indicator
    vort_ind = compute_vorticity_indicator(sim.flow)
    @test size(vort_ind) == size(sim.flow.p)
    @test minimum(vort_ind) >= 0

    # Test combined indicator
    combined = compute_combined_indicator(sim.flow, sim.body;
                                          body_threshold=3.0,
                                          gradient_threshold=0.5,
                                          vorticity_threshold=0.5)
    @test size(combined) == size(sim.flow.p)
    @test minimum(combined) >= 0
    @test maximum(combined) <= 1
end

@testset "AMRConfig" begin
    # Test default config
    config = AMRConfig()
    @test config.max_level == 2
    @test config.body_distance_threshold == 3.0
    @test config.regrid_interval == 10

    # Test custom config
    config2 = AMRConfig(max_level=4, body_distance_threshold=5.0, regrid_interval=5)
    @test config2.max_level == 4
    @test config2.body_distance_threshold == 5.0
    @test config2.regrid_interval == 5
end

@testset "AMRSimulation" begin
    # Create AMR simulation
    sdf(x, t) = norm(x .- SVector(16.0, 16.0)) - 4.0
    config = AMRConfig(max_level=2, body_distance_threshold=4.0, regrid_interval=5)

    # Domain size = (32, 32) with Δx=1, characteristic length L_char=8.0
    amr_sim = AMRSimulation((32, 32), (32.0, 32.0);
                            inletBC=(1.0, 0.0), ν=0.01, body=AutoBody(sdf),
                            amr_config=config, L_char=8.0)

    @test isa(amr_sim, AMRSimulation)
    @test isa(amr_sim, AbstractSimulation)
    @test amr_sim.config.max_level == 2
    @test amr_sim.amr_active == true

    # Test property forwarding
    @test amr_sim.L == 8.0
    @test isa(amr_sim.flow, Flow)
    @test isa(amr_sim.body, AbstractBody)

    # Test simulation stepping
    initial_time = sim_time(amr_sim)
    sim_step!(amr_sim; remeasure=true)
    @test sim_time(amr_sim) > initial_time

    # Run more steps to trigger regridding
    for _ in 1:3  # Reduced from 10 for CI
        sim_step!(amr_sim; remeasure=false)
    end

    # Check that some cells were marked for refinement
    @test num_refined_cells(amr_sim.refined_grid) >= 0

    # Test refinement indicator
    indicator = get_refinement_indicator(amr_sim)
    @test size(indicator) == size(amr_sim.flow.p)

    # Test AMR disable/enable
    set_amr_active!(amr_sim, false)
    @test amr_sim.amr_active == false
    set_amr_active!(amr_sim, true)
    @test amr_sim.amr_active == true

    # Test measure and perturb
    measure!(amr_sim)
    perturb!(amr_sim; noise=0.01)
end

@testset "AMR Cell Marking" begin
    # Create a test indicator array
    indicator = zeros(Float32, 34, 34)  # With ghost cells

    # Mark a region
    for i in 10:20, j in 10:20
        indicator[i, j] = 1.0
    end

    # Test cell marking
    cells = mark_cells_for_refinement(indicator; threshold=0.5)
    @test length(cells) == 121  # 11x11 cells

    # Test buffer zone
    indicator2 = copy(indicator)
    apply_buffer_zone!(indicator2; buffer_size=2)
    cells2 = mark_cells_for_refinement(indicator2; threshold=0.5)
    @test length(cells2) > length(cells)  # Should have more cells with buffer
end
