# AMR 2D XZ-plane smoke test
# Run with: julia --project examples/amr_2d_smoke.jl

using BioFlows

function main()
    # Base grid and dummy state
    nx, nz = 32, 16
    Lx, Lz = 2.0, 1.0
    grid = BioFlows.StaggeredGrid2D(nx, nz, Lx, Lz)

    state = BioFlows.SolutionState2D(nx, nz)
    # Create a simple pressure field with a localized gradient
    for j in 1:nz, i in 1:nx
        x = grid.x[i]; z = grid.z[j]
        state.p[i, j] = exp(-50*((x-1.0)^2 + (z-0.5)^2))
    end

    # One rigid circle body near the center
    bodies = BioFlows.RigidBodyCollection()
    circle = BioFlows.Circle(0.12, [1.0, 0.5])  # center [x,z]
    BioFlows.add_body!(bodies, circle)

    # Criteria
    crit = BioFlows.AdaptiveRefinementCriteria(
        velocity_gradient_threshold=0.0,
        pressure_gradient_threshold=0.5,
        vorticity_threshold=1e9,
        body_distance_threshold=0.2,
        max_refinement_level=2,
        min_grid_size=min(grid.dx, grid.dz)/4,
    )

    # Refined grid container
    rg = BioFlows.RefinedGrid(grid)

    # Compute indicators and select cells to refine
    ind = BioFlows.compute_refinement_indicators(grid, state, bodies, crit)
    cells = BioFlows.mark_cells_for_refinement!(rg, ind, crit)
    println("Marked ", length(cells), " cells for refinement")

    # Refine and build interpolation weights
    BioFlows.refine_cells!(rg, cells)

    # Quick validations
    @assert all(((i,j),) -> haskey(rg.refined_grids_2d, (i,j)), cells) "Refined grids missing"
    @assert all(((i,j),) -> haskey(rg.interpolation_weights_2d, (i,j)), cells) "Weights missing"
    println("AMR 2D smoke test OK. Levels assigned: ", length(rg.refined_cells_2d))
end

main()

