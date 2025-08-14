struct AdaptiveRefinementCriteria
    velocity_gradient_threshold::Float64
    pressure_gradient_threshold::Float64
    vorticity_threshold::Float64
    body_distance_threshold::Float64
    max_refinement_level::Int
    min_grid_size::Float64
end

function AdaptiveRefinementCriteria(;
    velocity_gradient_threshold::Float64=1.0,
    pressure_gradient_threshold::Float64=10.0,
    vorticity_threshold::Float64=5.0,
    body_distance_threshold::Float64=0.1,
    max_refinement_level::Int=3,
    min_grid_size::Float64=0.001)
    
    AdaptiveRefinementCriteria(velocity_gradient_threshold, pressure_gradient_threshold,
                              vorticity_threshold, body_distance_threshold,
                              max_refinement_level, min_grid_size)
end

mutable struct RefinedGrid
    base_grid::StaggeredGrid
    refined_cells::Dict{Any, Int}  # Flexible indexing: (i,j) for 2D, (i,j,k) for 3D -> refinement_level
    refined_grids::Dict{Any, StaggeredGrid}  # Flexible indexing -> local refined grid
    interpolation_weights::Dict{Any, Vector{Tuple{Any, Float64}}}  # Flexible indexing
end

function RefinedGrid(base_grid::StaggeredGrid)
    RefinedGrid(base_grid, Dict{Tuple{Int,Int}, Int}(), 
               Dict{Tuple{Int,Int}, StaggeredGrid}(),
               Dict{Tuple{Int,Int}, Vector{Tuple{Tuple{Int,Int}, Float64}}}())
end

function compute_refinement_indicators(grid::StaggeredGrid, state::SolutionState, 
                                     bodies::Union{RigidBodyCollection, FlexibleBodyCollection, Nothing},
                                     criteria::AdaptiveRefinementCriteria)
    nx, ny = grid.nx, grid.ny
    indicators = zeros(nx, ny)
    
    # Velocity gradient indicator
    velocity_indicator = compute_velocity_gradient_indicator(grid, state)
    
    # Pressure gradient indicator  
    pressure_indicator = compute_pressure_gradient_indicator(grid, state)
    
    # Vorticity indicator
    vorticity_indicator = compute_vorticity_indicator(grid, state)
    
    # Body proximity indicator
    body_indicator = zeros(nx, ny)
    if bodies !== nothing
        body_indicator = compute_body_proximity_indicator(grid, bodies, criteria.body_distance_threshold)
    end
    
    # Combine indicators
    for j = 1:ny, i = 1:nx
        vel_flag = velocity_indicator[i, j] > criteria.velocity_gradient_threshold
        press_flag = pressure_indicator[i, j] > criteria.pressure_gradient_threshold
        vort_flag = vorticity_indicator[i, j] > criteria.vorticity_threshold
        body_flag = body_indicator[i, j] > 0.5
        
        if vel_flag || press_flag || vort_flag || body_flag
            indicators[i, j] = 1.0
        end
    end
    
    return indicators
end

function compute_velocity_gradient_indicator(grid::StaggeredGrid, state::SolutionState)
    nx, ny = grid.nx, grid.ny
    indicator = zeros(nx, ny)
    
    # Use proper staggered grid interpolation from differential operators
    u_cc = interpolate_u_to_cell_center(state.u, grid)
    v_cc = interpolate_v_to_cell_center(state.v, grid)
    
    # Use proper 2nd order differential operators for velocity gradients
    dudx = ddx(u_cc, grid)
    dudy = ddy(u_cc, grid)
    dvdx = ddx(v_cc, grid)
    dvdy = ddy(v_cc, grid)
    
    # Magnitude of velocity gradient tensor
    for j = 1:ny, i = 1:nx
        indicator[i, j] = sqrt(dudx[i, j]^2 + dudy[i, j]^2 + dvdx[i, j]^2 + dvdy[i, j]^2)
    end
    
    return indicator
end

function compute_pressure_gradient_indicator(grid::StaggeredGrid, state::SolutionState)
    nx, ny = grid.nx, grid.ny
    indicator = zeros(nx, ny)
    
    # Use proper differential operators for pressure gradients
    dpdx = ddx(state.p, grid)
    dpdy = ddy(state.p, grid)
    
    for j = 1:ny, i = 1:nx
        indicator[i, j] = sqrt(dpdx[i, j]^2 + dpdy[i, j]^2)
    end
    
    return indicator
end

function compute_vorticity_indicator(grid::StaggeredGrid, state::SolutionState)
    nx, ny = grid.nx, grid.ny
    indicator = zeros(nx, ny)
    
    # Compute vorticity ω = ∂v/∂x - ∂u/∂y using proper staggered grid operators
    # First interpolate to cell centers
    u_cc = interpolate_u_to_cell_center(state.u, grid)
    v_cc = interpolate_v_to_cell_center(state.v, grid)
    
    # Then compute derivatives
    dvdx = ddx(v_cc, grid)
    dudy = ddy(u_cc, grid)
    
    for j = 1:ny, i = 1:nx
        indicator[i, j] = abs(dvdx[i, j] - dudy[i, j])
    end
    
    return indicator
end

function compute_body_proximity_indicator(grid::StaggeredGrid, 
                                        bodies::Union{RigidBodyCollection, FlexibleBodyCollection},
                                        distance_threshold::Float64)
    nx, ny = grid.nx, grid.ny
    indicator = zeros(nx, ny)
    
    for j = 1:ny, i = 1:nx
        x = grid.x[i]
        y = grid.y[j]
        
        min_distance = Inf
        
        if bodies isa RigidBodyCollection
            for body in bodies.bodies
                dist = distance_to_surface(body, x, y)
                min_distance = min(min_distance, abs(dist))
            end
        elseif bodies isa FlexibleBodyCollection
            for body in bodies.bodies
                # For flexible bodies, compute distance to closest Lagrangian point
                for k = 1:body.n_points
                    dist = sqrt((x - body.X[k, 1])^2 + (y - body.X[k, 2])^2)
                    min_distance = min(min_distance, dist)
                end
            end
        end
        
        if min_distance < distance_threshold
            indicator[i, j] = 1.0
        end
    end
    
    return indicator
end

function mark_cells_for_refinement!(refined_grid::RefinedGrid, indicators::Matrix{Float64},
                                   criteria::AdaptiveRefinementCriteria)
    nx, ny = size(indicators)
    cells_to_refine = Tuple{Int,Int}[]
    
    for j = 1:ny, i = 1:nx
        if indicators[i, j] > 0.5
            current_level = get(refined_grid.refined_cells, (i, j), 0)
            
            if current_level < criteria.max_refinement_level
                # Check if resulting grid size would be above minimum
                base_dx = refined_grid.base_grid.dx
                base_dy = refined_grid.base_grid.dy
                
                new_dx = base_dx / (2^(current_level + 1))
                new_dy = base_dy / (2^(current_level + 1))
                
                if new_dx >= criteria.min_grid_size && new_dy >= criteria.min_grid_size
                    push!(cells_to_refine, (i, j))
                end
            end
        end
    end
    
    return cells_to_refine
end

function refine_cells!(refined_grid::RefinedGrid, cells_to_refine::Vector{Tuple{Int,Int}})
    for (i, j) in cells_to_refine
        current_level = get(refined_grid.refined_cells, (i, j), 0)
        new_level = current_level + 1
        
        # Update refinement level
        refined_grid.refined_cells[(i, j)] = new_level
        
        # Create local refined grid for this cell
        base_grid = refined_grid.base_grid
        
        # Determine local grid bounds
        x_min = base_grid.x[i] - base_grid.dx/2
        x_max = base_grid.x[i] + base_grid.dx/2
        y_min = base_grid.y[j] - base_grid.dy/2
        y_max = base_grid.y[j] + base_grid.dy/2
        
        # Create refined grid with 2^level times finer resolution
        refine_factor = 2^new_level
        local_nx = 2 * refine_factor
        local_ny = 2 * refine_factor
        
        Lx_local = x_max - x_min
        Ly_local = y_max - y_min
        
        local_grid = StaggeredGrid2D(local_nx, local_ny, Lx_local, Ly_local;  # local_ny maps to nz, Ly_local to Lz
                                   origin_x=x_min, origin_z=y_min)  # y_min maps to z_min in XZ plane
        
        refined_grid.refined_grids[(i, j)] = local_grid
        
        # Compute interpolation weights for coupling with base grid
        compute_interpolation_weights!(refined_grid, (i, j), local_grid)
        
        println("Refined cell ($i, $j) to level $new_level")
    end
end

function compute_interpolation_weights!(refined_grid::RefinedGrid, cell_idx::Tuple{Int,Int},
                                      local_grid::StaggeredGrid)
    # Compute interpolation weights for transferring data between base grid and refined grid
    base_grid = refined_grid.base_grid
    i, j = cell_idx
    
    weights = Vector{Tuple{Tuple{Int,Int}, Float64}}()
    
    # For each point in the local refined grid, find interpolation weights from base grid
    for jj = 1:local_grid.ny, ii = 1:local_grid.nx
        x_local = local_grid.x[ii]
        y_local = local_grid.y[jj]
        
        # Find base grid cell containing this point
        i_base = max(1, min(base_grid.nx-1, Int(floor((x_local - base_grid.x[1]) / base_grid.dx)) + 1))
        j_base = max(1, min(base_grid.ny-1, Int(floor((y_local - base_grid.y[1]) / base_grid.dy)) + 1))
        
        # Bilinear interpolation weights
        x_rel = (x_local - base_grid.x[i_base]) / base_grid.dx
        y_rel = (y_local - base_grid.y[j_base]) / base_grid.dy
        
        # Four corner weights
        w00 = (1 - x_rel) * (1 - y_rel)
        w10 = x_rel * (1 - y_rel)
        w01 = (1 - x_rel) * y_rel
        w11 = x_rel * y_rel
        
        if w00 > 1e-12
            push!(weights, ((i_base, j_base), w00))
        end
        if w10 > 1e-12 && i_base + 1 <= base_grid.nx
            push!(weights, ((i_base + 1, j_base), w10))
        end
        if w01 > 1e-12 && j_base + 1 <= base_grid.ny
            push!(weights, ((i_base, j_base + 1), w01))
        end
        if w11 > 1e-12 && i_base + 1 <= base_grid.nx && j_base + 1 <= base_grid.ny
            push!(weights, ((i_base + 1, j_base + 1), w11))
        end
    end
    
    refined_grid.interpolation_weights[cell_idx] = weights
end

function coarsen_cells!(refined_grid::RefinedGrid, cells_to_coarsen::Vector{Tuple{Int,Int}})
    for (i, j) in cells_to_coarsen
        current_level = get(refined_grid.refined_cells, (i, j), 0)
        
        if current_level > 0
            new_level = current_level - 1
            
            if new_level == 0
                # Return to base grid
                delete!(refined_grid.refined_cells, (i, j))
                delete!(refined_grid.refined_grids, (i, j))
                delete!(refined_grid.interpolation_weights, (i, j))
            else
                # Reduce refinement level
                refined_grid.refined_cells[(i, j)] = new_level
                # Would need to recreate local grid with coarser resolution
            end
            
            println("Coarsened cell ($i, $j) to level $new_level")
        end
    end
end

function interpolate_to_refined_grid(refined_grid::RefinedGrid, base_solution::SolutionState,
                                   cell_idx::Tuple{Int,Int})
    # Interpolate solution from base grid to refined grid for a specific cell
    if !haskey(refined_grid.refined_grids, cell_idx)
        error("No refined grid found for cell $cell_idx")
    end
    
    local_grid = refined_grid.refined_grids[cell_idx]
    weights = refined_grid.interpolation_weights[cell_idx]
    
    # Create local solution state
    local_state = SolutionState2D(local_grid.nx, local_grid.nz)
    
    # Interpolate each variable
    # This is simplified - full implementation would handle staggered grid interpolation properly
    for jj = 1:local_grid.ny, ii = 1:local_grid.nx
        local_idx = (ii - 1) * local_grid.ny + jj
        
        u_interp = 0.0
        v_interp = 0.0
        p_interp = 0.0
        
        for ((i_base, j_base), weight) in weights
            if i_base <= size(base_solution.u, 1) && j_base <= size(base_solution.u, 2)
                u_interp += weight * base_solution.u[i_base, j_base]
            end
            if i_base <= size(base_solution.v, 1) && j_base <= size(base_solution.v, 2)
                v_interp += weight * base_solution.v[i_base, j_base]
            end
            if i_base <= size(base_solution.p, 1) && j_base <= size(base_solution.p, 2)
                p_interp += weight * base_solution.p[i_base, j_base]
            end
        end
        
        # Store interpolated values (simplified assignment)
        if ii <= size(local_state.u, 1) && jj <= size(local_state.u, 2)
            local_state.u[ii, jj] = u_interp
        end
        if ii <= size(local_state.v, 1) && jj <= size(local_state.v, 2)
            local_state.v[ii, jj] = v_interp
        end
        if ii <= size(local_state.p, 1) && jj <= size(local_state.p, 2)
            local_state.p[ii, jj] = p_interp
        end
    end
    
    return local_state
end

function project_to_base_grid!(base_solution::SolutionState, refined_grid::RefinedGrid,
                              local_solutions::Dict{Tuple{Int,Int}, SolutionState})
    # Project solutions from refined grids back to base grid
    
    for ((i, j), local_solution) in local_solutions
        if haskey(refined_grid.refined_grids, (i, j))
            local_grid = refined_grid.refined_grids[(i, j)]
            weights = refined_grid.interpolation_weights[(i, j)]
            
            # Project local solution back to base grid cell (i, j)
            # This involves conservative averaging or restriction operation
            
            # Simplified projection: average over local grid
            u_avg = sum(local_solution.u) / length(local_solution.u)
            v_avg = sum(local_solution.v) / length(local_solution.v)
            p_avg = sum(local_solution.p) / length(local_solution.p)
            
            # Update base grid (need to handle staggered grid properly)
            if i <= size(base_solution.u, 1) && j <= size(base_solution.u, 2)
                base_solution.u[i, j] = u_avg
            end
            if i <= size(base_solution.v, 1) && j <= size(base_solution.v, 2)
                base_solution.v[i, j] = v_avg
            end
            if i <= size(base_solution.p, 1) && j <= size(base_solution.p, 2)
                base_solution.p[i, j] = p_avg
            end
        end
    end
end

function adapt_grid!(refined_grid::RefinedGrid, state::SolutionState,
                    bodies::Union{RigidBodyCollection, FlexibleBodyCollection, Nothing},
                    criteria::AdaptiveRefinementCriteria)
    # Main adaptive refinement routine
    
    # Compute refinement indicators
    indicators = compute_refinement_indicators(refined_grid.base_grid, state, bodies, criteria)
    
    # Mark cells for refinement
    cells_to_refine = mark_cells_for_refinement!(refined_grid, indicators, criteria)
    
    # Refine marked cells
    if !isempty(cells_to_refine)
        refine_cells!(refined_grid, cells_to_refine)
        println("Refined $(length(cells_to_refine)) cells")
    end
    
    # Could also implement coarsening logic here
    # cells_to_coarsen = mark_cells_for_coarsening!(refined_grid, indicators, criteria)
    # coarsen_cells!(refined_grid, cells_to_coarsen)
    
    return length(cells_to_refine)
end

function get_effective_grid_size(refined_grid::RefinedGrid)
    # Calculate effective number of grid points including refinements
    base_points = refined_grid.base_grid.nx * refined_grid.base_grid.ny
    
    refined_points = 0
    for ((i, j), level) in refined_grid.refined_cells
        # Each refined cell replaces 1 base cell with 4^level fine cells
        factor = 4^level
        refined_points += factor - 1  # Subtract 1 for the original base cell
    end
    
    return base_points + refined_points
end