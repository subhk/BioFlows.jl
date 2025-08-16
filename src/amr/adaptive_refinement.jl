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
    # FIXED: Type-stable indexing for better performance and consistency
    refined_cells_2d::Dict{Tuple{Int,Int}, Int}  # 2D cell indices -> refinement_level
    refined_cells_3d::Dict{Tuple{Int,Int,Int}, Int}  # 3D cell indices -> refinement_level
    refined_grids_2d::Dict{Tuple{Int,Int}, StaggeredGrid}  # 2D local refined grids
    refined_grids_3d::Dict{Tuple{Int,Int,Int}, StaggeredGrid}  # 3D local refined grids
    interpolation_weights_2d::Dict{Tuple{Int,Int}, Vector{Tuple{Tuple{Int,Int}, Float64}}}
    interpolation_weights_3d::Dict{Tuple{Int,Int,Int}, Vector{Tuple{Tuple{Int,Int,Int}, Float64}}}
end

function RefinedGrid(base_grid::StaggeredGrid)
    RefinedGrid(base_grid, 
               Dict{Tuple{Int,Int}, Int}(), 
               Dict{Tuple{Int,Int,Int}, Int}(),
               Dict{Tuple{Int,Int}, StaggeredGrid}(),
               Dict{Tuple{Int,Int,Int}, StaggeredGrid}(),
               Dict{Tuple{Int,Int}, Vector{Tuple{Tuple{Int,Int}, Float64}}}(),
               Dict{Tuple{Int,Int,Int}, Vector{Tuple{Tuple{Int,Int,Int}, Float64}}}())
end

function compute_refinement_indicators(grid::StaggeredGrid, state::SolutionState, 
                                     bodies::Union{RigidBodyCollection, FlexibleBodyCollection, Nothing},
                                     criteria::AdaptiveRefinementCriteria)
    if grid.grid_type == TwoDimensional
        nx, nz = grid.nx, grid.nz
        indicators = zeros(nx, nz)
    else
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        indicators = zeros(nx, ny, nz)
    end
    
    # Velocity gradient indicator
    velocity_indicator = compute_velocity_gradient_indicator(grid, state)
    
    # Pressure gradient indicator  
    pressure_indicator = compute_pressure_gradient_indicator(grid, state)
    
    # Vorticity indicator
    vorticity_indicator = compute_vorticity_indicator(grid, state)
    
    # Body proximity indicator
    if grid.grid_type == TwoDimensional
        body_indicator = zeros(nx, nz)
    else
        body_indicator = zeros(nx, ny, nz)
    end
    
    if bodies !== nothing
        body_indicator = compute_body_proximity_indicator(grid, bodies, criteria.body_distance_threshold)
    end
    
    # Combine indicators
    if grid.grid_type == TwoDimensional
        for j = 1:nz, i = 1:nx
            vel_flag = velocity_indicator[i, j] > criteria.velocity_gradient_threshold
            press_flag = pressure_indicator[i, j] > criteria.pressure_gradient_threshold
            vort_flag = vorticity_indicator[i, j] > criteria.vorticity_threshold
            body_flag = body_indicator[i, j] > 0.5
            
            if vel_flag || press_flag || vort_flag || body_flag
                indicators[i, j] = 1.0
            end
        end
    else  # 3D case
        for k = 1:nz, j = 1:ny, i = 1:nx
            vel_flag = velocity_indicator[i, j, k] > criteria.velocity_gradient_threshold
            press_flag = pressure_indicator[i, j, k] > criteria.pressure_gradient_threshold
            vort_flag = vorticity_indicator[i, j, k] > criteria.vorticity_threshold
            body_flag = body_indicator[i, j, k] > 0.5
            
            if vel_flag || press_flag || vort_flag || body_flag
                indicators[i, j, k] = 1.0
            end
        end
    end
    
    return indicators
end

function compute_velocity_gradient_indicator(grid::StaggeredGrid, state::SolutionState)
    # FIXED: Consistent coordinate system handling for 2D and 3D
    if grid.grid_type == TwoDimensional
        nx, nz = grid.nx, grid.nz  # FIXED: Use XZ plane for 2D consistency
        indicator = zeros(nx, nz)
        
        # Use proper staggered grid interpolation from differential operators
        u_cc = interpolate_u_to_cell_center(state.u, grid)
        v_cc = interpolate_v_to_cell_center(state.v, grid)  # v represents w-velocity in XZ plane
        
        # Use proper 2nd order differential operators for velocity gradients
        dudx = ddx(u_cc, grid)
        dudz = ddz(u_cc, grid)  # FIXED: Use ddz for XZ plane
        dvdx = ddx(v_cc, grid)
        dvdz = ddz(v_cc, grid)  # FIXED: Use ddz for XZ plane
        
        # Magnitude of velocity gradient tensor for XZ plane
        for j = 1:nz, i = 1:nx
            indicator[i, j] = sqrt(dudx[i, j]^2 + dudz[i, j]^2 + dvdx[i, j]^2 + dvdz[i, j]^2)
        end
    else  # 3D case
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        indicator = zeros(nx, ny, nz)
        
        # 3D velocity gradient computation
        u_cc = interpolate_u_to_cell_center(state.u, grid)
        v_cc = interpolate_v_to_cell_center(state.v, grid)
        w_cc = interpolate_w_to_cell_center(state.w, grid)
        
        dudx = ddx(u_cc, grid)
        dudy = ddy(u_cc, grid)
        dudz = ddz(u_cc, grid)
        dvdx = ddx(v_cc, grid)
        dvdy = ddy(v_cc, grid)
        dvdz = ddz(v_cc, grid)
        dwdx = ddx(w_cc, grid)
        dwdy = ddy(w_cc, grid)
        dwdz = ddz(w_cc, grid)
        
        # Full 3D velocity gradient tensor magnitude
        for k = 1:nz, j = 1:ny, i = 1:nx
            grad_tensor_norm_sq = dudx[i, j, k]^2 + dudy[i, j, k]^2 + dudz[i, j, k]^2 +
                                 dvdx[i, j, k]^2 + dvdy[i, j, k]^2 + dvdz[i, j, k]^2 +
                                 dwdx[i, j, k]^2 + dwdy[i, j, k]^2 + dwdz[i, j, k]^2
            indicator[i, j, k] = sqrt(grad_tensor_norm_sq)
        end
    end
    
    return indicator
end

function compute_pressure_gradient_indicator(grid::StaggeredGrid, state::SolutionState)
    # FIXED: Consistent coordinate system for 2D and 3D
    if grid.grid_type == TwoDimensional
        nx, nz = grid.nx, grid.nz  # FIXED: Use XZ plane
        indicator = zeros(nx, nz)
        
        # Use proper differential operators for pressure gradients in XZ plane
        dpdx = ddx(state.p, grid)
        dpdz = ddz(state.p, grid)  # FIXED: Use ddz for XZ plane
        
        for j = 1:nz, i = 1:nx
            indicator[i, j] = sqrt(dpdx[i, j]^2 + dpdz[i, j]^2)
        end
    else  # 3D case
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        indicator = zeros(nx, ny, nz)
        
        # Full 3D pressure gradient
        dpdx = ddx(state.p, grid)
        dpdy = ddy(state.p, grid)
        dpdz = ddz(state.p, grid)
        
        for k = 1:nz, j = 1:ny, i = 1:nx
            indicator[i, j, k] = sqrt(dpdx[i, j, k]^2 + dpdy[i, j, k]^2 + dpdz[i, j, k]^2)
        end
    end
    
    return indicator
end

function compute_vorticity_indicator(grid::StaggeredGrid, state::SolutionState)
    # FIXED: Proper vorticity computation for 2D XZ plane and 3D
    if grid.grid_type == TwoDimensional
        nx, nz = grid.nx, grid.nz  # FIXED: Use XZ plane
        indicator = zeros(nx, nz)
        
        # Compute vorticity ω_y = ∂u/∂z - ∂w/∂x for XZ plane (y-component)
        u_cc = interpolate_u_to_cell_center(state.u, grid)
        v_cc = interpolate_v_to_cell_center(state.v, grid)  # v represents w-velocity
        
        # Compute derivatives for XZ plane vorticity
        dvdx = ddx(v_cc, grid)  # ∂w/∂x
        dudz = ddz(u_cc, grid)  # ∂u/∂z
        
        for j = 1:nz, i = 1:nx
            indicator[i, j] = abs(dvdx[i, j] - dudz[i, j])  # |ω_y|
        end
    else  # 3D case - compute vorticity magnitude
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        indicator = zeros(nx, ny, nz)
        
        u_cc = interpolate_u_to_cell_center(state.u, grid)
        v_cc = interpolate_v_to_cell_center(state.v, grid)
        w_cc = interpolate_w_to_cell_center(state.w, grid)
        
        # Compute all vorticity components
        dvdz = ddz(v_cc, grid)
        dwdy = ddy(w_cc, grid)
        dwdx = ddx(w_cc, grid)
        dudz = ddz(u_cc, grid)
        dudy = ddy(u_cc, grid)
        dvdx = ddx(v_cc, grid)
        
        for k = 1:nz, j = 1:ny, i = 1:nx
            ω_x = dvdz[i, j, k] - dwdy[i, j, k]
            ω_y = dwdx[i, j, k] - dudz[i, j, k]
            ω_z = dudy[i, j, k] - dvdx[i, j, k]
            indicator[i, j, k] = sqrt(ω_x^2 + ω_y^2 + ω_z^2)
        end
    end
    
    return indicator
end

function compute_body_proximity_indicator(grid::StaggeredGrid, 
                                        bodies::Union{RigidBodyCollection, FlexibleBodyCollection},
                                        distance_threshold::Float64)
    # FIXED: Proper coordinate system handling for 2D and 3D
    if grid.grid_type == TwoDimensional
        nx, nz = grid.nx, grid.nz  # FIXED: Use XZ plane coordinates
        indicator = zeros(nx, nz)
        
        for j = 1:nz, i = 1:nx
            x = grid.x[i]
            z = grid.z[j]  # FIXED: Use z coordinate for XZ plane
            
            min_distance = Inf
            
            if bodies isa RigidBodyCollection
                for body in bodies.bodies
                    # FIXED: Use distance_to_surface_2d for XZ plane
                    dist = distance_to_surface_2d(body, x, z)
                    min_distance = min(min_distance, abs(dist))
                end
            elseif bodies isa FlexibleBodyCollection
                for body in bodies.bodies
                    # For flexible bodies, compute distance to closest Lagrangian point
                    for k = 1:body.n_points
                        # FIXED: Use XZ coordinates for 2D flexible bodies
                        dist = sqrt((x - body.X[k, 1])^2 + (z - body.X[k, 2])^2)
                        min_distance = min(min_distance, dist)
                    end
                end
            end
            
            if min_distance < distance_threshold
                indicator[i, j] = 1.0
            end
        end
    else  # 3D case
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        indicator = zeros(nx, ny, nz)
        
        for k = 1:nz, j = 1:ny, i = 1:nx
            x = grid.x[i]
            y = grid.y[j]
            z = grid.z[k]
            
            min_distance = Inf
            
            if bodies isa RigidBodyCollection
                for body in bodies.bodies
                    dist = distance_to_surface_3d(body, x, y, z)
                    min_distance = min(min_distance, abs(dist))
                end
            elseif bodies isa FlexibleBodyCollection
                for body in bodies.bodies
                    # For flexible bodies, compute distance to closest Lagrangian point
                    for k_pt = 1:body.n_points
                        dist = sqrt((x - body.X[k_pt, 1])^2 + (y - body.X[k_pt, 2])^2 + (z - body.X[k_pt, 3])^2)
                        min_distance = min(min_distance, dist)
                    end
                end
            end
            
            if min_distance < distance_threshold
                indicator[i, j, k] = 1.0
            end
        end
    end
    
    return indicator
end

function mark_cells_for_refinement!(refined_grid::RefinedGrid, indicators::Array{Float64},
                                   criteria::AdaptiveRefinementCriteria)
    # FIXED: Handle both 2D and 3D indicators with proper coordinate system
    grid = refined_grid.base_grid
    
    if grid.grid_type == TwoDimensional
        nx, nz = size(indicators)  # FIXED: XZ plane dimensions
        cells_to_refine = Tuple{Int,Int}[]
        
        for j = 1:nz, i = 1:nx
            if indicators[i, j] > 0.5
                current_level = get(refined_grid.refined_cells_2d, (i, j), 0)
                
                if current_level < criteria.max_refinement_level
                    # Check if resulting grid size would be above minimum
                    base_dx = grid.dx
                    base_dz = grid.dz  # FIXED: Use dz for XZ plane
                    
                    new_dx = base_dx / (2^(current_level + 1))
                    new_dz = base_dz / (2^(current_level + 1))
                    
                    if new_dx >= criteria.min_grid_size && new_dz >= criteria.min_grid_size
                        push!(cells_to_refine, (i, j))
                    end
                end
            end
        end
        
        return cells_to_refine
    else  # 3D case
        nx, ny, nz = size(indicators)
        cells_to_refine = Tuple{Int,Int,Int}[]
        
        for k = 1:nz, j = 1:ny, i = 1:nx
            if indicators[i, j, k] > 0.5
                current_level = get(refined_grid.refined_cells_3d, (i, j, k), 0)
                
                if current_level < criteria.max_refinement_level
                    # Check if resulting grid size would be above minimum
                    base_dx = grid.dx
                    base_dy = grid.dy
                    base_dz = grid.dz
                    
                    new_dx = base_dx / (2^(current_level + 1))
                    new_dy = base_dy / (2^(current_level + 1))
                    new_dz = base_dz / (2^(current_level + 1))
                    
                    if new_dx >= criteria.min_grid_size && new_dy >= criteria.min_grid_size && new_dz >= criteria.min_grid_size
                        push!(cells_to_refine, (i, j, k))
                    end
                end
            end
        end
        
        return cells_to_refine
    end
end

function refine_cells!(refined_grid::RefinedGrid, cells_to_refine::Vector)
    base_grid = refined_grid.base_grid
    
    if base_grid.grid_type == TwoDimensional
        # FIXED: Proper 2D XZ plane refinement
        refine_cells_2d!(refined_grid, cells_to_refine)
    else
        # 3D refinement
        refine_cells_3d!(refined_grid, cells_to_refine)
    end
end

function refine_cells_2d!(refined_grid::RefinedGrid, cells_to_refine::Vector{Tuple{Int,Int}})
    for (i, j) in cells_to_refine
        current_level = get(refined_grid.refined_cells_2d, (i, j), 0)
        new_level = current_level + 1
        
        # Update refinement level
        refined_grid.refined_cells_2d[(i, j)] = new_level
        
        # Create local refined grid for this cell
        base_grid = refined_grid.base_grid
        
        # FIXED: Determine local grid bounds for XZ plane
        x_min = base_grid.x[i] - base_grid.dx/2
        x_max = base_grid.x[i] + base_grid.dx/2
        z_min = base_grid.z[j] - base_grid.dz/2  # FIXED: Use z coordinates
        z_max = base_grid.z[j] + base_grid.dz/2
        
        # Create refined grid with 2^level times finer resolution
        refine_factor = 2^new_level
        local_nx = 2 * refine_factor
        local_nz = 2 * refine_factor  # FIXED: Use nz for XZ plane
        
        Lx_local = x_max - x_min
        Lz_local = z_max - z_min  # FIXED: Use Lz for XZ plane
        
        # FIXED: Create proper 2D XZ plane grid
        local_grid = StaggeredGrid2D(local_nx, local_nz, Lx_local, Lz_local;
                                   origin_x=x_min, origin_z=z_min)
        
        refined_grid.refined_grids_2d[(i, j)] = local_grid
        
        # Compute interpolation weights for coupling with base grid
        compute_interpolation_weights_2d!(refined_grid, (i, j), local_grid)
        
        println("Refined 2D cell ($i, $j) to level $new_level")
    end
end

function refine_cells_3d!(refined_grid::RefinedGrid, cells_to_refine::Vector{Tuple{Int,Int,Int}})
    for (i, j, k) in cells_to_refine
        current_level = get(refined_grid.refined_cells_3d, (i, j, k), 0)
        new_level = current_level + 1
        
        # Update refinement level
        refined_grid.refined_cells_3d[(i, j, k)] = new_level
        
        # Create local refined grid for this cell
        base_grid = refined_grid.base_grid
        
        # Determine local grid bounds for 3D
        x_min = base_grid.x[i] - base_grid.dx/2
        x_max = base_grid.x[i] + base_grid.dx/2
        y_min = base_grid.y[j] - base_grid.dy/2
        y_max = base_grid.y[j] + base_grid.dy/2
        z_min = base_grid.z[k] - base_grid.dz/2
        z_max = base_grid.z[k] + base_grid.dz/2
        
        # Create refined grid with 2^level times finer resolution
        refine_factor = 2^new_level
        local_nx = 2 * refine_factor
        local_ny = 2 * refine_factor
        local_nz = 2 * refine_factor
        
        Lx_local = x_max - x_min
        Ly_local = y_max - y_min
        Lz_local = z_max - z_min
        
        local_grid = StaggeredGrid3D(local_nx, local_ny, local_nz, Lx_local, Ly_local, Lz_local;
                                   origin_x=x_min, origin_y=y_min, origin_z=z_min)
        
        refined_grid.refined_grids_3d[(i, j, k)] = local_grid
        
        # Compute interpolation weights for coupling with base grid
        compute_interpolation_weights_3d!(refined_grid, (i, j, k), local_grid)
        
        println("Refined 3D cell ($i, $j, $k) to level $new_level")
    end
end

# FIXED: Split into separate 2D and 3D functions for type stability
function compute_interpolation_weights_2d!(refined_grid::RefinedGrid, cell_idx::Tuple{Int,Int},
                                         local_grid::StaggeredGrid)
    # Compute interpolation weights for transferring data between base grid and refined grid (XZ plane)
    base_grid = refined_grid.base_grid
    i, j = cell_idx
    
    weights = Vector{Tuple{Tuple{Int,Int}, Float64}}()
    
    # For each point in the local refined grid, find interpolation weights from base grid
    for jj = 1:local_grid.nz, ii = 1:local_grid.nx  # FIXED: Use nz for XZ plane
        x_local = local_grid.x[ii]
        z_local = local_grid.z[jj]  # FIXED: Use z coordinate
        
        # Find base grid cell containing this point
        i_base = max(1, min(base_grid.nx-1, Int(floor((x_local - base_grid.x[1]) / base_grid.dx)) + 1))
        j_base = max(1, min(base_grid.nz-1, Int(floor((z_local - base_grid.z[1]) / base_grid.dz)) + 1))  # FIXED: Use nz and dz
        
        # Bilinear interpolation weights
        x_rel = (x_local - base_grid.x[i_base]) / base_grid.dx
        z_rel = (z_local - base_grid.z[j_base]) / base_grid.dz  # FIXED: Use z coordinate
        
        # Four corner weights for XZ plane
        w00 = (1 - x_rel) * (1 - z_rel)
        w10 = x_rel * (1 - z_rel)
        w01 = (1 - x_rel) * z_rel
        w11 = x_rel * z_rel
        
        if w00 > 1e-12
            push!(weights, ((i_base, j_base), w00))
        end
        if w10 > 1e-12 && i_base + 1 <= base_grid.nx
            push!(weights, ((i_base + 1, j_base), w10))
        end
        if w01 > 1e-12 && j_base + 1 <= base_grid.nz  # FIXED: Use nz
            push!(weights, ((i_base, j_base + 1), w01))
        end
        if w11 > 1e-12 && i_base + 1 <= base_grid.nx && j_base + 1 <= base_grid.nz  # FIXED: Use nz
            push!(weights, ((i_base + 1, j_base + 1), w11))
        end
    end
    
    refined_grid.interpolation_weights_2d[cell_idx] = weights
end

function compute_interpolation_weights_3d!(refined_grid::RefinedGrid, cell_idx::Tuple{Int,Int,Int},
                                         local_grid::StaggeredGrid)
    # Compute interpolation weights for 3D case
    base_grid = refined_grid.base_grid
    i, j, k = cell_idx
    
    weights = Vector{Tuple{Tuple{Int,Int,Int}, Float64}}()
    
    # For each point in the local refined grid, find interpolation weights from base grid
    for kk = 1:local_grid.nz, jj = 1:local_grid.ny, ii = 1:local_grid.nx
        x_local = local_grid.x[ii]
        y_local = local_grid.y[jj]
        z_local = local_grid.z[kk]
        
        # Find base grid cell containing this point
        i_base = max(1, min(base_grid.nx-1, Int(floor((x_local - base_grid.x[1]) / base_grid.dx)) + 1))
        j_base = max(1, min(base_grid.ny-1, Int(floor((y_local - base_grid.y[1]) / base_grid.dy)) + 1))
        k_base = max(1, min(base_grid.nz-1, Int(floor((z_local - base_grid.z[1]) / base_grid.dz)) + 1))
        
        # Trilinear interpolation weights
        x_rel = (x_local - base_grid.x[i_base]) / base_grid.dx
        y_rel = (y_local - base_grid.y[j_base]) / base_grid.dy
        z_rel = (z_local - base_grid.z[k_base]) / base_grid.dz
        
        # Eight corner weights for 3D trilinear interpolation
        w000 = (1 - x_rel) * (1 - y_rel) * (1 - z_rel)
        w100 = x_rel * (1 - y_rel) * (1 - z_rel)
        w010 = (1 - x_rel) * y_rel * (1 - z_rel)
        w110 = x_rel * y_rel * (1 - z_rel)
        w001 = (1 - x_rel) * (1 - y_rel) * z_rel
        w101 = x_rel * (1 - y_rel) * z_rel
        w011 = (1 - x_rel) * y_rel * z_rel
        w111 = x_rel * y_rel * z_rel
        
        # Add non-zero weights
        weights_3d = [(w000, (i_base, j_base, k_base)),
                      (w100, (i_base + 1, j_base, k_base)),
                      (w010, (i_base, j_base + 1, k_base)),
                      (w110, (i_base + 1, j_base + 1, k_base)),
                      (w001, (i_base, j_base, k_base + 1)),
                      (w101, (i_base + 1, j_base, k_base + 1)),
                      (w011, (i_base, j_base + 1, k_base + 1)),
                      (w111, (i_base + 1, j_base + 1, k_base + 1))]
        
        for (weight, (ib, jb, kb)) in weights_3d
            if weight > 1e-12 && ib <= base_grid.nx && jb <= base_grid.ny && kb <= base_grid.nz
                push!(weights, ((ib, jb, kb), weight))
            end
        end
    end
    
    refined_grid.interpolation_weights_3d[cell_idx] = weights
end

# FIXED: Separate 2D and 3D coarsening functions with proper field access
function coarsen_cells_2d!(refined_grid::RefinedGrid, cells_to_coarsen::Vector{Tuple{Int,Int}})
    for (i, j) in cells_to_coarsen
        current_level = get(refined_grid.refined_cells_2d, (i, j), 0)
        
        if current_level > 0
            new_level = current_level - 1
            
            if new_level == 0
                # Return to base grid
                delete!(refined_grid.refined_cells_2d, (i, j))
                delete!(refined_grid.refined_grids_2d, (i, j))
                delete!(refined_grid.interpolation_weights_2d, (i, j))
            else
                # Reduce refinement level
                refined_grid.refined_cells_2d[(i, j)] = new_level
                # Would need to recreate local grid with coarser resolution
            end
            
            println("Coarsened 2D cell ($i, $j) to level $new_level")
        end
    end
end

function coarsen_cells_3d!(refined_grid::RefinedGrid, cells_to_coarsen::Vector{Tuple{Int,Int,Int}})
    for (i, j, k) in cells_to_coarsen
        current_level = get(refined_grid.refined_cells_3d, (i, j, k), 0)
        
        if current_level > 0
            new_level = current_level - 1
            
            if new_level == 0
                # Return to base grid
                delete!(refined_grid.refined_cells_3d, (i, j, k))
                delete!(refined_grid.refined_grids_3d, (i, j, k))
                delete!(refined_grid.interpolation_weights_3d, (i, j, k))
            else
                # Reduce refinement level
                refined_grid.refined_cells_3d[(i, j, k)] = new_level
                # Would need to recreate local grid with coarser resolution
            end
            
            println("Coarsened 3D cell ($i, $j, $k) to level $new_level")
        end
    end
end

# FIXED: Separate 2D and 3D interpolation functions with proper coordinate handling
function interpolate_to_refined_grid_2d(refined_grid::RefinedGrid, base_solution::SolutionState,
                                       cell_idx::Tuple{Int,Int})
    # Interpolate solution from base grid to refined grid for a specific 2D cell
    if !haskey(refined_grid.refined_grids_2d, cell_idx)
        error("No refined grid found for 2D cell $cell_idx")
    end
    
    local_grid = refined_grid.refined_grids_2d[cell_idx]
    weights = refined_grid.interpolation_weights_2d[cell_idx]
    
    # Create local solution state for XZ plane
    local_state = SolutionState2D(local_grid.nx, local_grid.nz)
    
    # Interpolate each variable with proper staggered grid handling
    for jj = 1:local_grid.nz, ii = 1:local_grid.nx
        u_interp = 0.0
        v_interp = 0.0  # v represents w-velocity in XZ plane
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
        
        # Store interpolated values with bounds checking
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

function interpolate_to_refined_grid_3d(refined_grid::RefinedGrid, base_solution::SolutionState,
                                       cell_idx::Tuple{Int,Int,Int})
    # Interpolate solution from base grid to refined grid for a specific 3D cell
    if !haskey(refined_grid.refined_grids_3d, cell_idx)
        error("No refined grid found for 3D cell $cell_idx")
    end
    
    local_grid = refined_grid.refined_grids_3d[cell_idx]
    weights = refined_grid.interpolation_weights_3d[cell_idx]
    
    # Create local solution state for 3D
    local_state = SolutionState3D(local_grid.nx, local_grid.ny, local_grid.nz)
    
    # Interpolate each variable with proper staggered grid handling
    for kk = 1:local_grid.nz, jj = 1:local_grid.ny, ii = 1:local_grid.nx
        u_interp = 0.0
        v_interp = 0.0
        w_interp = 0.0
        p_interp = 0.0
        
        for ((i_base, j_base, k_base), weight) in weights
            if i_base <= size(base_solution.u, 1) && j_base <= size(base_solution.u, 2) && k_base <= size(base_solution.u, 3)
                u_interp += weight * base_solution.u[i_base, j_base, k_base]
            end
            if i_base <= size(base_solution.v, 1) && j_base <= size(base_solution.v, 2) && k_base <= size(base_solution.v, 3)
                v_interp += weight * base_solution.v[i_base, j_base, k_base]
            end
            if i_base <= size(base_solution.w, 1) && j_base <= size(base_solution.w, 2) && k_base <= size(base_solution.w, 3)
                w_interp += weight * base_solution.w[i_base, j_base, k_base]
            end
            if i_base <= size(base_solution.p, 1) && j_base <= size(base_solution.p, 2) && k_base <= size(base_solution.p, 3)
                p_interp += weight * base_solution.p[i_base, j_base, k_base]
            end
        end
        
        # Store interpolated values with bounds checking
        if ii <= size(local_state.u, 1) && jj <= size(local_state.u, 2) && kk <= size(local_state.u, 3)
            local_state.u[ii, jj, kk] = u_interp
        end
        if ii <= size(local_state.v, 1) && jj <= size(local_state.v, 2) && kk <= size(local_state.v, 3)
            local_state.v[ii, jj, kk] = v_interp
        end
        if ii <= size(local_state.w, 1) && jj <= size(local_state.w, 2) && kk <= size(local_state.w, 3)
            local_state.w[ii, jj, kk] = w_interp
        end
        if ii <= size(local_state.p, 1) && jj <= size(local_state.p, 2) && kk <= size(local_state.p, 3)
            local_state.p[ii, jj, kk] = p_interp
        end
    end
    
    return local_state
end

# FIXED: Separate 2D and 3D projection functions with proper field access
function project_to_base_grid_2d!(base_solution::SolutionState, refined_grid::RefinedGrid,
                                 local_solutions::Dict{Tuple{Int,Int}, SolutionState})
    # Project solutions from refined grids back to base grid (2D XZ plane)
    
    for ((i, j), local_solution) in local_solutions
        if haskey(refined_grid.refined_grids_2d, (i, j))
            local_grid = refined_grid.refined_grids_2d[(i, j)]
            weights = refined_grid.interpolation_weights_2d[(i, j)]
            
            # Project local solution back to base grid cell (i, j)
            # This involves conservative averaging or restriction operation
            
            # Conservative projection: volume-weighted average over local grid
            u_avg = sum(local_solution.u) / length(local_solution.u)
            v_avg = sum(local_solution.v) / length(local_solution.v)  # v represents w in XZ plane
            p_avg = sum(local_solution.p) / length(local_solution.p)
            
            # Update base grid with bounds checking
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

function project_to_base_grid_3d!(base_solution::SolutionState, refined_grid::RefinedGrid,
                                 local_solutions::Dict{Tuple{Int,Int,Int}, SolutionState})
    # Project solutions from refined grids back to base grid (3D)
    
    for ((i, j, k), local_solution) in local_solutions
        if haskey(refined_grid.refined_grids_3d, (i, j, k))
            local_grid = refined_grid.refined_grids_3d[(i, j, k)]
            weights = refined_grid.interpolation_weights_3d[(i, j, k)]
            
            # Conservative projection: volume-weighted average over local grid
            u_avg = sum(local_solution.u) / length(local_solution.u)
            v_avg = sum(local_solution.v) / length(local_solution.v)
            w_avg = sum(local_solution.w) / length(local_solution.w)
            p_avg = sum(local_solution.p) / length(local_solution.p)
            
            # Update base grid with bounds checking
            if i <= size(base_solution.u, 1) && j <= size(base_solution.u, 2) && k <= size(base_solution.u, 3)
                base_solution.u[i, j, k] = u_avg
            end
            if i <= size(base_solution.v, 1) && j <= size(base_solution.v, 2) && k <= size(base_solution.v, 3)
                base_solution.v[i, j, k] = v_avg
            end
            if i <= size(base_solution.w, 1) && j <= size(base_solution.w, 2) && k <= size(base_solution.w, 3)
                base_solution.w[i, j, k] = w_avg
            end
            if i <= size(base_solution.p, 1) && j <= size(base_solution.p, 2) && k <= size(base_solution.p, 3)
                base_solution.p[i, j, k] = p_avg
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

# Include helper functions
include("amr_helpers.jl")

function get_effective_grid_size(refined_grid::RefinedGrid)
    # Calculate effective number of grid points including refinements
    base_grid = refined_grid.base_grid
    
    if base_grid.grid_type == TwoDimensional
        base_points = base_grid.nx * base_grid.nz  # FIXED: Use nz for XZ plane
        
        refined_points = 0
        for ((i, j), level) in refined_grid.refined_cells_2d
            # Each refined cell replaces 1 base cell with 4^level fine cells (2D)
            factor = 4^level
            refined_points += factor - 1  # Subtract 1 for the original base cell
        end
        
        return base_points + refined_points
    else  # 3D case
        base_points = base_grid.nx * base_grid.ny * base_grid.nz
        
        refined_points = 0
        for ((i, j, k), level) in refined_grid.refined_cells_3d
            # Each refined cell replaces 1 base cell with 8^level fine cells (3D)
            factor = 8^level
            refined_points += factor - 1  # Subtract 1 for the original base cell
        end
        
        return base_points + refined_points
    end
end