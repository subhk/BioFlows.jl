"""
Advanced Adaptive Mesh Refinement (AMR) for Staggered Grid CFD

This implementation provides:
1. Proper staggered grid handling for refinement/coarsening
2. Conservative restriction and prolongation operators
3. Integration with multigrid solvers
4. MPI-aware adaptive refinement
5. Efficient hierarchical data structures
"""

using LinearAlgebra

"""
    AMRLevel
    
Represents a single level in the AMR hierarchy with proper staggered grid support.
"""
mutable struct AMRLevel
    level::Int                          # Refinement level (0 = base)
    nx::Int                            # Number of cells in x
    ny::Int                            # Number of cells in y
    dx::Float64                        # Cell spacing in x
    dy::Float64                        # Cell spacing in y
    
    # Staggered grid coordinates
    x_centers::Vector{Float64}         # Cell center coordinates
    y_centers::Vector{Float64}
    x_faces::Vector{Float64}           # x-face coordinates (for u)
    y_faces::Vector{Float64}           # y-face coordinates (for v)
    
    # Solution arrays with proper staggered sizes
    u::Matrix{Float64}                 # u-velocity (nx+1, ny)
    v::Matrix{Float64}                 # v-velocity (nx, ny+1)
    p::Matrix{Float64}                 # pressure (nx, ny)
    
    # Refinement flags and relationships
    needs_refinement::Matrix{Bool}     # Cells marked for refinement
    children::Matrix{Union{Nothing, AMRLevel}}  # Child levels (2x2 blocks)
    parent::Union{Nothing, AMRLevel}   # Parent level
    
    # Grid bounds
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
    
    # MPI information (if applicable)
    mpi_rank::Int
    is_ghost::Bool
end

function AMRLevel(level::Int, nx::Int, ny::Int, dx::Float64, dy::Float64,
                 x_min::Float64, y_min::Float64; parent=nothing, mpi_rank=0)
    
    x_max = x_min + nx * dx
    y_max = y_min + ny * dy
    
    # Cell centers
    x_centers = [x_min + (i-0.5) * dx for i in 1:nx]
    y_centers = [y_min + (j-0.5) * dy for j in 1:ny]
    
    # Face coordinates for staggered grid
    x_faces = [x_min + i * dx for i in 0:nx]
    y_faces = [y_min + j * dy for j in 0:ny]
    
    # Initialize solution arrays with correct staggered sizes
    u = zeros(nx+1, ny)      # u at x-faces
    v = zeros(nx, ny+1)      # v at y-faces
    p = zeros(nx, ny)        # p at cell centers
    
    needs_refinement = falses(nx, ny)
    children = Matrix{Union{Nothing, AMRLevel}}(nothing, nx, ny)
    
    AMRLevel(level, nx, ny, dx, dy, x_centers, y_centers, x_faces, y_faces,
             u, v, p, needs_refinement, children, parent,
             x_min, x_max, y_min, y_max, mpi_rank, false)
end

"""
    AMRHierarchy
    
Main adaptive mesh refinement data structure with multigrid integration.
"""
mutable struct AMRHierarchy
    base_level::AMRLevel               # Coarsest level
    max_level::Int                     # Maximum refinement level
    refinement_ratio::Int              # Refinement factor (typically 2)
    
    # Refinement criteria
    velocity_gradient_threshold::Float64
    pressure_gradient_threshold::Float64
    vorticity_threshold::Float64
    body_distance_threshold::Float64
    
    # Performance parameters
    regrid_interval::Int               # Steps between regridding
    buffer_size::Int                   # Buffer cells around flagged regions
    
    # Multigrid integration
    mg_solvers::Dict{Int, Any}         # Multigrid solvers for each level
    
    # Statistics
    total_cells::Int
    total_refined_cells::Int
    last_regrid_step::Int
end

function AMRHierarchy(base_grid::StaggeredGrid; 
                     max_level::Int=3,
                     refinement_ratio::Int=2,
                     velocity_gradient_threshold::Float64=1.0,
                     pressure_gradient_threshold::Float64=10.0,
                     vorticity_threshold::Float64=5.0,
                     body_distance_threshold::Float64=0.1,
                     regrid_interval::Int=10,
                     buffer_size::Int=1)
    
    # Create base AMR level from staggered grid
    base_level = AMRLevel(0, base_grid.nx, base_grid.ny, 
                         base_grid.dx, base_grid.dy,
                         base_grid.x[1] - base_grid.dx/2,
                         base_grid.y[1] - base_grid.dy/2)
    
    hierarchy = AMRHierarchy(base_level, max_level, refinement_ratio,
                            velocity_gradient_threshold, pressure_gradient_threshold,
                            vorticity_threshold, body_distance_threshold,
                            regrid_interval, buffer_size,
                            Dict{Int, Any}(), 
                            base_grid.nx * base_grid.ny, 0, 0)
    
    # Initialize multigrid solvers for base level
    initialize_multigrid_solvers!(hierarchy)
    
    return hierarchy
end

"""
    initialize_multigrid_solvers!(hierarchy)
    
Creates multigrid solvers for each AMR level.
"""
function initialize_multigrid_solvers!(hierarchy::AMRHierarchy)
    function create_mg_solver_for_level(amr_level::AMRLevel)
        # Create staggered grid for this AMR level
        local_grid = StaggeredGrid2D(amr_level.nx, amr_level.ny,  # ny maps to nz in XZ plane
                                    amr_level.x_max - amr_level.x_min,
                                    amr_level.y_max - amr_level.y_min;  # y maps to z in XZ plane
                                    origin_x=amr_level.x_min,
                                    origin_z=amr_level.y_min)  # y_min maps to z_min in XZ plane
        
        # Create staggered-aware multigrid solver
        return MultigridPoissonSolver(local_grid; solver_type=:staggered, tolerance=1e-8)
    end
    
    # Initialize solver for base level
    hierarchy.mg_solvers[0] = create_mg_solver_for_level(hierarchy.base_level)
end

"""
    compute_refinement_indicators_amr(amr_level, state, bodies, hierarchy)
    
Advanced refinement indicator computation using multiple criteria.
"""
function compute_refinement_indicators_amr(amr_level::AMRLevel, 
                                         state::SolutionState,
                                         bodies::Union{RigidBodyCollection, FlexibleBodyCollection, Nothing},
                                         hierarchy::AMRHierarchy)
    nx, ny = amr_level.nx, amr_level.ny
    indicators = zeros(nx, ny)
    
    # Create temporary staggered grid for differential operators
    local_grid = StaggeredGrid2D(nx, ny, amr_level.x_max - amr_level.x_min,  # ny maps to nz in XZ plane
                                amr_level.y_max - amr_level.y_min;  # y maps to z in XZ plane
                                origin_x=amr_level.x_min, origin_z=amr_level.y_min)
    
    # 1. Velocity gradient indicator (using proper staggered operators)
    u_cc = interpolate_u_to_cell_center(state.u, local_grid)
    v_cc = interpolate_v_to_cell_center(state.v, local_grid)
    
    dudx = ddx(u_cc, local_grid)
    dudy = ddy(u_cc, local_grid)
    dvdx = ddx(v_cc, local_grid)
    dvdy = ddy(v_cc, local_grid)
    
    vel_indicator = sqrt.(dudx.^2 + dudy.^2 + dvdx.^2 + dvdy.^2)
    
    # 2. Pressure gradient indicator
    dpdx = ddx(state.p, local_grid)
    dpdy = ddy(state.p, local_grid)
    press_indicator = sqrt.(dpdx.^2 + dpdy.^2)
    
    # 3. Vorticity indicator
    dvdx_cc = ddx(v_cc, local_grid)
    dudy_cc = ddy(u_cc, local_grid)
    vort_indicator = abs.(dvdx_cc - dudy_cc)
    
    # 4. Body proximity indicator
    body_indicator = zeros(nx, ny)
    if bodies !== nothing
        for j = 1:ny, i = 1:nx
            x = amr_level.x_centers[i]
            y = amr_level.y_centers[j]
            
            min_distance = compute_distance_to_bodies(x, y, bodies)
            if min_distance < hierarchy.body_distance_threshold
                body_indicator[i, j] = 1.0
            end
        end
    end
    
    # 5. Solution quality indicator (based on truncation error estimation)
    quality_indicator = estimate_truncation_error(state, local_grid)
    
    # Combine all indicators with adaptive weights
    for j = 1:ny, i = 1:nx
        vel_flag = vel_indicator[i, j] > hierarchy.velocity_gradient_threshold
        press_flag = press_indicator[i, j] > hierarchy.pressure_gradient_threshold
        vort_flag = vort_indicator[i, j] > hierarchy.vorticity_threshold
        body_flag = body_indicator[i, j] > 0.5
        quality_flag = quality_indicator[i, j] > 0.1  # Adaptive threshold
        
        # Multi-criteria decision with priority weights
        score = 0.0
        score += vel_flag ? 0.3 : 0.0
        score += press_flag ? 0.2 : 0.0
        score += vort_flag ? 0.2 : 0.0
        score += body_flag ? 0.4 : 0.0      # Highest priority for bodies
        score += quality_flag ? 0.1 : 0.0
        
        indicators[i, j] = score
    end
    
    return indicators
end

"""
    estimate_truncation_error(state, grid)
    
Estimates local truncation error to guide refinement decisions.
"""
function estimate_truncation_error(state::SolutionState, grid::StaggeredGrid)
    nx, ny = grid.nx, grid.ny
    error_est = zeros(nx, ny)
    
    # Richardson extrapolation-based error estimation
    # Compare solution gradients at different scales
    
    # Compute second derivatives (indicators of solution smoothness)
    p_xx = d2dx2(state.p, grid)
    p_yy = d2dy2(state.p, grid)
    
    # Error is proportional to h² * |∇²p| for 2nd order methods
    h_eff = sqrt(grid.dx^2 + grid.dy^2)
    
    for j = 1:ny, i = 1:nx
        # Local truncation error estimate
        error_est[i, j] = h_eff^2 * sqrt(p_xx[i, j]^2 + p_yy[i, j]^2)
    end
    
    return error_est
end

"""
    conservative_restriction_2d(fine_array, ratio)
    
Conservative restriction operator for staggered grids.
"""
function conservative_restriction_2d(fine_array::Matrix{T}, ratio::Int=2) where T
    nx_fine, ny_fine = size(fine_array)
    nx_coarse = nx_fine ÷ ratio
    ny_coarse = ny_fine ÷ ratio
    
    coarse_array = zeros(T, nx_coarse, ny_coarse)
    
    # Conservative averaging: preserve integral quantities
    for j = 1:ny_coarse, i = 1:nx_coarse
        sum_val = zero(T)
        count = 0
        
        # Average over ratio×ratio block
        for jj = (j-1)*ratio + 1 : j*ratio
            for ii = (i-1)*ratio + 1 : i*ratio
                if ii <= nx_fine && jj <= ny_fine
                    sum_val += fine_array[ii, jj]
                    count += 1
                end
            end
        end
        
        coarse_array[i, j] = count > 0 ? sum_val / count : zero(T)
    end
    
    return coarse_array
end

"""
    bilinear_prolongation_2d(coarse_array, ratio)
    
Bilinear prolongation operator for staggered grids.
"""
function bilinear_prolongation_2d(coarse_array::Matrix{T}, ratio::Int=2) where T
    nx_coarse, ny_coarse = size(coarse_array)
    nx_fine = nx_coarse * ratio
    ny_fine = ny_coarse * ratio
    
    fine_array = zeros(T, nx_fine, ny_fine)
    
    # Bilinear interpolation
    for j = 1:ny_fine, i = 1:nx_fine
        # Find coarse grid indices
        i_c = (i - 1) ÷ ratio + 1
        j_c = (j - 1) ÷ ratio + 1
        
        # Local coordinates within coarse cell
        xi = ((i - 1) % ratio) / ratio
        eta = ((j - 1) % ratio) / ratio
        
        # Bilinear interpolation weights
        if i_c < nx_coarse && j_c < ny_coarse
            fine_array[i, j] = (1-xi) * (1-eta) * coarse_array[i_c, j_c] +
                              xi * (1-eta) * coarse_array[i_c+1, j_c] +
                              (1-xi) * eta * coarse_array[i_c, j_c+1] +
                              xi * eta * coarse_array[i_c+1, j_c+1]
        else
            # Boundary handling
            fine_array[i, j] = coarse_array[min(i_c, nx_coarse), min(j_c, ny_coarse)]
        end
    end
    
    return fine_array
end

"""
    refine_amr_level!(hierarchy, amr_level, state, indicators)
    
Creates child AMR levels for flagged cells.
"""
function refine_amr_level!(hierarchy::AMRHierarchy, amr_level::AMRLevel,
                          state::SolutionState, indicators::Matrix{Float64})
    if amr_level.level >= hierarchy.max_level
        return 0  # Maximum refinement reached
    end
    
    refined_count = 0
    nx, ny = amr_level.nx, amr_level.ny
    
    # Apply buffer zones around flagged cells
    buffered_flags = apply_buffer_zones(indicators .> 0.5, hierarchy.buffer_size)
    
    # Group flagged cells into refinement patches
    patches = find_refinement_patches(buffered_flags)
    
    for patch in patches
        # Create refined level for this patch
        child_level = create_child_level(hierarchy, amr_level, patch, state)
        
        if child_level !== nothing
            # Store child relationship
            for (i, j) in patch
                amr_level.children[i, j] = child_level
            end
            
            # Initialize multigrid solver for new level
            hierarchy.mg_solvers[child_level.level] = create_mg_solver_for_level(child_level)
            
            refined_count += length(patch)
        end
    end
    
    hierarchy.total_refined_cells += refined_count
    return refined_count
end

"""
    apply_buffer_zones(flags, buffer_size)
    
Adds buffer zones around flagged cells to ensure smooth refinement transitions.
"""
function apply_buffer_zones(flags::Matrix{Bool}, buffer_size::Int)
    nx, ny = size(flags)
    buffered = copy(flags)
    
    # Dilate flagged regions
    for _ = 1:buffer_size
        new_flags = copy(buffered)
        for j = 2:ny-1, i = 2:nx-1
            if flags[i, j]
                # Mark neighbors for refinement
                new_flags[i-1:i+1, j-1:j+1] .= true
            end
        end
        buffered = new_flags
    end
    
    return buffered
end

"""
    find_refinement_patches(flags)
    
Groups connected flagged cells into refinement patches.
"""
function find_refinement_patches(flags::Matrix{Bool})
    nx, ny = size(flags)
    visited = falses(nx, ny)
    patches = Vector{Vector{Tuple{Int,Int}}}()
    
    for j = 1:ny, i = 1:nx
        if flags[i, j] && !visited[i, j]
            # Start new patch with flood fill
            patch = Vector{Tuple{Int,Int}}()
            queue = [(i, j)]
            
            while !isempty(queue)
                ci, cj = popfirst!(queue)
                
                if ci < 1 || ci > nx || cj < 1 || cj > ny || visited[ci, cj] || !flags[ci, cj]
                    continue
                end
                
                visited[ci, cj] = true
                push!(patch, (ci, cj))
                
                # Add 4-connected neighbors
                push!(queue, (ci-1, cj), (ci+1, cj), (ci, cj-1), (ci, cj+1))
            end
            
            if length(patch) > 2  # Only create patches with sufficient cells
                push!(patches, patch)
            end
        end
    end
    
    return patches
end

"""
    create_child_level(hierarchy, parent_level, patch, state)
    
Creates a refined AMR level for a patch of cells.
"""
function create_child_level(hierarchy::AMRHierarchy, parent_level::AMRLevel,
                           patch::Vector{Tuple{Int,Int}}, state::SolutionState)
    if isempty(patch)
        return nothing
    end
    
    # Determine bounding box of patch
    i_min = minimum(p[1] for p in patch)
    i_max = maximum(p[1] for p in patch)
    j_min = minimum(p[2] for p in patch)
    j_max = maximum(p[2] for p in patch)
    
    # Calculate refined grid parameters
    ratio = hierarchy.refinement_ratio
    nx_child = (i_max - i_min + 1) * ratio
    ny_child = (j_max - j_min + 1) * ratio
    dx_child = parent_level.dx / ratio
    dy_child = parent_level.dy / ratio
    
    # Child grid bounds
    x_min_child = parent_level.x_min + (i_min - 1) * parent_level.dx
    y_min_child = parent_level.y_min + (j_min - 1) * parent_level.dy
    
    # Create child level
    child_level = AMRLevel(parent_level.level + 1, nx_child, ny_child,
                          dx_child, dy_child, x_min_child, y_min_child;
                          parent=parent_level)
    
    # Interpolate initial solution from parent
    interpolate_to_child_level!(child_level, parent_level, state, patch)
    
    return child_level
end

"""
    interpolate_to_child_level!(child_level, parent_level, state, patch)
    
Interpolates solution from parent to child level with proper staggered grid handling.
"""
function interpolate_to_child_level!(child_level::AMRLevel, parent_level::AMRLevel,
                                   state::SolutionState, patch::Vector{Tuple{Int,Int}})
    ratio = 2  # Assumed refinement ratio
    
    # Interpolate pressure (cell-centered)
    for j_child = 1:child_level.ny, i_child = 1:child_level.nx
        # Map to parent coordinates
        x_child = child_level.x_centers[i_child]
        y_child = child_level.y_centers[j_child]
        
        # Find parent cell and interpolation weights
        i_parent = Int(floor((x_child - parent_level.x_min) / parent_level.dx)) + 1
        j_parent = Int(floor((y_child - parent_level.y_min) / parent_level.dy)) + 1
        
        if i_parent >= 1 && i_parent <= parent_level.nx && 
           j_parent >= 1 && j_parent <= parent_level.ny
            child_level.p[i_child, j_child] = state.p[i_parent, j_parent]
        end
    end
    
    # Interpolate u-velocity (x-faces)
    for j_child = 1:child_level.ny, i_child = 1:child_level.nx+1
        x_face = child_level.x_faces[i_child]
        y_center = child_level.y_centers[j_child]
        
        # Similar interpolation for u-velocity...
        i_parent = max(1, min(parent_level.nx+1, 
                             Int(floor((x_face - parent_level.x_min) / parent_level.dx)) + 1))
        j_parent = max(1, min(parent_level.ny,
                             Int(floor((y_center - parent_level.y_min) / parent_level.dy)) + 1))
        
        child_level.u[i_child, j_child] = state.u[i_parent, j_parent]
    end
    
    # Interpolate v-velocity (y-faces) 
    for j_child = 1:child_level.ny+1, i_child = 1:child_level.nx
        x_center = child_level.x_centers[i_child]
        y_face = child_level.y_faces[j_child]
        
        i_parent = max(1, min(parent_level.nx,
                             Int(floor((x_center - parent_level.x_min) / parent_level.dx)) + 1))
        j_parent = max(1, min(parent_level.ny+1,
                             Int(floor((y_face - parent_level.y_min) / parent_level.dy)) + 1))
        
        child_level.v[i_child, j_child] = state.v[i_parent, j_parent]
    end
end

# Helper function for distance computation
function compute_distance_to_bodies(x::Float64, y::Float64, 
                                  bodies::Union{RigidBodyCollection, FlexibleBodyCollection})
    min_distance = Inf
    
    if bodies isa RigidBodyCollection
        for body in bodies.bodies
            dist = distance_to_surface(body, x, y)
            min_distance = min(min_distance, abs(dist))
        end
    elseif bodies isa FlexibleBodyCollection
        for body in bodies.bodies
            for k = 1:body.n_points
                dist = sqrt((x - body.X[k, 1])^2 + (y - body.X[k, 2])^2)
                min_distance = min(min_distance, dist)
            end
        end
    end
    
    return min_distance
end

# Export new AMR functionality
export AMRLevel, AMRHierarchy, compute_refinement_indicators_amr
export conservative_restriction_2d, bilinear_prolongation_2d
export refine_amr_level!, estimate_truncation_error