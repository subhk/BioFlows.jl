"""
    Body Refinement Indicator for BioFlows.jl

Computes refinement indicators based on body proximity and flow gradients.
Used by the AMR system to determine which cells need refinement.
"""

# Helper function for body proximity indicator
@inline function _body_indicator_value(body::AbstractBody, I::CartesianIndex, T::Type, threshold, t)
    x = loc(0, I, T)
    d = sdf(body, x, T(t))
    abs(d) < threshold ? one(T) : zero(T)
end

"""
    compute_body_refinement_indicator(flow::Flow, body::AbstractBody;
                                      distance_threshold=2.0, t=0.0)

Compute a refinement indicator based on proximity to the immersed body.
Returns an array with values 1.0 where refinement is needed (near body),
and 0.0 elsewhere.

# Arguments
- `flow`: The BioFlows Flow struct
- `body`: The immersed body (AbstractBody)
- `distance_threshold`: Distance in grid cells within which to refine (default: 2.0)
- `t`: Current simulation time (for moving bodies)

# Returns
- Array of same size as `flow.p` with indicator values in [0, 1]
"""
function compute_body_refinement_indicator(flow::Flow{N,T}, body::AbstractBody;
                                           distance_threshold::Real=2.0,
                                           t::Real=0.0) where {N,T}
    indicator = similar(flow.p)
    fill!(indicator, zero(T))
    threshold = T(distance_threshold)
    tt = T(t)

    # Iterate over interior cells (GPU-compatible via @loop)
    R = inside(flow.p)
    @loop indicator[I] = _body_indicator_value(body, I, T, threshold, tt) over I ∈ R

    return indicator
end

# Helper for 2D velocity gradient at cell center
# Uses stencils consistent with Metrics.jl ∂(i,j,I,u) function
@inline function _velocity_gradient_2d(u, I, inv_dx, inv_dz, inv4_dx, inv4_dz)
    # ∂u/∂x at cell center (forward difference on staggered grid)
    dudx = (u[I+δ(1,I), 1] - u[I, 1]) * inv_dx
    # ∂u/∂z at cell center (4-point stencil)
    dudz = (u[I+δ(2,I), 1] + u[I+δ(2,I)+δ(1,I), 1] -
            u[I-δ(2,I), 1] - u[I-δ(2,I)+δ(1,I), 1]) * inv4_dz
    # ∂w/∂x at cell center (4-point stencil)
    dwdx = (u[I+δ(1,I), 2] + u[I+δ(1,I)+δ(2,I), 2] -
            u[I-δ(1,I), 2] - u[I-δ(1,I)+δ(2,I), 2]) * inv4_dx
    # ∂w/∂z at cell center (forward difference on staggered grid)
    dwdz = (u[I+δ(2,I), 2] - u[I, 2]) * inv_dz

    sqrt(dudx^2 + dudz^2 + dwdx^2 + dwdz^2)
end

"""
    compute_velocity_gradient_indicator(flow::Flow; threshold=nothing)

Compute a refinement indicator based on velocity gradient magnitude.
High gradients indicate regions that benefit from finer resolution.
Derivatives are scaled by `flow.Δx`.

# Returns
- Array of same size as `flow.p` with gradient magnitude values.
  If `threshold` is provided, values are binarized to 0/1.
"""
function compute_velocity_gradient_indicator(flow::Flow{N,T};
                                             threshold::Union{Nothing,Real}=nothing) where {N,T}
    indicator = similar(flow.p)
    fill!(indicator, zero(T))
    u = flow.u
    Δx = flow.Δx
    R = inside(flow.p)

    if N == 2
        inv_dx = inv(Δx[1])
        inv_dz = inv(Δx[2])
        inv4_dx = inv(4 * Δx[1])
        inv4_dz = inv(4 * Δx[2])
        # GPU-compatible via @loop
        @loop indicator[I] = _velocity_gradient_2d(u, I, inv_dx, inv_dz, inv4_dx, inv4_dz) over I ∈ R
    else  # N == 3
        # Uses stencils consistent with Metrics.jl ∂(i,j,I,u) function
        invΔx = ntuple(d -> inv(Δx[d]), N)
        inv4Δx = ntuple(d -> inv(4 * Δx[d]), N)
        # 3D gradient computation (GPU-compatible via @loop)
        @loop indicator[I] = _velocity_gradient_3d(u, I, invΔx, inv4Δx) over I ∈ R
    end

    if threshold !== nothing
        thresh = T(threshold)
        # GPU-compatible threshold binarization
        @loop indicator[I] = indicator[I] > thresh ? one(T) : zero(T) over I ∈ R
    end

    return indicator
end

# Helper for 3D velocity gradient at cell center
@inline function _velocity_gradient_3d(u, I, invΔx, inv4Δx)
    T = eltype(u)
    grad_sq = zero(T)
    # Unroll the loops for GPU compatibility
    # Diagonal terms (forward difference)
    grad_sq += ((u[I+δ(1,I), 1] - u[I, 1]) * invΔx[1])^2
    grad_sq += ((u[I+δ(2,I), 2] - u[I, 2]) * invΔx[2])^2
    grad_sq += ((u[I+δ(3,I), 3] - u[I, 3]) * invΔx[3])^2
    # Cross terms (4-point stencil) - du_i/dx_j for i≠j
    # du1/dx2
    grad_sq += ((u[I+δ(2,I), 1] + u[I+δ(2,I)+δ(1,I), 1] -
                 u[I-δ(2,I), 1] - u[I-δ(2,I)+δ(1,I), 1]) * inv4Δx[2])^2
    # du1/dx3
    grad_sq += ((u[I+δ(3,I), 1] + u[I+δ(3,I)+δ(1,I), 1] -
                 u[I-δ(3,I), 1] - u[I-δ(3,I)+δ(1,I), 1]) * inv4Δx[3])^2
    # du2/dx1
    grad_sq += ((u[I+δ(1,I), 2] + u[I+δ(1,I)+δ(2,I), 2] -
                 u[I-δ(1,I), 2] - u[I-δ(1,I)+δ(2,I), 2]) * inv4Δx[1])^2
    # du2/dx3
    grad_sq += ((u[I+δ(3,I), 2] + u[I+δ(3,I)+δ(2,I), 2] -
                 u[I-δ(3,I), 2] - u[I-δ(3,I)+δ(2,I), 2]) * inv4Δx[3])^2
    # du3/dx1
    grad_sq += ((u[I+δ(1,I), 3] + u[I+δ(1,I)+δ(3,I), 3] -
                 u[I-δ(1,I), 3] - u[I-δ(1,I)+δ(3,I), 3]) * inv4Δx[1])^2
    # du3/dx2
    grad_sq += ((u[I+δ(2,I), 3] + u[I+δ(2,I)+δ(3,I), 3] -
                 u[I-δ(2,I), 3] - u[I-δ(2,I)+δ(3,I), 3]) * inv4Δx[2])^2
    sqrt(grad_sq)
end

# Helper for 2D vorticity at cell center
# ω = ∂w/∂x - ∂u/∂z (out-of-plane component)
# Uses stencils consistent with Metrics.jl ∂(i,j,I,u) function
@inline function _vorticity_2d(u, I, inv4_dx, inv4_dz)
    # ∂w/∂x at cell center (4-point stencil)
    dwdx = (u[I+δ(1,I), 2] + u[I+δ(1,I)+δ(2,I), 2] -
            u[I-δ(1,I), 2] - u[I-δ(1,I)+δ(2,I), 2]) * inv4_dx
    # ∂u/∂z at cell center (4-point stencil)
    dudz = (u[I+δ(2,I), 1] + u[I+δ(2,I)+δ(1,I), 1] -
            u[I-δ(2,I), 1] - u[I-δ(2,I)+δ(1,I), 1]) * inv4_dz
    abs(dwdx - dudz)
end

"""
    compute_vorticity_indicator(flow::Flow; threshold=nothing)

Compute a refinement indicator based on vorticity magnitude.
Useful for capturing vortex shedding and wake structures.
Derivatives are scaled by `flow.Δx`.

# Returns
- Array of same size as `flow.p` with vorticity magnitude.
  If `threshold` is provided, values are binarized to 0/1.
"""
function compute_vorticity_indicator(flow::Flow{N,T};
                                     threshold::Union{Nothing,Real}=nothing) where {N,T}
    indicator = similar(flow.p)
    fill!(indicator, zero(T))
    u = flow.u
    Δx = flow.Δx
    R = inside(flow.p)

    if N == 2
        inv4_dx = inv(4 * Δx[1])
        inv4_dz = inv(4 * Δx[2])
        # GPU-compatible via @loop
        @loop indicator[I] = _vorticity_2d(u, I, inv4_dx, inv4_dz) over I ∈ R
    else  # N == 3
        # Uses stencils consistent with Metrics.jl ∂(i,j,I,u) function
        inv4_dx = inv(4 * Δx[1])
        inv4_dy = inv(4 * Δx[2])
        inv4_dz = inv(4 * Δx[3])
        # GPU-compatible via @loop
        @loop indicator[I] = _vorticity_3d(u, I, inv4_dx, inv4_dy, inv4_dz) over I ∈ R
    end

    if threshold !== nothing
        thresh = T(threshold)
        # GPU-compatible threshold binarization
        @loop indicator[I] = indicator[I] > thresh ? one(T) : zero(T) over I ∈ R
    end

    return indicator
end

# Helper for 3D vorticity at cell center
# ω = (ωₓ, ωᵧ, ω_z) using 4-point stencils
@inline function _vorticity_3d(u, I, inv4_dx, inv4_dy, inv4_dz)
    # ωₓ = ∂w/∂y - ∂v/∂z (4-point stencils at cell center)
    dwdy = (u[I+δ(2,I), 3] + u[I+δ(2,I)+δ(3,I), 3] -
            u[I-δ(2,I), 3] - u[I-δ(2,I)+δ(3,I), 3]) * inv4_dy
    dvdz = (u[I+δ(3,I), 2] + u[I+δ(3,I)+δ(2,I), 2] -
            u[I-δ(3,I), 2] - u[I-δ(3,I)+δ(2,I), 2]) * inv4_dz
    omega_x = dwdy - dvdz

    # ωᵧ = ∂u/∂z - ∂w/∂x (4-point stencils at cell center)
    dudz = (u[I+δ(3,I), 1] + u[I+δ(3,I)+δ(1,I), 1] -
            u[I-δ(3,I), 1] - u[I-δ(3,I)+δ(1,I), 1]) * inv4_dz
    dwdx = (u[I+δ(1,I), 3] + u[I+δ(1,I)+δ(3,I), 3] -
            u[I-δ(1,I), 3] - u[I-δ(1,I)+δ(3,I), 3]) * inv4_dx
    omega_y = dudz - dwdx

    # ω_z = ∂v/∂x - ∂u/∂y (4-point stencils at cell center)
    dvdx = (u[I+δ(1,I), 2] + u[I+δ(1,I)+δ(2,I), 2] -
            u[I-δ(1,I), 2] - u[I-δ(1,I)+δ(2,I), 2]) * inv4_dx
    dudy = (u[I+δ(2,I), 1] + u[I+δ(2,I)+δ(1,I), 1] -
            u[I-δ(2,I), 1] - u[I-δ(2,I)+δ(1,I), 1]) * inv4_dy
    omega_z = dvdx - dudy

    sqrt(omega_x^2 + omega_y^2 + omega_z^2)
end

"""
    compute_combined_indicator(flow::Flow, body::AbstractBody;
                               body_threshold=2.0, gradient_threshold=1.0,
                               vorticity_threshold=1.0, t=0.0,
                               body_weight=0.5, gradient_weight=0.3, vorticity_weight=0.2)

Compute a combined refinement indicator using body proximity, velocity gradients,
and vorticity. Returns values in [0, 1] where higher values indicate stronger
need for refinement.

# Arguments
- `flow`: The BioFlows Flow struct
- `body`: The immersed body
- `body_threshold`: Distance threshold for body proximity (grid cells)
- `gradient_threshold`: Threshold for velocity gradient magnitude
- `vorticity_threshold`: Threshold for vorticity magnitude
- `t`: Current simulation time
- `body_weight`: Weight for body proximity indicator
- `gradient_weight`: Weight for velocity gradient indicator
- `vorticity_weight`: Weight for vorticity indicator

# Returns
- Combined indicator array with values in [0, 1]
"""
function compute_combined_indicator(flow::Flow{N,T}, body::AbstractBody;
                                    body_threshold::Real=2.0,
                                    gradient_threshold::Real=1.0,
                                    vorticity_threshold::Real=1.0,
                                    t::Real=0.0,
                                    body_weight::Real=0.5,
                                    gradient_weight::Real=0.3,
                                    vorticity_weight::Real=0.2) where {N,T}

    # Normalize weights
    total_weight = body_weight + gradient_weight + vorticity_weight
    bw = T(body_weight / total_weight)
    gw = T(gradient_weight / total_weight)
    vw = T(vorticity_weight / total_weight)

    # Compute individual indicators
    body_ind = compute_body_refinement_indicator(flow, body;
                                                  distance_threshold=body_threshold, t=t)
    grad_ind = compute_velocity_gradient_indicator(flow)
    vort_ind = compute_vorticity_indicator(flow)

    # Threshold-based activation
    gt = T(gradient_threshold)
    vt = T(vorticity_threshold)

    # Combine with weights (GPU-compatible via @loop)
    combined = similar(flow.p)
    fill!(combined, zero(T))
    R = inside(flow.p)
    @loop combined[I] = bw * body_ind[I] +
                        gw * (grad_ind[I] > gt ? one(T) : zero(T)) +
                        vw * (vort_ind[I] > vt ? one(T) : zero(T)) over I ∈ R

    # Set boundary values to zero
    if N == 2
        fill!(view(combined, 1, :), zero(T))
        fill!(view(combined, size(combined, 1), :), zero(T))
        fill!(view(combined, :, 1), zero(T))
        fill!(view(combined, :, size(combined, 2)), zero(T))
    else  # N == 3
        fill!(view(combined, 1, :, :), zero(T))
        fill!(view(combined, size(combined, 1), :, :), zero(T))
        fill!(view(combined, :, 1, :), zero(T))
        fill!(view(combined, :, size(combined, 2), :), zero(T))
        fill!(view(combined, :, :, 1), zero(T))
        fill!(view(combined, :, :, size(combined, 3)), zero(T))
    end

    return combined
end

"""
    mark_cells_for_refinement(indicator::AbstractArray, threshold::Real=0.5)

Mark cells for refinement based on the indicator values.
Returns a vector of cell indices that should be refined.

# Arguments
- `indicator`: Array of indicator values
- `threshold`: Cells with indicator > threshold are marked for refinement

# Returns
- Vector of CartesianIndex for cells to refine
"""
function mark_cells_for_refinement(indicator::AbstractArray{T,N};
                                   threshold::Real=0.5) where {T,N}
    cells = CartesianIndex{N}[]
    thresh = T(threshold)

    for I in inside(indicator)
        # Use >= to include cells at exactly the threshold
        # This is important because body-only cells with default weights
        # will have exactly 0.5 indicator value (0.5 weight × 1.0 indicator)
        if indicator[I] >= thresh
            push!(cells, I)
        end
    end

    return cells
end

"""
    apply_buffer_zone!(indicator::AbstractArray, buffer_size::Int=1)

Expand refinement indicator to include buffer cells around marked regions.
This helps maintain smooth transitions between refinement levels.

# Arguments
- `indicator`: Array of indicator values (modified in place)
- `buffer_size`: Number of buffer cells to add around refined regions
"""
function apply_buffer_zone!(indicator::AbstractArray{T,N};
                            buffer_size::Int=1) where {T,N}
    original = copy(indicator)
    dims = size(indicator)

    for I in CartesianIndices(indicator)
        if original[I] > 0
            # Mark neighboring cells within buffer
            for offset in CartesianIndices(ntuple(_ -> -buffer_size:buffer_size, N))
                neighbor = I + offset
                if checkbounds(Bool, indicator, neighbor)
                    indicator[neighbor] = max(indicator[neighbor], original[I])
                end
            end
        end
    end

    return indicator
end

"""
    smooth_indicator!(indicator::AbstractArray; iterations::Int=1)

Smooth the refinement indicator using averaging.
Helps create gradual transitions between refinement levels.
"""
function smooth_indicator!(indicator::AbstractArray{T,N};
                           iterations::Int=1) where {T,N}
    for _ in 1:iterations
        temp = copy(indicator)
        for I in CartesianIndices(indicator)
            if all(i -> 2 <= I[i] <= size(indicator, i) - 1, 1:N)
                # Average with neighbors
                sum_val = temp[I]
                count = 1
                for d in 1:N
                    sum_val += temp[I - δ(d, I)] + temp[I + δ(d, I)]
                    count += 2
                end
                indicator[I] = sum_val / count
            end
        end
    end
    return indicator
end
