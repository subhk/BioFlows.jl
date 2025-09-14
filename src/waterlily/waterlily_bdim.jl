# WaterLily.jl style BDIM implementation for BioFlows.jl
# Based on WaterLily.jl's approach with μ₀, μ₁ moments and proper BDIM correction

using StaticArrays

# Body geometry structures
abstract type AbstractBodyWL end

struct NoBodyWL <: AbstractBodyWL end

struct AutoBodyWL{F1<:Function,F2<:Function} <: AbstractBodyWL
    sdf::F1
    map::F2
    function AutoBodyWL(sdf, map=(x,t)->x; compose=true)
        comp(x,t) = compose ? sdf(map(x,t),t) : sdf(x,t)
        new{typeof(comp),typeof(map)}(comp, map)
    end
end

# BDIM flow state (extends regular flow state)
mutable struct BDIMFlowState
    # Regular flow variables
    u::Matrix{Float64}      # x-velocity (staggered)
    w::Matrix{Float64}      # z-velocity (staggered)  
    p::Matrix{Float64}      # pressure
    
    # BDIM fields (WaterLily style)
    V::Array{Float64,3}     # body velocity vector field [nx,nz,2]
    μ₀::Array{Float64,3}    # zeroth-moment vector [nx,nz,2]
    μ₁::Array{Float64,4}    # first-moment tensor [nx,nz,2,2]
    
    # Working arrays
    u⁰::Matrix{Float64}     # previous velocity
    f::Array{Float64,3}     # force vector [nx,nz,2]
    σ::Matrix{Float64}      # divergence scalar
end

# Convolution kernel functions (from WaterLily)
@fastmath kern(d) = 0.5 + 0.5*cos(π*d)
@fastmath kern₀(d) = 0.5 + 0.5*d + 0.5*sin(π*d)/π
@fastmath kern₁(d) = 0.25*(1-d^2) - 0.5*(d*sin(π*d) + (1+cos(π*d))/π)/π

μ₀_kern(d, ϵ) = kern₀(clamp(d/ϵ, -1, 1))
μ₁_kern(d, ϵ) = ϵ * kern₁(clamp(d/ϵ, -1, 1))

# SDF for circle
function circle_sdf(x, center, radius)
    return sqrt((x[1] - center[1])^2 + (x[2] - center[2])^2) - radius
end

# Measure function (WaterLily style)
function measure_body(body::AutoBodyWL, x, t=0.0)
    # Get signed distance
    d = body.sdf(x, t)
    
    # Skip expensive calculations if far from boundary
    d^2 > 9 && return (d, SVector(0.0, 0.0), SVector(0.0, 0.0))
    
    # Compute gradient for normal vector using finite differences
    h = 1e-6
    dx_p = body.sdf(SVector(x[1] + h, x[2]), t)
    dx_m = body.sdf(SVector(x[1] - h, x[2]), t)
    dz_p = body.sdf(SVector(x[1], x[2] + h), t)
    dz_m = body.sdf(SVector(x[1], x[2] - h), t)
    
    n = SVector((dx_p - dx_m)/(2h), (dz_p - dz_m)/(2h))
    
    # Normalize and correct distance
    m = sqrt(n[1]^2 + n[2]^2)
    if m > 1e-12
        d /= m
        n = n / m
    else
        n = SVector(0.0, 0.0)
    end
    
    # Body velocity (zero for static body)
    V = SVector(0.0, 0.0)
    
    return (d, n, V)
end

measure_body(body::NoBodyWL, x, t=0.0) = (Inf, SVector(0.0, 0.0), SVector(0.0, 0.0))

# Create BDIM flow state
function create_bdim_flow_state(nx, nz)
    u = zeros(nx+1, nz)    # u-velocity (staggered in x)
    w = zeros(nx, nz+1)    # w-velocity (staggered in z)
    p = zeros(nx, nz)      # pressure at cell centers
    
    # BDIM fields
    V = zeros(nx+2, nz+2, 2)      # body velocity
    μ₀ = ones(nx+2, nz+2, 2)      # zeroth moment (starts as 1)
    μ₁ = zeros(nx+2, nz+2, 2, 2)  # first moment tensor
    
    # Working arrays
    u⁰ = copy(u)
    f = zeros(nx+2, nz+2, 2)
    σ = zeros(nx+2, nz+2)
    
    return BDIMFlowState(u, w, p, V, μ₀, μ₁, u⁰, f, σ)
end

# Measure function - fill BDIM arrays (core WaterLily functionality)
function measure!(state::BDIMFlowState, body::AbstractBodyWL, grid, t=0.0, ϵ=1.0)
    nx, nz = size(state.p)
    dx, dz = grid.dx, grid.dz
    
    # Reset BDIM fields
    fill!(state.V, 0.0)
    fill!(state.μ₀, 1.0)
    fill!(state.μ₁, 0.0)
    
    d²_thresh = (2 + ϵ)^2
    
    # Loop over all grid points (including ghost cells for BDIM)
    for j = 1:nz+2, i = 1:nx+2
        # Cell center location
        x_center = (i - 1.5) * dx
        z_center = (j - 1.5) * dz
        x_vec = SVector(x_center, z_center)
        
        # Get distance to body at cell center
        d_center = body.sdf(x_vec, t)
        
        # Skip if too far from boundary
        if d_center^2 >= d²_thresh
            if d_center < 0  # Inside solid
                state.μ₀[i, j, 1] = 0.0
                state.μ₀[i, j, 2] = 0.0
            end
            continue
        end
        
        # Process u-velocity point (staggered in x)
        x_u = SVector((i - 1.0) * dx, (j - 1.5) * dz)
        d_u, n_u, V_u = measure_body(body, x_u, t)
        
        if abs(d_u) < 2*ϵ  # Near boundary
            state.V[i, j, 1] = V_u[1]
            state.μ₀[i, j, 1] = μ₀_kern(d_u, ϵ)
            state.μ₁[i, j, 1, 1] = μ₁_kern(d_u, ϵ) * n_u[1]  # ∂μ₁/∂x
            state.μ₁[i, j, 1, 2] = μ₁_kern(d_u, ϵ) * n_u[2]  # ∂μ₁/∂z
        elseif d_u < 0  # Inside solid
            state.μ₀[i, j, 1] = 0.0
        end
        
        # Process w-velocity point (staggered in z)
        x_w = SVector((i - 1.5) * dx, (j - 1.0) * dz)
        d_w, n_w, V_w = measure_body(body, x_w, t)
        
        if abs(d_w) < 2*ϵ  # Near boundary
            state.V[i, j, 2] = V_w[2]
            state.μ₀[i, j, 2] = μ₀_kern(d_w, ϵ)
            state.μ₁[i, j, 2, 1] = μ₁_kern(d_w, ϵ) * n_w[1]  # ∂μ₁/∂x  
            state.μ₁[i, j, 2, 2] = μ₁_kern(d_w, ϵ) * n_w[2]  # ∂μ₁/∂z
        elseif d_w < 0  # Inside solid
            state.μ₀[i, j, 2] = 0.0
        end
    end
end

# BDIM correction step (core WaterLily algorithm)
function BDIM!(state::BDIMFlowState, dt)
    nx, nz = size(state.p)
    
    # Prepare force field: f = u⁰ + dt*f - V
    for j = 1:nz+2, i = 1:nx+2
        # For u-component
        if i <= nx+1 && j <= nz
            u_idx_i = min(i, nx+1)
            u_idx_j = min(j, nz)
            state.f[i, j, 1] = state.u⁰[u_idx_i, u_idx_j] + dt * state.f[i, j, 1] - state.V[i, j, 1]
        end
        
        # For w-component  
        if i <= nx && j <= nz+1
            w_idx_i = min(i, nx)
            w_idx_j = min(j, nz+1)
            state.f[i, j, 2] = state.u⁰[w_idx_i, w_idx_j] + dt * state.f[i, j, 2] - state.V[i, j, 2]
        end
    end
    
    # Apply BDIM correction: u += μddn(μ₁,f) + V + μ₀*f
    for j = 2:nz+1, i = 2:nx+1
        # u-velocity correction
        if i <= nx+1 && j <= nz
            u_idx_i = min(i, nx+1)
            u_idx_j = min(j, nz)
            
            # Compute μddn term: μ₁ · ∇f
            dfdx = 0.0
            dfdz = 0.0
            
            if i > 1 && i < nx+2
                dfdx = 0.5 * (state.f[i+1, j, 1] - state.f[i-1, j, 1])
            end
            if j > 1 && j < nz+2
                dfdz = 0.5 * (state.f[i, j+1, 1] - state.f[i, j-1, 1])
            end
            
            μddn_u = state.μ₁[i, j, 1, 1] * dfdx + state.μ₁[i, j, 1, 2] * dfdz
            
            state.u[u_idx_i, u_idx_j] += μddn_u + state.V[i, j, 1] + state.μ₀[i, j, 1] * state.f[i, j, 1]
        end
        
        # w-velocity correction
        if i <= nx && j <= nz+1
            w_idx_i = min(i, nx)
            w_idx_j = min(j, nz+1)
            
            # Compute μddn term: μ₁ · ∇f
            dfdx = 0.0
            dfdz = 0.0
            
            if i > 1 && i < nx+2
                dfdx = 0.5 * (state.f[i+1, j, 2] - state.f[i-1, j, 2])
            end
            if j > 1 && j < nz+2
                dfdz = 0.5 * (state.f[i, j+1, 2] - state.f[i, j-1, 2])
            end
            
            μddn_w = state.μ₁[i, j, 2, 1] * dfdx + state.μ₁[i, j, 2, 2] * dfdz
            
            state.w[w_idx_i, w_idx_j] += μddn_w + state.V[i, j, 2] + state.μ₀[i, j, 2] * state.f[i, j, 2]
        end
    end
end

# Create cylinder body
function create_cylinder_body(center, radius)
    sdf_func(x, t) = circle_sdf(x, center, radius)
    return AutoBodyWL(sdf_func)
end

# Export main functions
export BDIMFlowState, AbstractBodyWL, NoBodyWL, AutoBodyWL
export create_bdim_flow_state, measure!, BDIM!, create_cylinder_body
export measure_body, circle_sdf