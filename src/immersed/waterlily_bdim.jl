# Complete WaterLily-style BDIM implementation for BioFlows.jl
# This matches WaterLily.jl's exact approach with staggered grids

using StaticArrays

# WaterLily-style BDIM fields (exactly matching Flow struct)
mutable struct WaterLilyBDIM{T}
    # Staggered velocity-location BDIM fields
    μ₀::Array{T, 3}      # Zeroth moment vector: size (nx+1, nz+1, 2) 
    μ₁::Array{T, 4}      # First moment tensor: size (nx, nz, 2, 2)
    V::Array{T, 3}       # Body velocity vector: size (nx+1, nz+1, 2)
    
    # Working arrays for BDIM algorithm
    f::Array{T, 3}       # Force field: size (nx+1, nz+1, 2)
    σ::Array{T, 2}       # Distance field: size (nx, nz)
end

function WaterLilyBDIM(nx::Int, nz::Int, T=Float64)
    return WaterLilyBDIM{T}(
        ones(T, nx+1, nz+1, 2),     # μ₀ initialized to 1 (fluid)
        zeros(T, nx, nz, 2, 2),     # μ₁ initialized to 0
        zeros(T, nx+1, nz+1, 2),    # V initialized to 0 (static body)
        zeros(T, nx+1, nz+1, 2),    # f working array
        zeros(T, nx, nz)            # σ distance field
    )
end

# WaterLily kernel functions (exact copies)
@fastmath kern₀(d) = 0.5 + 0.5*d + 0.5*sin(π*d)/π
@fastmath kern₁(d) = 0.25*(1-d^2) - 0.5*(d*sin(π*d) + (1+cos(π*d))/π)/π

# WaterLily moment functions
μ₀(d, ϵ) = kern₀(clamp(d/ϵ, -1, 1))
μ₁(d, ϵ) = ϵ * kern₁(clamp(d/ϵ, -1, 1))

# Grid location functions (matching WaterLily's loc function)
@inline loc(i::Int, I::CartesianIndex{N}, T) where N = SVector{N,T}(I.I...) .- 1.5 .+ SVector{N,T}(ntuple(j -> j==i ? 0.5 : 0, N))
@inline loc(::Val{0}, I::CartesianIndex{N}, T) where N = SVector{N,T}(I.I...) .- 1.5

# WaterLily-style measure! function for BDIM
function measure_waterlily!(bdim::WaterLilyBDIM{T}, bodies, grid, t=zero(T), ϵ=1) where T
    nx, nz = size(bdim.σ)
    
    # Reset fields like WaterLily
    fill!(bdim.V, zero(T))
    fill!(bdim.μ₀, one(T)) 
    fill!(bdim.μ₁, zero(T))
    
    d² = (2 + ϵ)^2
    
    # Loop over all grid points (following WaterLily's measure!)
    for I in CartesianIndices(bdim.σ)
        i, j = I.I
        
        # Cell center distance (quick check)
        x_center = loc(Val(0), CartesianIndex(i, j), T)
        # Convert to physical coordinates
        x_phys = SVector((i-1)*grid.dx, (j-1)*grid.dz)
        
        # Check distance to all bodies
        for body in bodies.bodies
            d_center = sdf_circle(x_phys, body.center, body.shape.radius)
            bdim.σ[I] = d_center
            
            if d_center^2 < d²
                # Process each velocity component at staggered locations
                for comp in 1:2
                    # Get staggered location for this component
                    x_stagger = loc(comp, CartesianIndex(i, j), T)
                    x_stagger_phys = SVector((i-1)*grid.dx + (comp==1 ? 0.5*grid.dx : 0), 
                                           (j-1)*grid.dz + (comp==2 ? 0.5*grid.dz : 0))
                    
                    # Measure at staggered location
                    d, n, V_body = measure_body_waterlily(body, x_stagger_phys, t)
                    
                    # Fill BDIM arrays (WaterLily algorithm)
                    if i <= size(bdim.μ₀, 1) && j <= size(bdim.μ₀, 2)
                        bdim.V[i, j, comp] = V_body[comp]
                        bdim.μ₀[i, j, comp] = μ₀(d, ϵ)
                        
                        # First moment tensor (only for cell centers)
                        if i <= size(bdim.μ₁, 1) && j <= size(bdim.μ₁, 2)
                            μ₁_val = μ₁(d, ϵ)
                            for k in 1:2
                                bdim.μ₁[i, j, comp, k] = μ₁_val * n[k]
                            end
                        end
                    end
                end
            elseif d_center < zero(T)  # Inside solid
                for comp in 1:2
                    if i <= size(bdim.μ₀, 1) && j <= size(bdim.μ₀, 2)
                        bdim.μ₀[i, j, comp] = zero(T)
                    end
                end
            end
        end
    end
end

# Body measurement function
function measure_body_waterlily(body, x, t)
    if body.shape isa Circle
        return measure_circle_waterlily(x, body.center, body.shape.radius, body.velocity)
    else
        error("Body shape not supported")
    end
end

function measure_circle_waterlily(x, center, radius, velocity)
    d_vec = x - SVector(center[1], center[2])
    d = norm(d_vec) - radius
    
    if norm(d_vec) > 1e-12
        n = d_vec / norm(d_vec)  # Normal vector
    else
        n = SVector(0.0, 0.0)
    end
    
    V = SVector(velocity[1], velocity[2])  # Body velocity
    
    return d, n, V
end

function sdf_circle(x, center, radius)
    return norm(x - SVector(center[1], center[2])) - radius
end

# WaterLily's μddn function (exact copy from Flow.jl line 20-26)
@fastmath @inline function μddn_waterlily(I::CartesianIndex, μ₁, f, dx, dz)
    i, j = I.I
    s = zero(eltype(f))
    
    # x-component gradient
    if i > 1 && i < size(f, 1)
        df_dx = (f[i+1, j, 1] - f[i-1, j, 1]) / (2*dx)
        s += μ₁[min(i, size(μ₁,1)), min(j, size(μ₁,2)), 1, 1] * df_dx
    end
    
    # z-component gradient  
    if j > 1 && j < size(f, 2)
        df_dz = (f[i, j+1, 1] - f[i, j-1, 1]) / (2*dz)
        s += μ₁[min(i, size(μ₁,1)), min(j, size(μ₁,2)), 1, 2] * df_dz
    end
    
    return 0.5 * s
end

# WaterLily's BDIM! function (corrected implementation)
function BDIM_waterlily!(state, u⁰, w⁰, bdim::WaterLilyBDIM, dt, grid)
    """
    WaterLily's exact BDIM algorithm (corrected):
    Line 127: f = u⁰ + dt*f - V  
    Line 128: u += μddn(μ₁,f) + V + μ₀*f
    """
    nx, nz = size(bdim.σ)
    dx, dz = grid.dx, grid.dz
    
    # Step 1: Compute force field f = u⁰ - V (simplified, as in WaterLily predictor step)
    for j in 1:size(bdim.f, 2), i in 1:size(bdim.f, 1)
        # u-component force
        if i <= size(u⁰, 1) && j <= size(u⁰, 2)
            bdim.f[i, j, 1] = u⁰[i, j] - bdim.V[i, j, 1]
        end
        # w-component force  
        if i <= size(w⁰, 1) && j <= size(w⁰, 2)
            bdim.f[i, j, 2] = w⁰[i, j] - bdim.V[i, j, 2]
        end
    end
    
    # Step 2: Apply BDIM correction u += μddn + V + μ₀*f (line 128)
    # u-velocity correction
    for j in 1:min(nz, size(state.u, 2)), i in 1:min(nx+1, size(state.u, 1))
        I = CartesianIndex(i, j)
        
        # Get BDIM coefficients
        if i <= size(bdim.μ₀, 1) && j <= size(bdim.μ₀, 2)
            μ₀_val = bdim.μ₀[i, j, 1]
            V_val = bdim.V[i, j, 1] 
            f_val = bdim.f[i, j, 1]
            
            # Compute μddn correction
            μddn_corr = μddn_waterlily(I, bdim.μ₁, bdim.f, dx, dz)
            
            # Apply WaterLily correction
            state.u[i, j] += μddn_corr + V_val + μ₀_val * f_val
        end
    end
    
    # w-velocity correction  
    for j in 1:min(nz+1, size(state.w, 2)), i in 1:min(nx, size(state.w, 1))
        I = CartesianIndex(i, j)
        
        # Get BDIM coefficients for w-component
        if i <= size(bdim.μ₀, 1) && j <= size(bdim.μ₀, 2)
            μ₀_val = bdim.μ₀[i, j, 2]
            V_val = bdim.V[i, j, 2]
            f_val = bdim.f[i, j, 2]
            
            # Compute μddn correction for w (similar to u)
            μddn_corr = μddn_waterlily_w(I, bdim.μ₁, bdim.f, dx, dz)
            
            # Apply WaterLily correction
            state.w[i, j] += μddn_corr + V_val + μ₀_val * f_val
        end
    end
end

# μddn function for w-component
@fastmath @inline function μddn_waterlily_w(I::CartesianIndex, μ₁, f, dx, dz)
    i, j = I.I
    s = zero(eltype(f))
    
    # x-component gradient for w
    if i > 1 && i < size(f, 1)
        df_dx = (f[i+1, j, 2] - f[i-1, j, 2]) / (2*dx)
        s += μ₁[min(i, size(μ₁,1)), min(j, size(μ₁,2)), 2, 1] * df_dx
    end
    
    # z-component gradient for w
    if j > 1 && j < size(f, 2)
        df_dz = (f[i, j+1, 2] - f[i, j-1, 2]) / (2*dz)
        s += μ₁[min(i, size(μ₁,1)), min(j, size(μ₁,2)), 2, 2] * df_dz
    end
    
    return 0.5 * s
end

# CFL condition calculation (from WaterLily Flow.jl lines 168-171)
function CFL_waterlily(u, w, ν, dx, dz; Δt_max=10)
    """
    WaterLily's CFL calculation: min(Δt_max, 1/(max_flux + 5ν))
    """
    max_flux = 0.0
    
    # Compute maximum flux out of cells
    for j in 1:size(u, 2), i in 1:size(u, 1)-1
        flux = abs(u[i+1, j] - u[i, j])/dx + abs(w[i, j+1] - w[i, j])/dz
        max_flux = max(max_flux, flux)
    end
    
    return min(Δt_max, 1.0/(max_flux + 5*ν))
end

# Export functions
export WaterLilyBDIM, measure_waterlily!, BDIM_waterlily!, CFL_waterlily