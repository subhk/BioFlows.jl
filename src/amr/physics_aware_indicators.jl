"""
Physics-aware AMR indicators inspired by Trixi.jl
Implements Löhner curvature indicator and Hennemann-Gassner shock indicator
for more selective and conservative mesh refinement.
"""

"""
    compute_lohner_indicator(field, grid)

Compute Löhner curvature indicator based on weighted second derivatives.
This is more physics-aware than simple gradient thresholds.
"""
function compute_lohner_indicator(field::Matrix{T}, grid) where T
    nx, nz = size(field)
    dx, dz = grid.dx, grid.dz
    indicator = zeros(T, nx, nz)
    
    for j = 2:nz-1, i = 2:nx-1
        # Second derivatives in x and z directions
        d2u_dx2 = (field[i+1, j] - 2*field[i, j] + field[i-1, j]) / dx^2
        d2u_dz2 = (field[i, j+1] - 2*field[i, j] + field[i, j-1]) / dz^2
        
        # First derivatives for normalization
        du_dx = (field[i+1, j] - field[i-1, j]) / (2*dx)
        du_dz = (field[i, j+1] - field[i, j-1]) / (2*dz)
        
        # Löhner indicator: normalized curvature
        numerator = abs(d2u_dx2) + abs(d2u_dz2)
        denominator = abs(du_dx)/dx + abs(du_dz)/dz + 
                     abs(field[i, j])/(min(dx, dz)) + 1e-12
        
        indicator[i, j] = numerator / denominator
    end
    
    return indicator
end

"""
    compute_hennemann_gassner_indicator(u, w, p, grid)

Compute Hennemann-Gassner shock indicator for detecting discontinuities.
This helps identify regions needing refinement for flow features.
"""
function compute_hennemann_gassner_indicator(u::Matrix{T}, w::Matrix{T}, p::Matrix{T}, grid) where T
    nx, nz = size(p)
    indicator = zeros(T, nx, nz)
    
    for j = 2:nz-1, i = 2:nx-1
        # Velocity magnitude squared
        u_center = (u[i, j] + u[i+1, j]) / 2  # Average to cell center
        w_center = (w[i, j] + w[i, j+1]) / 2  # Average to cell center
        vel_mag_sq = u_center^2 + w_center^2
        
        # Pressure-based shock indicator
        # High gradients in pressure indicate potential shocks/discontinuities
        dp_dx = (p[i+1, j] - p[i-1, j]) / (2*grid.dx)
        dp_dz = (p[i, j+1] - p[i, j-1]) / (2*grid.dz)
        pressure_gradient_mag = sqrt(dp_dx^2 + dp_dz^2)
        
        # Normalized indicator
        # Uses pressure gradient magnitude normalized by dynamic pressure
        dynamic_pressure = 0.5 * 1000.0 * vel_mag_sq + 1e-6  # ρ = 1000 kg/m³
        indicator[i, j] = pressure_gradient_mag / (abs(p[i, j]) + dynamic_pressure + 1e-6)
    end
    
    return indicator
end

"""
    compute_vorticity_indicator(u, w, grid)

Compute vorticity magnitude for detecting rotational flow structures.
Refined around vortices to capture wake dynamics accurately.
"""
function compute_vorticity_indicator(u::Matrix{T}, w::Matrix{T}, grid) where T
    nx_u, nz_u = size(u)
    nx_w, nz_w = size(w)
    nx = min(nx_u-1, nx_w)
    nz = min(nz_u, nz_w-1)
    
    vorticity = zeros(T, nx, nz)
    
    for j = 1:nz, i = 1:nx
        # Compute vorticity: ω = ∂w/∂x - ∂u/∂z
        if i < nx && j < nz
            # Finite differences on staggered grid
            dw_dx = (w[i+1, j] - w[i, j]) / grid.dx  # w is at cell centers
            du_dz = (u[i, j+1] - u[i, j]) / grid.dz  # u is at cell centers
            
            vorticity[i, j] = abs(dw_dx - du_dz)
        end
    end
    
    return vorticity
end

"""
    trixi_style_amr_indicator(state, grid, config)

Physics-aware AMR indicator combining multiple criteria like Trixi.jl.
Returns a refined indicator field that's much more selective than simple thresholds.
"""
function trixi_style_amr_indicator(state, grid, config)
    # Compute individual indicators
    lohner_u = compute_lohner_indicator(state.u[1:end-1, :], grid)  # Remove staggered dimension
    lohner_p = compute_lohner_indicator(state.p, grid)
    
    hg_indicator = compute_hennemann_gassner_indicator(state.u, state.w, state.p, grid)
    
    vort_indicator = compute_vorticity_indicator(state.u, state.w, grid)
    
    # Combined indicator using weighted maximum
    nx, nz = size(state.p)
    combined_indicator = zeros(nx, nz)
    
    # Weight factors (inspired by Trixi's approach)
    weight_curvature = 0.4
    weight_shock = 0.4  
    weight_vorticity = 0.2
    
    for j = 1:nz, i = 1:nx
        if i <= size(lohner_u, 1) && j <= size(lohner_u, 2)
            curvature_score = max(lohner_u[i, j], lohner_p[i, j])
        else
            curvature_score = 0.0
        end
        
        shock_score = i <= size(hg_indicator, 1) && j <= size(hg_indicator, 2) ? hg_indicator[i, j] : 0.0
        
        vort_score = i <= size(vort_indicator, 1) && j <= size(vort_indicator, 2) ? vort_indicator[i, j] : 0.0
        
        # Weighted combination
        combined_indicator[i, j] = weight_curvature * curvature_score + 
                                  weight_shock * shock_score + 
                                  weight_vorticity * vort_score
    end
    
    return combined_indicator
end