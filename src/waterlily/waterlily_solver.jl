# Complete WaterLily.jl style solver integration for BioFlows.jl
# Implements RK2 time stepping with full BDIM immersed boundary method

include("waterlily_bdim.jl")

# WaterLily-style divergence operator
function div_waterlily!(div_field, state::BDIMFlowState, grid)
    nx, nz = size(state.p)
    dx_inv, dz_inv = 1.0/grid.dx, 1.0/grid.dz
    
    fill!(div_field, 0.0)
    
    # Interior divergence calculation (∇·u)
    for j = 1:nz, i = 1:nx
        du_dx = (state.u[i+1, j] - state.u[i, j]) * dx_inv
        dw_dz = (state.w[i, j+1] - state.w[i, j]) * dz_inv
        div_field[i, j] = du_dx + dw_dz
    end
end

# WaterLily-style pressure gradient operator
function grad_pressure_waterlily!(state::BDIMFlowState, pressure, grid, L_coeffs)
    nx, nz = size(state.p)
    dx_inv, dz_inv = 1.0/grid.dx, 1.0/grid.dz
    
    # Apply pressure gradient to u-velocity
    for j = 1:nz, i = 1:nx+1
        if i > 1 && i <= nx
            dp_dx = (pressure[i, j] - pressure[i-1, j]) * dx_inv
            state.u[i, j] -= L_coeffs[i, j, 1] * dp_dx
        end
    end
    
    # Apply pressure gradient to w-velocity  
    for j = 1:nz+1, i = 1:nx
        if j > 1 && j <= nz
            dp_dz = (pressure[i, j] - pressure[i, j-1]) * dz_inv
            state.w[i, j] -= L_coeffs[i, j, 2] * dp_dz
        end
    end
end

# Simple convective term (central difference scheme)
function conv_diff_waterlily!(state::BDIMFlowState, grid, ν)
    nx, nz = size(state.p)
    dx, dz = grid.dx, grid.dz
    dx_inv, dz_inv = 1.0/dx, 1.0/dz
    dx2_inv, dz2_inv = dx_inv*dx_inv, dz_inv*dz_inv
    
    # Reset force field
    fill!(state.f, 0.0)
    
    # u-momentum equation
    for j = 2:nz-1, i = 2:nx
        u_ij = state.u[i, j]
        
        # Convective terms (simplified central difference)
        u_center = 0.25 * (state.u[i, j] + state.u[i+1, j] + state.u[i, j-1] + state.u[i+1, j-1])
        w_center = 0.25 * (state.w[i-1, j] + state.w[i, j] + state.w[i-1, j+1] + state.w[i, j+1])
        
        conv_x = u_center * (state.u[i+1, j] - state.u[i-1, j]) * 0.5 * dx_inv
        conv_z = w_center * (state.u[i, j+1] - state.u[i, j-1]) * 0.5 * dz_inv
        
        # Viscous terms
        visc_x = ν * (state.u[i+1, j] - 2*state.u[i, j] + state.u[i-1, j]) * dx2_inv
        visc_z = ν * (state.u[i, j+1] - 2*state.u[i, j] + state.u[i, j-1]) * dz2_inv
        
        state.f[i, j, 1] = -conv_x - conv_z + visc_x + visc_z
    end
    
    # w-momentum equation
    for j = 2:nz, i = 2:nx-1
        w_ij = state.w[i, j]
        
        # Convective terms
        u_center = 0.25 * (state.u[i, j-1] + state.u[i+1, j-1] + state.u[i, j] + state.u[i+1, j])
        w_center = 0.25 * (state.w[i, j] + state.w[i, j+1] + state.w[i-1, j] + state.w[i-1, j+1])
        
        conv_x = u_center * (state.w[i+1, j] - state.w[i-1, j]) * 0.5 * dx_inv
        conv_z = w_center * (state.w[i, j+1] - state.w[i, j-1]) * 0.5 * dz_inv
        
        # Viscous terms
        visc_x = ν * (state.w[i+1, j] - 2*state.w[i, j] + state.w[i-1, j]) * dx2_inv
        visc_z = ν * (state.w[i, j+1] - 2*state.w[i, j] + state.w[i, j-1]) * dz2_inv
        
        state.f[i, j, 2] = -conv_x - conv_z + visc_x + visc_z
    end
end

# Apply boundary conditions (WaterLily style)
function apply_bc_waterlily!(state::BDIMFlowState, U_inlet, grid, t=0.0)
    nx, nz = size(state.p)
    
    # Inlet boundary (left, i=1)
    for j = 1:nz
        state.u[1, j] = U_inlet  # Dirichlet for u
    end
    
    # Outlet boundary (right, i=nx+1) - zero gradient
    for j = 1:nz
        state.u[nx+1, j] = state.u[nx, j]  # Neumann for u
    end
    
    # Wall boundaries (top and bottom) - no-slip
    for i = 1:nx+1
        # Bottom wall (j=1)
        # u already satisfies no-slip at wall faces
        
        # Top wall (j=nz)  
        # u already satisfies no-slip at wall faces
    end
    
    # w-velocity boundaries
    for i = 1:nx
        state.w[i, 1] = 0.0     # Bottom wall no-slip
        state.w[i, nz+1] = 0.0  # Top wall no-slip
    end
    
    # w at inlet/outlet (zero gradient)
    for j = 2:nz
        # Left boundary
        state.w[1, j] = state.w[2, j]
        # Right boundary  
        state.w[nx, j] = state.w[nx-1, j]
    end
end

# WaterLily-style momentum step with BDIM
function waterlily_momentum_step!(state::BDIMFlowState, body::AbstractBodyWL, grid, dt, ν, U_inlet, pressure_solver)
    # Store previous velocity
    copy!(state.u⁰, state.u)
    u⁰_w = copy(state.w)  # Also store w⁰
    
    # Compute convective and viscous terms
    conv_diff_waterlily!(state, grid, ν)
    
    # Measure body geometry and update BDIM fields
    measure!(state, body, grid, 0.0, 1.0)
    
    # Apply BDIM correction
    BDIM!(state, dt)
    
    # Apply boundary conditions
    apply_bc_waterlily!(state, U_inlet, grid)
    
    # Project velocity to be divergence free
    waterlily_project!(state, grid, pressure_solver)
    
    # Apply BC again after projection
    apply_bc_waterlily!(state, U_inlet, grid)
    
    # Corrector step (RK2)
    # Store intermediate velocity
    u_star = copy(state.u)
    w_star = copy(state.w)
    
    # Compute RHS for corrector
    conv_diff_waterlily!(state, grid, ν)
    
    # Update BDIM fields at new time
    measure!(state, body, grid, 0.0, 1.0)
    BDIM!(state, dt)
    
    # Average predictor and corrector (RK2)
    for j = 1:size(state.u, 2), i = 1:size(state.u, 1)
        state.u[i, j] = 0.5 * (state.u⁰[i, j] + state.u[i, j])
    end
    for j = 1:size(state.w, 2), i = 1:size(state.w, 1)
        state.w[i, j] = 0.5 * (u⁰_w[i, j] + state.w[i, j])
    end
    
    # Apply BC and final projection
    apply_bc_waterlily!(state, U_inlet, grid)
    waterlily_project!(state, grid, pressure_solver)
    apply_bc_waterlily!(state, U_inlet, grid)
end

# WaterLily-style projection step
function waterlily_project!(state::BDIMFlowState, grid, pressure_solver)
    nx, nz = size(state.p)
    
    # Compute divergence
    div_waterlily!(state.σ, state, grid)
    
    # Set up Poisson system: ∇²φ = ∇·u
    # Use simplified coefficient matrix (uniform grid)
    dx2_inv, dz2_inv = 1.0/(grid.dx^2), 1.0/(grid.dz^2)
    
    # Simple Jacobi iteration for pressure correction
    φ = zeros(nx, nz)  # pressure correction
    rhs = copy(state.σ)  # divergence as RHS
    
    # Jacobi iterations
    for iter = 1:100
        φ_old = copy(φ)
        
        for j = 2:nz-1, i = 2:nx-1
            # Interior points
            φ[i, j] = (rhs[i, j] + dx2_inv*(φ_old[i+1, j] + φ_old[i-1, j]) + 
                       dz2_inv*(φ_old[i, j+1] + φ_old[i, j-1])) / 
                      (2*dx2_inv + 2*dz2_inv)
        end
        
        # Boundary conditions for φ (zero Neumann)
        for i = 1:nx
            φ[i, 1] = φ[i, 2]      # Bottom
            φ[i, nz] = φ[i, nz-1]  # Top
        end
        for j = 1:nz
            φ[1, j] = φ[2, j]      # Left
            φ[nx, j] = φ[nx-1, j]  # Right
        end
        
        # Check convergence
        if maximum(abs.(φ - φ_old)) < 1e-8
            break
        end
    end
    
    # Apply pressure correction to velocities
    dx_inv, dz_inv = 1.0/grid.dx, 1.0/grid.dz
    
    # Update u-velocity
    for j = 1:nz, i = 2:nx
        dp_dx = (φ[i, j] - φ[i-1, j]) * dx_inv
        state.u[i, j] -= dp_dx
    end
    
    # Update w-velocity
    for j = 2:nz, i = 1:nx
        dp_dz = (φ[i, j] - φ[i, j-1]) * dz_inv
        state.w[i, j] -= dp_dz
    end
    
    # Update pressure
    state.p .+= φ
end

# Complete WaterLily solver step
function waterlily_step!(state::BDIMFlowState, body::AbstractBodyWL, grid, dt, ν, U_inlet)
    pressure_solver = nothing  # We'll use simple Jacobi iteration
    waterlily_momentum_step!(state, body, grid, dt, ν, U_inlet, pressure_solver)
end

export div_waterlily!, grad_pressure_waterlily!, conv_diff_waterlily!
export apply_bc_waterlily!, waterlily_momentum_step!, waterlily_project!
export waterlily_step!