module MaskedIB

using ..BioFlows
# Bring key types/functions into scope to avoid qualification errors at precompile
using ..BioFlows: SolutionState, RigidBodyCollection,
                  NavierStokesSolver3D, gradient_pressure_3d!, solve_step_3d!,
                  build_solid_mask_faces_3d, masked_divergence_3d!

export masked_ib_step!

"""
    masked_ib_step!(solver, state_new, state_old, dt, bodies)

Perform one step using the existing projection path and apply the
masked immersed-boundary (BDIM) forcing. This is a thin adapter that
selects the BDIM method via the public IB API without exposing any
WaterLily naming.
"""
function masked_ib_step!(solver, state_new::SolutionState, state_old::SolutionState,
                         dt::Float64, bodies::RigidBodyCollection)
    # Standard projection step
    solve_step_2d!(solver, state_new, state_old, dt, bodies)
    # Apply BDIM forcing
    apply_immersed_boundary_forcing!(state_new, bodies, solver.grid, dt; method=BDIM)
    return nothing
end

"""
    masked_ib_step!(solver::NavierStokesSolver3D, state_new, state_old, dt, bodies)

3D masked IB step: run standard 3D projection step, then perform a masked
pressure projection/correction using smooth 3D face masks to mimic WaterLily's
masked projection behavior in 3D.
"""
function masked_ib_step!(solver::NavierStokesSolver3D, state_new::SolutionState,
                         state_old::SolutionState, dt::Float64, bodies::RigidBodyCollection)
    grid = solver.grid
    bc = solver.bc
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    # First, take a standard 3D step to compute tentative velocities
    solve_step_3d!(solver, state_new, state_old, dt)
    
    # Build smooth face masks (χ_u, χ_v, χ_w)
    chi_u, chi_v, chi_w = build_solid_mask_faces_3d(bodies, grid; eps_mul=2.0)
    
    # Recompute masked divergence RHS at cell centers
    rhs = zeros(Float64, nx, ny, nz)
    masked_divergence_3d!(rhs, state_new.u, state_new.v, state_new.w, grid, chi_u, chi_v, chi_w)
    rhs .*= (1.0/dt)
    
    # Solve Poisson for pressure correction
    phi = zeros(Float64, nx, ny, nz)
    solve_poisson!(solver.multigrid_solver, phi, rhs, grid, bc)
    
    # Correct velocities only in fluid (1-χ)
    dpdx = zeros(Float64, nx+1, ny, nz)
    dpdy = zeros(Float64, nx, ny+1, nz)
    dpdz = zeros(Float64, nx, ny, nz+1)
    gradient_pressure_3d!(dpdx, dpdy, dpdz, phi, grid)
    state_new.u .-= dt .* (1 .- chi_u) .* dpdx
    state_new.v .-= dt .* (1 .- chi_v) .* dpdy
    state_new.w .-= dt .* (1 .- chi_w) .* dpdz
    
    # Update pressure (add correction)
    ρ = solver.fluid.ρ isa ConstantDensity ? solver.fluid.ρ.ρ : 1.0
    state_new.p .+= ρ .* phi
    
    # Final BCs
    apply_boundary_conditions!(grid, state_new, bc, state_old.t + dt)
    return nothing
end

end # module
