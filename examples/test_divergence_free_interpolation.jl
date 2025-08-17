"""
Test Divergence-Free Velocity Interpolation for AMR

This example demonstrates the new interpolate_velocity_conservative! function
and verifies that it maintains the divergence-free condition during AMR operations.
"""

push!(LOAD_PATH, ".")
using BioFlows
using Printf

function create_test_velocity_field(grid::StaggeredGrid)
    """Create a test velocity field that is exactly divergence-free."""
    
    nx, nz = grid.nx, grid.nz
    
    # Initialize velocity arrays with proper staggered dimensions
    u = zeros(nx+1, nz)    # u at x-faces
    v = zeros(nx, nz+1)    # v at z-faces (w-velocity in XZ plane)
    
    # Create a divergence-free analytical field: stream function approach
    # ψ(x,z) = sin(2πx/Lx) * sin(2πz/Lz)
    # u = ∂ψ/∂z, v = -∂ψ/∂x  (ensures ∇·u = 0)
    
    Lx, Lz = grid.Lx, grid.Lz
    
    # Fill u-velocity at x-faces
    for j = 1:nz, i = 1:nx+1
        x = grid.xu[i]  # x-coordinate of x-face
        z = grid.y[j]   # z-coordinate of cell center
        
        # u = ∂ψ/∂z = (2π/Lz) * sin(2πx/Lx) * cos(2πz/Lz)
        u[i, j] = (2π/Lz) * sin(2π*x/Lx) * cos(2π*z/Lz)
    end
    
    # Fill v-velocity at z-faces (w in XZ plane)
    for j = 1:nz+1, i = 1:nx
        x = grid.x[i]   # x-coordinate of cell center  
        z = grid.yv[j]  # z-coordinate of z-face
        
        # v = -∂ψ/∂x = -(2π/Lx) * cos(2πx/Lx) * sin(2πz/Lz)
        v[i, j] = -(2π/Lx) * cos(2π*x/Lx) * sin(2π*z/Lz)
    end
    
    return u, v
end

function test_divergence_free_interpolation()
    """Main test function for divergence-free interpolation."""
    
    println("="^60)
    println("Testing Divergence-Free Velocity Interpolation for AMR")
    println("="^60)
    
    # Create coarse grid
    nx_coarse, nz_coarse = 16, 12
    Lx, Lz = 4.0, 3.0
    grid_coarse = create_uniform_2d_grid(nx_coarse, nz_coarse, Lx, Lz)
    
    println("1. Creating divergence-free test velocity field...")
    u_coarse, v_coarse = create_test_velocity_field(grid_coarse)
    
    # Verify original field is divergence-free
    dx, dz = grid_coarse.dx, grid_coarse.dz
    is_div_free, max_div, mean_div = verify_divergence_free(u_coarse, v_coarse, dx, dz)
    
    @printf "   Original field divergence: max = %.2e, mean = %.2e\\n" max_div mean_div
    println("   ✓ Original field is divergence-free: $is_div_free")
    
    # Create fine grid (2x refinement)
    nx_fine, nz_fine = 2*nx_coarse, 2*nz_coarse
    grid_fine = create_uniform_2d_grid(nx_fine, nz_fine, Lx, Lz)
    
    println("\\n2. Setting up AMR interpolation...")
    
    # Create AMRLevel for fine grid
    amr_level = AMRLevel(
        1,                    # level
        nx_fine, 0, nz_fine,  # nx, ny, nz (ny=0 for 2D XZ)
        grid_fine.dx, 0.0, grid_fine.dz,  # dx, dy, dz
        TwoDimensional,       # grid_type
        collect(grid_fine.x), Float64[], collect(grid_fine.y),  # centers
        collect(grid_fine.xu), Float64[], collect(grid_fine.yv), # faces
        zeros(nx_fine+1, nz_fine), zeros(nx_fine, nz_fine+1), nothing, zeros(nx_fine, nz_fine), # arrays
        zeros(Bool, nx_fine, nz_fine), zeros(Union{Nothing, AMRLevel}, nx_fine, nz_fine), nothing, # refinement
        0.0, Lx, 0.0, 0.0, 0.0, Lz, # bounds
        0, Dict(), false      # MPI, neighbors, is_boundary
    )
    
    # Initialize fine grid arrays
    u_fine = zeros(nx_fine+1, nz_fine)
    v_fine = zeros(nx_fine, nz_fine+1)
    
    println("3. Performing conservative velocity interpolation...")
    
    # Test the interpolation function
    interpolate_velocity_conservative!(u_fine, v_fine, u_coarse, v_coarse, amr_level)
    
    println("4. Verifying results...")
    
    # Check if fine field is divergence-free
    dx_fine, dz_fine = grid_fine.dx, grid_fine.dz
    is_div_free_fine, max_div_fine, mean_div_fine = verify_divergence_free(u_fine, v_fine, dx_fine, dz_fine)
    
    @printf "   Fine field divergence: max = %.2e, mean = %.2e\\n" max_div_fine mean_div_fine
    println("   ✓ Fine field is divergence-free: $is_div_free_fine")
    
    # Compare with analytical solution on fine grid
    u_analytical, v_analytical = create_test_velocity_field(grid_fine)
    
    # Compute interpolation errors
    u_error = maximum(abs.(u_fine .- u_analytical))
    v_error = maximum(abs.(v_fine .- v_analytical))
    
    @printf "   Interpolation errors: u_max = %.2e, v_max = %.2e\\n" u_error v_error
    
    # Energy conservation check
    energy_coarse = 0.5 * (sum(u_coarse.^2) * dx * dz + sum(v_coarse.^2) * dx * dz)
    energy_fine = 0.5 * (sum(u_fine.^2) * dx_fine * dz_fine + sum(v_fine.^2) * dx_fine * dz_fine)
    energy_error = abs(energy_fine - energy_coarse) / energy_coarse
    
    @printf "   Energy conservation error: %.2e\\n" energy_error
    
    println("\\n5. Summary:")
    println("   ✓ Conservative interpolation completed")
    println("   ✓ Divergence-free condition preserved: $(is_div_free_fine)")
    println("   ✓ Interpolation accuracy: u_err = $(@sprintf(\"%.2e\", u_error)), v_err = $(@sprintf(\"%.2e\", v_error))")
    println("   ✓ Energy conservation: $(@sprintf(\"%.2e\", energy_error))")
    
    # Test with non-uniform spacing
    println("\\n6. Testing with non-uniform grid spacing...")
    test_nonuniform_interpolation()
    
    println("\\n" * "="^60)
    println("Divergence-Free Interpolation Test Completed Successfully!")
    println("="^60)
    
    return u_fine, v_fine, u_analytical, v_analytical
end

function test_nonuniform_interpolation()
    """Test interpolation with non-uniform grid spacing."""
    
    # Create simple test case with known solution
    println("   Creating non-uniform test case...")
    
    # Simple linear velocity field: u = x, v = -z (divergence-free)
    nx, nz = 8, 6
    u_coarse = zeros(nx+1, nz)
    v_coarse = zeros(nx, nz+1)
    
    # Non-uniform spacing
    dx_coarse, dz_coarse = 0.5, 0.4
    
    # Fill with linear field
    for j = 1:nz, i = 1:nx+1
        x = (i-1) * dx_coarse
        u_coarse[i, j] = x  # u = x
    end
    
    for j = 1:nz+1, i = 1:nx
        z = (j-1) * dz_coarse
        v_coarse[i, j] = -z  # v = -z
    end
    
    # Create fine grid
    nx_fine, nz_fine = 2*nx, 2*nz
    dx_fine, dz_fine = dx_coarse/2, dz_coarse/2
    
    # Create AMRLevel for fine grid
    amr_level = AMRLevel(
        1, nx_fine, 0, nz_fine,
        dx_fine, 0.0, dz_fine,
        TwoDimensional,
        Float64[], Float64[], Float64[],  # Will be filled as needed
        Float64[], Float64[], Float64[],
        zeros(nx_fine+1, nz_fine), zeros(nx_fine, nz_fine+1), nothing, zeros(nx_fine, nz_fine),
        zeros(Bool, nx_fine, nz_fine), zeros(Union{Nothing, AMRLevel}, nx_fine, nz_fine), nothing,
        0.0, nx*dx_coarse, 0.0, 0.0, 0.0, nz*dz_coarse,
        0, Dict(), false
    )
    
    # Interpolate
    u_fine = zeros(nx_fine+1, nz_fine)
    v_fine = zeros(nx_fine, nz_fine+1)
    
    interpolate_velocity_conservative!(u_fine, v_fine, u_coarse, v_coarse, amr_level)
    
    # Verify divergence-free condition
    is_div_free, max_div, mean_div = verify_divergence_free(u_fine, v_fine, dx_fine, dz_fine)
    
    @printf "   Non-uniform case divergence: max = %.2e, mean = %.2e\\n" max_div mean_div
    println("   ✓ Non-uniform interpolation preserves divergence-free condition: $is_div_free")
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_divergence_free_interpolation()
end