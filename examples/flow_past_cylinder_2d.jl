# =============================================================================
# Flow Past Cylinder 2D Example
# =============================================================================
# Classic benchmark: uniform flow past a stationary cylinder.
# Demonstrates force coefficient and field snapshot output.
# =============================================================================

using BioFlows

# --- Simulation Parameters ---
nx, nz = 240, 240          # Grid resolution
Lx, Lz = 4.0, 4.0          # Domain size (meters)
U = 1.0                    # Inlet velocity (m/s)
ν = 0.001                  # Kinematic viscosity (m²/s)
radius = 0.2               # Cylinder radius (meters)
diameter = 2 * radius      # Characteristic length
inletBC = (U, 0.0)         # Uniform inlet velocity

# --- Grid Setup ---
dx = Lx / nx
center_x = nx / 12 - 1     # Cylinder center (grid units)
center_z = nz / 2 - 1
radius_cells = radius / dx

# --- Define Cylinder Geometry ---
sdf(x, t) = sqrt((x[1] - center_x)^2 + (x[2] - center_z)^2) - radius_cells
body = AutoBody(sdf)

# --- Create Simulation ---
# Time stepping: adaptive CFL (default) or fixed
# For fixed time step, add: fixed_Δt = 0.001
sim = Simulation((nx, nz), (Lx, Lz);
                 ν = ν,
                 body = body,
                 L_char = diameter,
                 perdir = (2,),       # Periodic in z
                 inletBC = inletBC,
                 outletBC = true,     # Convective outlet
                 fixed_Δt=0.001)     

# --- Output Writers ---
force_writer = ForceWriter("force_coefficients.jld2";
                           interval = 0.1,
                           reference_area = diameter)

field_writer = CenterFieldWriter("flow_fields.jld2";
                                 interval = 1.0)

# --- Time Stepping ---
final_time = 50.0
print_interval = 100

iter = 0
while sim_time(sim) < final_time
    global iter += 1
    sim_step!(sim)

    # Save outputs
    file_save!(force_writer, sim)
    file_save!(field_writer, sim)

    # Print progress
    if iter % print_interval == 0
        t = round(sim_time(sim), digits=2)
        Cd, Cl = force_coefficients(sim)
        println("Step $iter: t = $t, Cd = $(round(Cd, digits=4)), Cl = $(round(Cl, digits=4))")
    end
end

# --- Summary ---
println("\nSimulation complete!")
println("  Final time: $(round(sim_time(sim), digits=2))")
println("  Force samples: $(force_writer.samples)")
println("  Field snapshots: $(field_writer.samples)")
println("\nOutput files:")
println("  $(force_writer.filename)")
println("  $(field_writer.filename)")
