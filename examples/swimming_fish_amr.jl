#!/usr/bin/env julia
#=
Swimming Fish with Adaptive Mesh Refinement
============================================

This example demonstrates:
1. Creating a flexible fish body using EulerBernoulliBeam
2. Setting up AMR that follows the deforming body
3. Applying traveling wave muscle activation
4. Running coupled beam-fluid simulation

The mesh automatically refines near the fish body and coarsens
in regions far from the fish, saving computational cost while
maintaining accuracy near the body.
=#

using BioFlows

println("=" ^ 70)
println("SWIMMING FISH WITH ADAPTIVE MESH REFINEMENT")
println("=" ^ 70)

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Fish parameters
L_fish = 0.2        # Fish body length [m]
h_max = 0.02        # Maximum body thickness [m]
ρ_fish = 1050.0     # Fish density [kg/m³]
E = 5e5             # Young's modulus [Pa]

# Flow parameters
Re = 500            # Reynolds number (lower for faster test)
St = 0.3            # Strouhal number for tail beat

# Domain and grid
domain = (2.0, 1.0)         # Physical domain [m]
grid_size = (128, 64)       # Base grid (coarse for testing)

# Time stepping
dt_beam = 1e-4      # Beam time step [s]
n_steps = 500       # Number of steps (short test)
print_interval = 100

# =============================================================================
# CREATE SIMULATION
# =============================================================================

println("\n[1] Setting up simulation...")

# AMR configuration for flexible body
amr_config = BeamAMRConfig(
    max_level = 2,                  # Up to 4x refinement
    beam_distance_threshold = 4.0,  # Refine within 4 cells of beam
    beam_weight = 0.7,              # Prioritize beam proximity
    gradient_weight = 0.2,          # Also refine at velocity gradients
    vorticity_weight = 0.1,         # And at vorticity features
    min_regrid_interval = 10,       # Don't regrid too often
    motion_threshold = 0.002,       # Regrid when beam moves 2mm
    regrid_interval = 50            # Force regrid every 50 steps
)

# Create the simulation using convenience constructor
sim = swimming_fish_simulation(
    L_fish = L_fish,
    Re = Re,
    St = St,
    n_nodes = 31,           # 31 beam nodes
    grid_size = grid_size,
    domain = domain,
    amr_config = amr_config,
    E = E,
    ρ_fish = ρ_fish,
    h_max = h_max
)

println("  Domain: $(domain[1]) × $(domain[2]) m")
println("  Grid: $(grid_size[1]) × $(grid_size[2]) (base)")
println("  Reynolds number: $Re")
println("  Strouhal number: $St")
println("  Fish length: $(L_fish * 100) cm")

# =============================================================================
# INITIAL STATUS
# =============================================================================

println("\n[2] Initial state:")
beam_info(sim)

# =============================================================================
# RUN SIMULATION
# =============================================================================

println("\n[3] Running simulation...")
println("-" ^ 50)

# Track metrics
force_history = Float64[]
displacement_history = Float64[]
regrid_count = 0
last_regrid = 0

for step in 1:n_steps
    # Advance simulation
    sim_step!(sim)

    # Track max displacement
    w_max = maximum(abs.(sim.beam.w))
    push!(displacement_history, w_max)

    # Check if regrid occurred
    if sim.tracker.last_regrid_step > last_regrid
        regrid_count += 1
        last_regrid = sim.tracker.last_regrid_step
    end

    # Print progress
    if step % print_interval == 0
        n_refined = num_refined_cells(sim.amr_sim.refined_grid)
        println("  Step $step: w_max=$(round(w_max*1000, digits=2))mm, " *
                "refined_cells=$n_refined, regrids=$regrid_count")
    end
end

# =============================================================================
# FINAL STATUS
# =============================================================================

println("\n[4] Final state:")
beam_info(sim)

# =============================================================================
# SUMMARY
# =============================================================================

println("\n" * "=" ^ 70)
println("SIMULATION SUMMARY")
println("=" ^ 70)

println("  Total steps: $n_steps")
println("  Total regrids: $regrid_count")
println("  Max tail displacement: $(round(maximum(displacement_history)*1000, digits=2)) mm")
println("  Final refined cells: $(num_refined_cells(sim.amr_sim.refined_grid))")

# Compute average displacement
if length(displacement_history) > 100
    avg_disp = sum(displacement_history[end-99:end]) / 100
    println("  Average displacement (last 100 steps): $(round(avg_disp*1000, digits=2)) mm")
end

println("\n" * "=" ^ 70)
println("SIMULATION COMPLETE!")
println("=" ^ 70)

#=
EXPECTED OUTPUT:
- The fish tail oscillates due to the traveling wave forcing
- AMR refines cells near the fish body
- Regridding occurs when the beam moves significantly
- Displacement amplitude reaches a quasi-steady state

NOTES:
- For production runs, use higher resolution (256×128 or 512×256)
- Increase n_steps for longer simulations
- Adjust Re and St for different swimming regimes
=#
