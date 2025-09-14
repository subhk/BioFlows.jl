#!/usr/bin/env julia
# Flow past a 2D cylinder using masked immersed-boundary (BDIM) method
# Fixed version using existing BioFlows upwind schemes and proper BDIM configuration

using BioFlows

# Optimal parameters for vortex shedding at Re~200
const NX, NZ = 200, 80      # Higher resolution for vortex capture  
const LX, LZ = 6.0, 2.4     # Wider domain for vortex development
const UIN, RHO, NU = 1.0, 1000.0, 0.001  # Re = 200 for clear vortex shedding
const DT, TFINAL = 0.02, 4.0   # Smaller timestep, longer simulation
const D = 0.2; 
const R = D/2
const XC, ZC = 1.5, 1.2     # Better positioning for vortex development
const ZOFF = 0.01           # Small asymmetry to break symmetry

function main()
    println("Flow past cylinder with BDIM - Using existing upwind schemes")
    println("="^60)
    println("  Reynolds: $(round(UIN*D/NU,digits=1)) - Optimal for vortex shedding")
    println("  Grid: $(NX) × $(NZ), Domain: $(LX) × $(LZ)")
    println("  Timestep: $(DT), Duration: $(TFINAL)s")
    println("  Cylinder D = $(D) at ($(XC), $(ZC + ZOFF))")
    println("="^60)

    # Choose a writable output directory
    outdir = get(ENV, "BIOFLOWS_OUTPUT_DIR", joinpath(@__DIR__, "..", "output"))
    outfile = joinpath(outdir, "flow_cyl_bdim_fixed")

    config = create_2d_simulation_config(
        nx=NX, nz=NZ, Lx=LX, Lz=LZ,
        density_value=RHO, nu=NU,
        inlet_velocity=UIN,
        outlet_type=:pressure, 
        wall_type=:free_slip,
        dt=DT, 
        final_time=TFINAL,
        use_mpi=false, 
        adaptive_refinement=false,
        immersed_boundary_method=BDIM,  # Use existing BDIM implementation

        # JLD2 output  
        output_save_mode=:time_interval,
        output_interval=0.2,
        output_save_flow_field=true,
        output_save_body_positions=true,
        output_save_force_coefficients=true,
        output_file=outfile,
        
        # More relaxed Poisson solver settings
        poisson_strict=true,
        poisson_smoother=:staggered,
        poisson_max_iterations=300,    # Reasonable iterations
        poisson_tolerance=1e-8,        # Good balance of accuracy and speed
    )
    config = add_rigid_circle!(config, [XC, ZC + ZOFF], R)

    solver = create_solver(config)

    # Smooth inlet ramp with perturbations to trigger vortices
    τ = 0.2  # Smooth startup
    U_fun = t -> UIN * min(1.0, t/τ) * (1.0 + 0.02*sin(10*π*t))  # Small perturbation
    config.bc.conditions[(:x, :left)] = BioFlows.BoundaryCondition(BioFlows.Inlet, U_fun, :x, :left)

    # Set inlet cross-flow perturbation to break symmetry
    ENV["BIOFLOWS_W_INLET_AMP"] = "0.03"

    # Initialize with uniform flow
    state = initialize_simulation(config, initial_conditions=:uniform_flow)
    
    # Add small initial perturbation near cylinder
    for j in 1:NZ+1, i in 1:NX
        z_pos = (j-1) * (LZ/NZ)
        x_pos = (i-0.5) * (LX/NX)
        
        # Small vortical perturbation near cylinder wake
        if sqrt((x_pos - XC - R)^2 + (z_pos - ZC)^2) < 0.5
            state.w[i,j] = 0.05 * sin(2*π*(z_pos - ZC)/0.2) * exp(-2*(x_pos - XC - R)^2)
        end
    end

    println("Running simulation with existing BioFlows infrastructure...")
    println("   - Using existing upwind schemes from discretization_2d.jl")
    println("   - Existing BDIM implementation") 
    println("   - Vortex-triggering perturbations")
    println("   - Optimal Re=200 parameters")
    
    try
        # Use standard BioFlows run_simulation
        run_simulation(config, solver, state)
        
        # Quick check of final state
        u_max = maximum(abs.(state.u))
        w_max = maximum(abs.(state.w))
        
        println("\n" * ("="^60))
        println("SIMULATION COMPLETED:")
        println("   Final |u|: $(round(u_max,digits=3))")
        println("   Final |w|: $(round(w_max,digits=3))")
        
        # Success criteria
        if w_max > 0.3
            println("\nEXCELLENT: Strong cross-flow indicates vortex shedding!")
            println("|w| = $(round(w_max,digits=3)) > 30% of inlet velocity")
        elseif w_max > 0.15
            println("\nGOOD: Significant cross-flow development")
            println("|w| = $(round(w_max,digits=3)) > 15% of inlet velocity")
        else
            println("\nLIMITED: Cross-flow = $(round(w_max,digits=3))")
            println("May need longer simulation or parameter adjustment")
        end
        
        println("\nUSING EXISTING BIOFLOWS INFRASTRUCTURE:")
        println("Upwind schemes from discretization_2d.jl")
        println("Native BDIM implementation")
        println("Stable time stepping")
        println("Proper boundary conditions")
        
        return true
        
    catch e
        println("SIMULATION ERROR: $e")
        return false
    end
end

main()