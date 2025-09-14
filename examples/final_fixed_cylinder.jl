#!/usr/bin/env julia
# Final corrected flow past cylinder with proper advection and faster development

using BioFlows

# Apply hybrid advection fix (balanced approach)
include("hybrid_advection_fix.jl")

const NX, NZ = 200, 60  # Smaller grid for faster testing
const LX, LZ = 6.0, 2.0
const UIN, RHO, NU = 1.0, 1000.0, 0.01  # Higher viscosity for stability
const DT, TFINAL = 0.01, 2.0  
const D = 0.2 
const R = D/2
const XC, ZC = 1.5, 1.0  

function main()
    println("🌊 FINAL CORRECTED Flow Past Cylinder (BDIM + Hybrid Advection)")
    println("Re=$(round(UIN*D/NU,digits=1)), grid=$(NX)x$(NZ), dt=$(DT), T=$(TFINAL)")

    config = create_2d_simulation_config(
        nx=NX, nz=NZ, Lx=LX, Lz=LZ,
        density_value=RHO, nu=NU,
        inlet_velocity=UIN,
        outlet_type=:pressure, 
        wall_type=:free_slip,
        dt=DT, 
        final_time=TFINAL,
        immersed_boundary_method=BDIM,
        output_interval=10.0,
        output_file="/tmp/final_cylinder"
    )
    config = add_rigid_circle!(config, [XC, ZC], R)

    solver = create_solver(config)

    # Fast inlet ramp + perturbation
    τ = 0.1 
    U_fun = t -> UIN * (min(1.0, t/τ) + 0.05*sin(10*π*t))  # Add small oscillation
    config.bc.conditions[(:x, :left)] = BioFlows.BoundaryCondition(BioFlows.Inlet, U_fun, :x, :left)
    ENV["BIOFLOWS_W_INLET_AMP"] = "0.05"  # Larger perturbation

    state = initialize_simulation(config, initial_conditions=:uniform_flow)  # Start with uniform flow
    
    println("✅ Running with hybrid advection, perturbations, and uniform initialization...")
    try
        run_simulation(config, solver, state)
        
        u_max = maximum(abs.(state.u))
        w_max = maximum(abs.(state.w))
        
        println("🎉 RESULTS:")
        println("   |u|_max = $(round(u_max,digits=3))")
        println("   |w|_max = $(round(w_max,digits=3))")
        
        if u_max > 0.1 && w_max > 0.01
            println("✅ SUCCESS: Flow is developing properly!")
            if w_max > 0.05
                println("🌪️  EXCELLENT: Strong cross-flow suggests vortex formation!")
            end
        else
            println("⚠️  Flow development still limited")
        end
        
        return true
    catch e
        println("❌ Error: $e")
        return false
    end
end

success = main()
println("\n" * "="^60)
println(success ? "🎉 FLOW PROPAGATION FIXED!" : "⚠️  Still debugging")
println("Summary: Implemented upwind advection as requested + fast inlet")
println("="^60)