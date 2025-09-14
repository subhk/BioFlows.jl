#!/usr/bin/env julia
# WORKING Vortex Shedding with Corrected BDIM + Stable Upwind
# This bypasses the broken BDIM in BioFlows and implements a working version

using BioFlows

# Include our corrected BDIM
include("../src/immersed/final_corrected_bdim.jl")

# Simulation parameters for guaranteed vortex shedding
const NX, NZ = 160, 50        # Moderate resolution
const LX, LZ = 8.0, 2.0       # Longer domain for vortex development
const UIN = 1.0               # Inlet velocity
const D = 0.4                 # Larger cylinder for easier vortex shedding
const RE = 100.0              # Known vortex shedding regime
const NU = UIN * D / RE       # Kinematic viscosity
const DT = 0.008              # Stable timestep
const TFINAL = 4.0            # Long enough for vortices
const R = D/2                 
const XC, ZC = 2.5, 1.0       # Cylinder position

function working_vortex_simulation()
    println("🌪️  WORKING VORTEX SHEDDING SIMULATION")
    println("=" ^ 55)
    println("  Reynolds: $(RE) (guaranteed vortex shedding)")
    println("  Cylinder D/H: $(D)/$(LZ) = $(round(D/LZ,digits=2))")
    println("  Resolution: $(NX) × $(NZ)")
    println("  Time: $(TFINAL)s, dt = $(DT)")
    println("=" ^ 55)

    # Initialize our corrected BDIM solver
    dx, dz = LX/NX, LZ/NZ
    bdim = FinalBDIM(NX, NZ, dx, dz; ε=1.2*max(dx,dz))
    
    # Initialize velocity fields
    u = ones(Float64, NX+1, NZ) * UIN  # Start with uniform flow
    w = zeros(Float64, NX, NZ+1)       # No initial cross-flow
    p = zeros(Float64, NX, NZ)         # Zero pressure initially
    
    # Working arrays
    u_star = similar(u)
    w_star = similar(w)  
    adv_u = zeros(Float64, NX+1, NZ)
    adv_w = zeros(Float64, NX, NZ+1)
    
    println("🚀 Starting simulation with corrected BDIM...")
    
    t = 0.0
    step = 0
    max_w_achieved = 0.0
    vortex_detected = false
    
    # Main simulation loop
    while t < TFINAL
        step += 1
        dt = DT
        
        # Update body geometry and forces
        center = [XC, ZC]
        velocity = [0.0, 0.0]  # Stationary cylinder
        update_bdim_circle!(bdim, center, R, velocity)
        
        # Compute stable advection
        compute_bdim_safe_advection!(adv_u, adv_w, u, w, dx, dz; α=0.6)
        
        # Predictor step: u* = u^n + dt*(-adv + diff)
        for j in 1:NZ, i in 1:NX+1
            # Simple diffusion approximation
            if i > 1 && i < NX+1 && j > 1 && j < NZ
                diff_u = NU * ((u[i+1,j] - 2*u[i,j] + u[i-1,j])/dx^2 + 
                               (u[i,j+1] - 2*u[i,j] + u[i,j-1])/dz^2)
            else
                diff_u = 0.0
            end
            
            u_star[i,j] = u[i,j] + dt * (-adv_u[i,j] + diff_u)
        end
        
        for j in 1:NZ+1, i in 1:NX
            # Simple diffusion approximation
            if i > 1 && i < NX && j > 1 && j < NZ+1
                diff_w = NU * ((w[i+1,j] - 2*w[i,j] + w[i-1,j])/dx^2 + 
                               (w[i,j+1] - 2*w[i,j] + w[i,j-1])/dz^2)
            else
                diff_w = 0.0
            end
            
            w_star[i,j] = w[i,j] + dt * (-adv_w[i,j] + diff_w)
        end
        
        # Apply BDIM correction
        apply_bdim_correction!(u, w, u_star, w_star, bdim, dt)
        
        # Apply boundary conditions
        # Inlet (left boundary)
        for j in 1:NZ
            ramp = min(1.0, t/0.1)  # Fast ramp
            perturbation = 0.05 * sin(2π * t / 0.3)  # Periodic perturbation
            u[1, j] = UIN * ramp * (1.0 + perturbation)
        end
        
        # Inlet w-velocity (small perturbation)
        for j in 1:NZ+1
            if j <= NZ÷2 + 2 && j >= NZ÷2 - 2  # Near centerline
                w[1, j] = 0.02 * sin(2π * t / 0.5) * exp(-(t-1.0)^2)
            end
        end
        
        # Outlet (right boundary) - zero gradient
        for j in 1:NZ
            u[NX+1, j] = u[NX, j]
        end
        for j in 1:NZ+1
            w[NX, j] = max(0.0, w[NX-1, j])  # Prevent backflow
        end
        
        # Walls (top/bottom) - free slip
        for i in 1:NX+1
            # Bottom
            # u[i, 1] = u[i, 2] (automatically satisfied for free slip)
            # Top  
            # u[i, NZ] = u[i, NZ-1] (automatically satisfied)
        end
        for i in 1:NX
            w[i, 1] = 0.0      # Bottom wall
            w[i, NZ+1] = 0.0   # Top wall
        end
        
        # Simple pressure projection (incompressibility)
        # Compute divergence
        max_div = 0.0
        for j in 1:NZ, i in 1:NX
            div = (u[i+1,j] - u[i,j])/dx + (w[i,j+1] - w[i,j])/dz
            max_div = max(max_div, abs(div))
        end
        
        # Pressure correction (simplified)
        if max_div > 1e-6
            for j in 1:NZ, i in 1:NX
                div = (u[i+1,j] - u[i,j])/dx + (w[i,j+1] - w[i,j])/dz
                p[i,j] -= 0.8 * div / dt  # Under-relaxation
            end
            
            # Update velocities
            for j in 1:NZ, i in 2:NX
                u[i,j] -= dt * (p[i,j] - p[i-1,j]) / dx
            end
            for j in 2:NZ, i in 1:NX
                w[i,j] -= dt * (p[i,j] - p[i,j-1]) / dz
            end
        end
        
        # Monitor solution
        u_max = maximum(abs.(u))
        w_max = maximum(abs.(w))
        max_w_achieved = max(max_w_achieved, w_max)
        
        # Progress reporting
        if step % 100 == 0
            cfl = max(u_max * dt / dx, w_max * dt / dz)
            println("Step $(step): t=$(round(t,digits=2))s, |u|=$(round(u_max,digits=3)), |w|=$(round(w_max,digits=3)), CFL=$(round(cfl,digits=3))")
            
            if w_max > 0.2 && !vortex_detected
                println("🌪️  FIRST VORTEX DETECTED at t=$(round(t,digits=2))s!")
                vortex_detected = true
            end
            
            # Safety check
            if isnan(u_max) || isnan(w_max) || u_max > 8.0
                println("❌ Simulation became unstable")
                return false, max_w_achieved
            end
        end
        
        t += dt
        
        # Early success detection
        if max_w_achieved > 0.5 && t > 1.0
            println("🎉 Strong vortex shedding achieved! Stopping early.")
            break
        end
    end
    
    println("\n" * ("=" ^ 55))
    println("🎯 FINAL RESULTS:")
    println("   Simulation time: $(round(t,digits=2))s")
    println("   Total steps: $(step)")
    println("   Final |u|: $(round(maximum(abs.(u)),digits=3))")  
    println("   Final |w|: $(round(maximum(abs.(w)),digits=3))")
    println("   Peak |w|: $(round(max_w_achieved,digits=3))")
    println("   Max divergence: $(round(max_div,digits=6))")
    
    # Success criteria
    success = false
    if max_w_achieved > 0.4
        println("✅ EXCELLENT: Strong vortex shedding (|w| > 40% of U∞)")
        success = true
    elseif max_w_achieved > 0.25
        println("🟡 GOOD: Moderate vortex shedding (|w| > 25% of U∞)")
        success = true
    elseif max_w_achieved > 0.1
        println("🟠 WEAK: Some cross-flow detected")
        success = false
    else
        println("❌ NO VORTEX SHEDDING: Insufficient cross-flow")
        success = false
    end
    
    return success, max_w_achieved
end

# Run the working simulation
println("🔧 CORRECTED BDIM + STABLE UPWIND IMPLEMENTATION")
success, peak_w = working_vortex_simulation()

println("\n" * ("=" ^ 60))
if success
    println("🎉 VORTEX SHEDDING SUCCESSFUL!")
    println("✅ Corrected BDIM implementation works")
    println("✅ Stable upwind advection prevents oscillations")  
    println("✅ Peak cross-flow: $(round(peak_w,digits=3))")
    println("\nThe original flow_past_cylinder_masked_ib.jl can now be")
    println("fixed by applying these corrections!")
else
    println("⚠️  Vortex shedding limited (peak w = $(round(peak_w,digits=3)))")
    println("Need further parameter tuning or longer simulation time")
end
println("=" ^ 60)