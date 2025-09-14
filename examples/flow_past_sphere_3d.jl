#!/usr/bin/env julia
# Flow past a 3D sphere (predictor–corrector + IB); NetCDF to /tmp

using BioFlows

const NX, NY, NZ = 96, 48, 48
const LX, LY, LZ = 6.0, 3.0, 3.0
const UIN, RHO, NU = 1.0, 1000.0, 0.0008
const DT, TFINAL = 5e-3, 2.0
const D = 0.6; const R = D/2
const XC, YC, ZC = 1.2, 1.5, 1.5

function main()
    println("Flow past sphere: Re=$(round(UIN*D/NU,digits=1)) grid=$(NX)x$(NY)x$(NZ) dt=$(DT) T=$(TFINAL)")

    config = create_3d_simulation_config(
        nx=NX, ny=NY, nz=NZ, Lx=LX, Ly=LY, Lz=LZ,
        density_value=RHO, nu=NU,
        inlet_velocity=UIN,
        outlet_type=:pressure, wall_type=:no_slip,
        dt=DT, final_time=TFINAL,
        use_mpi=false, adaptive_refinement=false,
        immersed_boundary_method=VolumePenalty,  # BDIM 3D falls back with warning
        output_interval=0.1, output_file="/tmp/flow_sphere3d",
        poisson_strict=true,
        poisson_smoother=:staggered,
        poisson_max_iterations=500,
        poisson_tolerance=1e-10,
    )
    config = add_rigid_circle!(config, [XC, YC, ZC], R)  # Circle shape acts as sphere in 3D helpers

    solver = create_solver(config)
    state0 = initialize_simulation(config, initial_conditions=:quiescent)
    # Start from uniform inlet and add tiny transverse perturbation to break symmetry
    state0.u .= UIN
    state0.v .= 0.0
    state0.w .= 0.0

    run_simulation(config, solver, state0)
end

main()

