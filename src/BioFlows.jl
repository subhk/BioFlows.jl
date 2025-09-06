module BioFlows

    using LinearAlgebra
    using Statistics
    using ParametricBodies  
    using PencilArrays  
    using ForwardDiff  
    using NetCDF
    using MPI

    # Core types and infrastructure first
    include("core/types.jl")
    include("core/grids.jl")
    include("core/differential_operators.jl")
    include("boundary/boundary_conditions.jl")
    include("timestep/time_stepping.jl")
    include("mg/multigrid_solver.jl")
    include("mg/staggered_multigrid.jl")

    # Bodies need grids to be defined first, but come before AMR
    include("bodies/rigid_bodies.jl")
    
    # Flexible body features
    include("bodies/flexible_bodies.jl")
    include("bodies/distance_utilities.jl")
    include("bodies/flexible_body_controller.jl")
    include("bodies/coordinated_system_factory.jl")
    include("bodies/horizontal_plane_utilities.jl")

    # AMR can be included after body types are defined
    include("amr/adaptive_refinement.jl")
    include("amr/adaptive_refinement_v2.jl")
    include("amr/divergence_free_interpolation.jl")  # New divergence-free velocity interpolation
    include("amr/amr_pressure_solver.jl")  # Complete AMR pressure solver with interface conditions

    # Temporarily disable MPI AMR to fix dependencies
    # include("amr/adaptive_refinement_mpi.jl")
    include("boundary/boundary_layer_amr.jl")
    include("immersed/immersed_boundary.jl")

    include("2D/grid_2d.jl")
    include("2D/discretization_2d.jl")
    include("2D/navier_stokes_2d.jl")
    include("2D/mpi_2d.jl")

    include("3D/grid_3d.jl")
    include("3D/discretization_3d.jl")
    include("3D/navier_stokes_3d.jl")
    include("3D/mpi_3d.jl")

    # AMR integration (depends on solver types above)
    include("amr/amr_integration.jl")

    include("output/netcdf_writer.jl")
    include("api/simulation_api.jl")

    # High-level user API exports
    export SimulationConfig, create_2d_simulation_config, create_3d_simulation_config
    export add_rigid_circle!, add_rigid_square!  # add_flexible_body! temporarily disabled
    export create_solver, create_2d_solver, create_3d_solver, initialize_simulation, run_simulation
    export run_bioflow_2d, run_bioflow_3d

    # Grid and solver exports
    export StaggeredGrid, StaggeredGrid2D, StaggeredGrid3D
    export GridType, TwoDimensional, ThreeDimensional
    export NavierStokesSolver2D, NavierStokesSolver3D
    export MPINavierStokesSolver2D, MPINavierStokesSolver3D

    # Grid utilities
    export create_uniform_2d_grid, create_stretched_2d_grid, create_uniform_3d_grid, create_stretched_3d_grid
    export create_channel_3d_grid, create_wake_refined_3d_grid, create_cylindrical_3d_grid
    export refine_2d_grid_near_bodies, refine_3d_grid_near_bodies
    export print_grid_info_2d, print_grid_info_3d, validate_2d_grid, validate_3d_grid

    # Body exports  
    export RigidBody, RigidBodyCollection, FlexibleBody, FlexibleBodyCollection
    export Circle, Square, Rectangle
    export add_body!, is_inside, distance_to_surface, surface_normal
    export update_body_motion!, get_body_velocity_at_point, bodies_mask_2d, bodies_mask_3d
    export create_flag, create_vertical_flag, create_curved_flag, create_angled_flag, create_flag_collection
    export compute_body_distance, get_body_point, validate_control_points
    export compute_multi_body_distances, find_closest_points, distance_statistics
    export compute_body_center_of_mass, compute_body_bounding_box, print_distance_analysis
    export FlexibleBodyController, set_target_distances!, set_control_parameters!, reset_controller_state!
    export update_controller!, apply_harmonic_boundary_conditions!, monitor_distance_control, print_controller_status

    # Solution and state exports
    export SolutionState, SolutionState2D, SolutionState3D
    export FluidProperties, ConstantDensity, VariableDensity
    export SimulationParameters

    # Time stepping exports
    export TimeSteppingScheme, AdamsBashforth, RungeKutta2, RungeKutta4
    export time_step!

    # Boundary condition exports
    export BoundaryConditions, BoundaryConditions2D, BoundaryConditions3D
    export InletBC, PressureOutletBC, VelocityOutletBC, NoSlipBC, FreeSlipBC, PeriodicBC
    export apply_immersed_boundary_forcing!

    # Adaptive refinement exports (original)
    export AdaptiveRefinementCriteria, RefinedGrid, adapt_grid!

    # Advanced adaptive refinement exports (v2)
    export AMRLevel, AMRHierarchy, MPIAMRHierarchy
    export compute_refinement_indicators_amr, conservative_restriction_2d, bilinear_prolongation_2d
    export refine_amr_level!, estimate_truncation_error, coordinate_global_refinement!

    # Divergence-free velocity interpolation exports
    export interpolate_velocity_conservative!, interpolate_velocity_conservative_2d!
    export verify_divergence_free, compute_divergence_2d!, apply_divergence_correction_2d!

    # AMR pressure solver exports
    export solve_poisson_amr!, solve_poisson_level!, apply_interface_conditions!
    export compute_global_residual, compute_level_residual

    # Boundary layer AMR exports
    export BoundaryLayerAMRCriteria, compute_boundary_layer_indicators
    export compute_wall_distance_field, compute_y_plus_field
    export refine_for_boundary_layers!, apply_anisotropic_refinement!

    # Output exports
    export NetCDFConfig, NetCDFWriter, write_solution!, close!
    export save_body_force_coefficients!, save_complete_snapshot!, setup_netcdf_output
    # Temporarily disabled: export save_flexible_body_positions!, create_position_only_writer
    export save_body_kinematics_snapshot!, save_body_positions_only!

    # Differential operator exports
    export ddx, ddy, ddz, d2dx2, d2dy2, d2dz2
    export ddx_at_faces, ddy_at_faces
    export div, grad, laplacian
    export interpolate_u_to_cell_center, interpolate_v_to_cell_center, interpolate_to_cell_centers
    export verify_operator_accuracy, check_staggered_grid_consistency

    # Discretization exports  
    export divergence_2d!, gradient_pressure_2d!, advection_2d!, compute_diffusion_2d!
    export divergence_3d!, gradient_pressure_3d!, advection_3d!, compute_diffusion_3d!
    export compute_cfl_2d, compute_cfl_3d

    # Solver step functions
    export solve_step_2d!, solve_step_3d!

    # Advanced force calculation exports
    export compute_flexible_body_forces, compute_stress_force_accurate, compute_penalty_force_accurate
    export compute_constraint_force_accurate, regularized_delta_2d, interpolate_with_delta_function
    export compute_local_surface_properties, compute_adaptive_stiffness, compute_local_reynolds

    export create_coordinated_flag_system, setup_simple_two_flag_system, setup_multi_flag_chain
    export validate_system_configuration, print_system_summary
    export detect_horizontal_groups, create_horizontal_distance_matrix, setup_horizontal_plane_system
    export validate_horizontal_plane_configuration, print_horizontal_plane_analysis
    export compute_drag_lift_coefficients, compute_body_coefficients_collection, compute_instantaneous_power
    export add_flexible_bodies_with_controller!

    # Multigrid solver exports
    export MultigridPoissonSolver, solve_poisson!, show_solver_info
    export compute_pressure_gradient_to_faces!, compute_velocity_divergence_from_faces!
    export create_amr_integrated_solver, amr_solve_step!

end
