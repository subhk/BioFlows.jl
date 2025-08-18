# Project Overview  
- Can you write a full structure Julia code that simulates biological flows both in 2D or 3D.
  All the features should be in a separate folder (2D should be a 2D folder and so for 3D) with 
  a different file names. Also need to have an option to switch to 2D in x-z plane where z is the vertical direction.

# Governing equations (IMPORTANT):
- 2D or 3D Navier stokes equations. The equation should be dimensionalized.
- I also need an options for equation of density (deafult option: constant density).

# I need following features (IMPORTANT):
- The disctretization should be using finite volume method (2nd order accuracy) using staggered velocity-pressure variable placement.
- For 2D and 3D flow, the code should have different grid structure and user have a choice to define which one to choose with a nice API.
- The code should use boundary data immersion method to simulate the fluid flow around immersed bodies.
- Should have MPI capabilities.
- The pressure solve should use geometric multigrid solver.
- Need to have adpative grid point refinement, and data saving shoule be on the original grid points.
- I need both rigid-bodies and flexible bodies like flexible plate
- The output file can be rewritten in netcdf file format with a both iteration step and time interval save format. 
  It should have options of how many snapshots one file can save. 
- I need low-storage Adams-bashforth, RK3 and RK4 time-stepping schmes with user a choose. 

# For fexible bodies, I need following featues following boundary conditions (IMPORTANT):
- I can give a sinusoidal motion at the front with given amplitude and frequncy, with the rear-end it's free
- the front can be fixed, or it can rotate freely, with the rear-end it's free.
- It's should be only for the 2D flow. The governing equations are in Lagrangian-cordinates.
  The user should also have options for many points with thickness, rigidity etc.
  The flexible govering equations are in the flexible_bodies.pdf (equations (2.5) to (2.9); the pdf is in the repository). 

# Boundary conditions of the code  (IMPORTANT):
- An inlet (x-dircetion) I can presecribed while outlet can be pressure or velocity flux boundary condition both in 2D and 3D flows.
- In the vertical (z) dircetion, it can be no-slip, free-slip or periodic.
- In the y-direction, it can be can be no-slip, free-slip or periodic.

# Feraures (IMPORTANT):
- It should have many multiple bodies (for rigid- circles, squares)
- The grid should refined along the boundary of the bodies.

# Using codes (IMPORTANT):
- For multigrid solver use Julia code GeometricMultigrid.jl of Julia
- To generate body geomtry, use ParametricBodies.jl of Julia
- For MPI, use PencilArrays of Julia for MPI capabilite. Separate for 2D and 3D flows.
- Use ForwardDiff.jl of Julia for automatic differentiation 
 
# Reference codes
- WaterLily.jl of Julia
- LUMA code (webiste: https://github.com/cfdemons/LUMA)
- LUMA code documenation in the reposiroy LMUAdocs.pdf