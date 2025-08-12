# BioFlow.jl

CPU-only, minimal incompressible flow simulator in Julia with 2D/3D variants and immersed-body support via Brinkman penalization. Inspired by WaterLily.jl, but intentionally kept simple and dependency-light for clarity and portability.

Features
- MAC grid (staggered) solver with projection method
- Semi-Lagrangian advection and explicit viscosity
- Conjugate gradient Poisson solver (Neumann BC)
- Immersed obstacles and swimmers via level-sets and penalization
- Examples: flow past cylinder/sphere, pulsing “jellyfish” disk/sphere

Quick start
- 2D cylinder: `julia --project -e 'include("examples/cylinder2d.jl")'`
- 3D sphere/cylinder proxy: `julia --project -e 'include("examples/cylinder3d.jl")'`
- 2D jellyfish: `julia --project -e 'include("examples/jellyfish2d.jl")'`
- 3D jellyfish: `julia --project -e 'include("examples/jellyfish3d.jl")'`

Notes
- This is a teaching/reference implementation. For performance, stability and features (AMR/GPU), see WaterLily.jl.
- Examples print max divergence periodically; you can add your own callbacks for IO/visualization.
