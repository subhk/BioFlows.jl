# BioFlows.jl

[![CI](https://github.com/subhk/BioFlows.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/subhk/BioFlows.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/subhk/BioFlows.jl/actions/workflows/documentation.yml/badge.svg)](https://subhk.github.io/BioFlows.jl/dev/)

A Julia package for computational fluid dynamics (CFD) simulations with immersed
boundary methods. BioFlows provides a complete solver for incompressible viscous
flow on Cartesian grids using the Boundary Data Immersion Method (BDIM).

## Features

- Pure Julia solver for incompressible Navier-Stokes equations
- Immersed boundary method via BDIM (Boundary Data Immersion Method)
- Implicit geometry definition through signed distance functions
- Adaptive Mesh Refinement (AMR) near bodies and flow features

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/subhk/BioFlows.jl")
```

Or activate the project locally:

```bash
git clone https://github.com/subhk/BioFlows.jl.git
cd BioFlows.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

