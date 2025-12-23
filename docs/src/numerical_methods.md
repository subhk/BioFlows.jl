# Numerical Methods

This section describes the numerical methods used in BioFlows.jl for solving the incompressible Navier-Stokes equations.

## Governing Equations

BioFlows.jl solves the incompressible Navier-Stokes equations in dimensional form:

```math
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}
```

```math
\nabla \cdot \mathbf{u} = 0
```

where:
- $\mathbf{u} = (u, v)$ is the velocity field (m/s)
- $p$ is the pressure (Pa)
- $\rho$ is the density (kg/m³)
- $\nu$ is the kinematic viscosity (m²/s)

## Staggered Grid (MAC Grid)

BioFlows.jl uses a **staggered grid** arrangement, also known as a Marker-And-Cell (MAC) grid. This arrangement naturally satisfies the discrete divergence-free condition and avoids spurious pressure oscillations (checkerboard modes).

### Variable Locations in 2D

On a staggered grid, different variables are stored at different locations within each cell:

- **Pressure** ($p$): Cell centers
- **x-velocity** ($u$): Vertical cell faces (between left/right neighbors)
- **y-velocity** ($v$): Horizontal cell faces (between bottom/top neighbors)

```
         Δx
    ├─────────────┤

    ┌─────────────┬─────────────┬─────────────┐  ─┬─
    │             │             │             │   │
    │      ●      v      ●      v      ●      │   │
    │    p_i,j+1  │   p_i+1,j+1 │  p_i+2,j+1  │   │
    │             │             │             │   Δy
    u             u             u             u   │
    │             │             │             │   │
    │      ●      v      ●      v      ●      │  ─┴─
    │    p_i,j    │   p_i+1,j   │  p_i+2,j    │
    │             │             │             │
    u             u             u             u
    │             │             │             │
    │      ●      v      ●      v      ●      │
    │   p_i,j-1   │  p_i+1,j-1  │  p_i+2,j-1  │
    │             │             │             │
    └─────────────┴─────────────┴─────────────┘

    Legend:
    ●  = Pressure (p) at cell center
    u  = x-velocity component at vertical faces
    v  = y-velocity component at horizontal faces
```

### Detailed Single Cell View

For a single cell $(i,j)$, the staggered arrangement is:

```
                    v[i,j+1]
                       ↑
              ┌────────┼────────┐
              │        │        │
              │        │        │
    u[i,j] ───┼───── p[i,j] ────┼─── u[i+1,j]
      →       │        ●        │       →
              │                 │
              │                 │
              └────────┼────────┘
                       │
                       ↓
                    v[i,j]

    Grid spacing: Δx (horizontal), Δy (vertical)
    Cell center:  (i-½, j-½) in grid coordinates
    u location:   (i, j-½) - on left/right faces
    v location:   (i-½, j) - on bottom/top faces
```

### Indexing Convention

In BioFlows.jl, the arrays are indexed as follows:

| Variable | Array Index | Physical Location |
|----------|-------------|-------------------|
| `p[i,j]` | Cell $(i,j)$ | Center of cell $(i,j)$ |
| `u[i,j,1]` | Face $(i,j)$ | Left face of cell $(i,j)$ |
| `u[i,j,2]` | Face $(i,j)$ | Bottom face of cell $(i,j)$ |

The velocity field is stored in a single array `u[I,d]` where `I` is the cell index and `d` is the direction (1=x, 2=y for 2D; 1=x, 2=y, 3=z for 3D).

## Extension to 3D

In three dimensions, the staggered grid extends naturally:

- **Pressure** ($p$): Cell centers
- **x-velocity** ($u$): yz-faces (perpendicular to x)
- **y-velocity** ($v$): xz-faces (perpendicular to y)
- **z-velocity** ($w$): xy-faces (perpendicular to z)

```
                        z
                        │   y
                        │  /
                        │ /
                        │/
            ────────────┼──────────── x


                    ┌───────────────────┐
                   /│                  /│
                  / │       w_top     / │
                 /  │       ↑        /  │
                ┌───────────────────┐   │
                │   │      ●        │   │
          u_L → │   │    p_ijk      │ → u_R
                │   │               │   │
                │   └ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ┘
                │  /                │  /
                │ /      ↓         │ /
                │/    w_bottom     │/
                └───────────────────┘
                      ↗       ↘
                   v_front   v_back


    Legend:
    ●     = Pressure at cell center (i,j,k)
    u_L   = u[i,j,k,1] at left face
    u_R   = u[i+1,j,k,1] at right face
    v     = v[i,j,k,2] at front/back faces
    w     = w[i,j,k,3] at top/bottom faces
```

## Finite Difference Operators

### Divergence

The discrete divergence operator at cell center $(i,j)$:

```math
(\nabla \cdot \mathbf{u})_{i,j} = \frac{u_{i+1,j} - u_{i,j}}{\Delta x} + \frac{v_{i,j+1} - v_{i,j}}{\Delta y}
```

### Gradient

The discrete pressure gradient at face locations:

```math
\left(\frac{\partial p}{\partial x}\right)_{i,j} = \frac{p_{i,j} - p_{i-1,j}}{\Delta x} \quad \text{(at u-location)}
```

```math
\left(\frac{\partial p}{\partial y}\right)_{i,j} = \frac{p_{i,j} - p_{i,j-1}}{\Delta y} \quad \text{(at v-location)}
```

### Laplacian

The discrete Laplacian for the Poisson equation:

```math
\nabla^2 p_{i,j} = \frac{p_{i+1,j} - 2p_{i,j} + p_{i-1,j}}{\Delta x^2} + \frac{p_{i,j+1} - 2p_{i,j} + p_{i,j-1}}{\Delta y^2}
```

## Time Integration

BioFlows.jl uses a **2nd-order predictor-corrector** (Heun's method) combined with pressure projection to ensure incompressibility. This provides 2nd-order temporal accuracy.

### Predictor Step

First, compute a forward Euler prediction:

```math
\mathbf{u}^* = \mathbf{u}^n + \Delta t \left[ -(\mathbf{u}^n \cdot \nabla)\mathbf{u}^n + \nu \nabla^2 \mathbf{u}^n + \mathbf{g} \right]
```

Then project onto divergence-free space:

```math
\nabla^2 \phi = \nabla \cdot \mathbf{u}^*
```
```math
\mathbf{u}' = \mathbf{u}^* - \nabla \phi
```

### Corrector Step

Re-evaluate the right-hand side at the predicted velocity:

```math
\mathbf{f}' = -(\mathbf{u}' \cdot \nabla)\mathbf{u}' + \nu \nabla^2 \mathbf{u}' + \mathbf{g}
```

Average the predictor and corrector contributions (Heun's method):

```math
\mathbf{u}^{**} = \frac{1}{2}\left( \mathbf{u}' + \mathbf{u}^n + \Delta t \, \mathbf{f}' \right)
```

This is equivalent to the trapezoidal rule:

```math
\mathbf{u}^{**} = \mathbf{u}^n + \frac{\Delta t}{2} \left( \mathbf{f}^n + \mathbf{f}' \right)
```

Finally, project onto divergence-free space:

```math
\nabla^2 \psi = \frac{2}{\Delta t} \nabla \cdot \mathbf{u}^{**}
```
```math
\mathbf{u}^{n+1} = \mathbf{u}^{**} - \frac{\Delta t}{2} \nabla \psi
```

### Summary of the Algorithm

```
Algorithm: 2nd-order Predictor-Corrector with Pressure Projection
─────────────────────────────────────────────────────────────────
Input: uⁿ (divergence-free velocity at time tⁿ)
Output: uⁿ⁺¹ (divergence-free velocity at time tⁿ⁺¹)

1. Save: u⁰ ← uⁿ

2. PREDICTOR:
   a. f ← RHS(u⁰)                    // Convection + diffusion
   b. u* ← u⁰ + Δt·f                 // Forward Euler
   c. Solve ∇²φ = ∇·u*               // Pressure Poisson
   d. u' ← u* - ∇φ                   // Project to div-free

3. CORRECTOR:
   a. f' ← RHS(u')                   // Re-evaluate at predicted
   b. u** ← u' + u⁰ + Δt·f'          // Accumulate
   c. u** ← 0.5·u**                  // Average (Heun)
   d. Solve ∇²ψ = (2/Δt)·∇·u**       // Pressure Poisson
   e. uⁿ⁺¹ ← u** - (Δt/2)·∇ψ        // Final projection

4. Compute Δt from CFL condition
```

### CFL Condition

The time step is constrained by the CFL (Courant-Friedrichs-Lewy) condition:

```math
\Delta t \leq \left( \sum_d \frac{|u_d|}{\Delta x_d} + \sum_d \frac{2\nu}{\Delta x_d^2} \right)^{-1}
```

where the first term is the convective constraint and the second is the diffusive constraint.

## Pressure Solver

The pressure Poisson equation is solved using a **geometric multigrid** method with:

- Jacobi smoothing iterations
- Full-weighting restriction
- Bilinear interpolation for prolongation
- V-cycle iteration until convergence

The multigrid solver operates on a hierarchy of progressively coarser grids, enabling efficient solution of the elliptic pressure equation.

## Immersed Boundary Method

BioFlows.jl implements the **Boundary Data Immersion Method (BDIM)** for handling complex geometries. Bodies are defined implicitly through signed distance functions (SDFs):

```math
\phi(\mathbf{x}, t) < 0 \quad \text{inside body}
```
```math
\phi(\mathbf{x}, t) = 0 \quad \text{on boundary}
```
```math
\phi(\mathbf{x}, t) > 0 \quad \text{in fluid}
```

The SDF is used to:
1. Identify solid and fluid regions
2. Interpolate boundary conditions
3. Compute surface normals: $\mathbf{n} = \nabla\phi / |\nabla\phi|$
4. Calculate hydrodynamic forces on immersed bodies

## References

1. Harlow, F.H. and Welch, J.E. (1965). "Numerical calculation of time-dependent viscous incompressible flow of fluid with free surface." *Physics of Fluids*, 8(12), 2182-2189.

2. Weymouth, G.D. and Yue, D.K.P. (2011). "Boundary data immersion method for Cartesian-grid simulations of fluid-body interaction problems." *Journal of Computational Physics*, 230(16), 6233-6247.
