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

## Convection-Diffusion Discretization

The momentum equation contains convection and diffusion terms that must be carefully discretized on the staggered grid.

### Momentum Equation

For velocity component $u_i$, the semi-discrete momentum equation is:

```math
\frac{\partial u_i}{\partial t} = -\sum_j \frac{\partial (u_j u_i)}{\partial x_j} + \nu \sum_j \frac{\partial^2 u_i}{\partial x_j^2} - \frac{1}{\rho}\frac{\partial p}{\partial x_i}
```

The convection term $\partial(u_j u_i)/\partial x_j$ is in **conservative (flux) form**, representing the divergence of momentum flux. This is distinct from the non-conservative form $u_j \partial u_i/\partial x_j$.

### 2D Momentum Equations (Conservative Form)

For 2D flow with coordinates $(x, z)$ and velocities $(u, w)$:

**x-momentum (u):**
```math
\frac{\partial u}{\partial t} = -\frac{\partial (uu)}{\partial x} - \frac{\partial (wu)}{\partial z} + \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial z^2}\right) - \frac{1}{\rho}\frac{\partial p}{\partial x}
```

**z-momentum (w):**
```math
\frac{\partial w}{\partial t} = -\frac{\partial (uw)}{\partial x} - \frac{\partial (ww)}{\partial z} + \nu\left(\frac{\partial^2 w}{\partial x^2} + \frac{\partial^2 w}{\partial z^2}\right) - \frac{1}{\rho}\frac{\partial p}{\partial z}
```

The terms $\partial(uu)/\partial x$, $\partial(wu)/\partial z$, $\partial(uw)/\partial x$, and $\partial(ww)/\partial z$ are the conservative convective fluxes.

### Finite Volume Formulation

BioFlows.jl uses a **conservative finite volume method (FVM)** where fluxes are computed at cell faces and applied symmetrically to adjacent cells. This ensures exact conservation of momentum.

For a control volume around velocity $u_i$ at location $I$:

```math
\frac{d u_i^I}{d t} = \sum_j \left( F_{i,j}^{I} - F_{i,j}^{I-\delta_j} \right)
```

where $F_{i,j}^I$ is the total flux (convective + diffusive) of momentum component $i$ through face $j$ at index $I$.

### Flux Computation at Cell Faces

At each face in direction $j$, two fluxes contribute to the momentum balance:

#### 1. Convective Flux

The convective flux transports momentum with the flow velocity:

```math
F_{i,j}^{conv} = \frac{u_j^{face}}{\Delta x_j} \cdot \phi(u_i)
```

where:
- $u_j^{face}$ = face-normal velocity (interpolated to face)
- $\phi(u_i)$ = upwind-biased reconstruction of $u_i$ at the face
- $\Delta x_j$ = grid spacing in direction $j$

The face velocity $u_j^{face}$ is computed by interpolation:
```math
u_j^{face} = \frac{1}{2}(u_j^L + u_j^R)
```

#### 2. Diffusive Flux

The diffusive flux represents viscous stress:

```math
F_{i,j}^{diff} = -\frac{\nu}{\Delta x_j} \left( u_i^I - u_i^{I-\delta_j} \right)
```

This is a central difference approximation to $-\nu \partial u_i / \partial x_j$.

#### Total Flux

The total flux at face $I$ in direction $j$ for momentum component $i$:

```math
F_{i,j}^I = F_{i,j}^{conv,I} + F_{i,j}^{diff,I}
```

### Upwind Schemes for Convection

BioFlows.jl implements several upwind schemes for reconstructing face values. Given the stencil values $u_U$ (upwind), $u_C$ (center), $u_D$ (downwind), the schemes compute the face value $\phi$.

#### QUICK with Median Limiter

BioFlows uses a modified QUICK scheme with a median limiter for stability:

```math
\phi_{QUICK} = \text{median}\left( \frac{5u_C + 2u_D - u_U}{6}, \, u_C, \, \text{median}(10u_C - 9u_U, \, u_C, \, u_D) \right)
```

The median limiter prevents spurious oscillations near discontinuities while maintaining high accuracy in smooth regions. For smooth monotonic profiles, this reduces to the quadratic interpolation $(5u_C + 2u_D - u_U)/6$.

**Code:** `quick(u,c,d) = median((5c+2d-u)/6, c, median(10c-9u,c,d))`

#### Van Leer (TVD)

The van Leer scheme uses a monotonicity-preserving limiter:

```math
\phi_{vanLeer} = \begin{cases}
u_C & \text{if } u_C \leq \min(u_U, u_D) \text{ or } u_C \geq \max(u_U, u_D) \\
u_C + (u_D - u_C) \cdot \frac{u_C - u_U}{u_D - u_U} & \text{otherwise}
\end{cases}
```

This ensures the interpolated value lies between neighboring values, preventing oscillations.

**Code:** `vanLeer(u,c,d) = (c≤min(u,d) || c≥max(u,d)) ? c : c+(d-c)*(c-u)/(d-u)`

#### Central Difference (CDS)

The central difference scheme provides 2nd-order accuracy but may oscillate:

```math
\phi_{CDS} = \frac{u_C + u_D}{2}
```

**Code:** `cds(u,c,d) = (c+d)/2`

#### Stencil Selection

The upwind direction is determined by the face velocity $u_j^{face}$:

```math
\phi(u_i) = \begin{cases}
\lambda(u_i^{I-2\delta}, u_i^{I-\delta}, u_i^{I}) & \text{if } u_j^{face} > 0 \\
\lambda(u_i^{I+\delta}, u_i^{I}, u_i^{I-\delta}) & \text{if } u_j^{face} < 0
\end{cases}
```

where $\lambda$ is the chosen scheme (quick, vanLeer, or cds).

### Conservative Flux Application

The key feature of FVM is that **the same flux value is added to one cell and subtracted from its neighbor**:

```
Cell I-1:    r[I-1] -= F[I]    (flux leaves)
Cell I:      r[I]   += F[I]    (flux enters)
```

This ensures that momentum is exactly conserved — no momentum is created or destroyed at internal faces.

```
        Face I
          ↓
    ┌─────┼─────┐
    │     │     │
    │ I-1 │  I  │
    │     │     │
    └─────┼─────┘
          │
     -F ←─┼─→ +F
```

### Code Implementation

The FVM is implemented in `src/Flow.jl`. Here's how the math maps to code:

#### Flux Storage (Optional)

```julia
# Flow struct fields for explicit flux storage
F_conv :: Array{T,D+2}  # Convective flux F_conv[I,j,i]
F_diff :: Array{T,D+2}  # Diffusive flux F_diff[I,j,i]
store_fluxes :: Bool    # Enable FVM mode
```

The flux tensor has indices:
- `I` = spatial cell index (D-dimensional)
- `j` = face direction (1=x, 2=y, 3=z)
- `i` = momentum component

#### Computing Fluxes

```julia
# From compute_face_flux! in src/Flow.jl
for i ∈ 1:n, j ∈ 1:n
    inv_Δxj = 1/Δx[j]
    ν_Δxj = ν/Δx[j]

    # Interior faces
    @loop (
        # Convective flux: (1/Δx) * u_face * ϕ(u)
        F_conv[I,j,i] = inv_Δxj * ϕu(j, CI(I,i), u, ϕ(i,CI(I,j),u), λ);
        # Diffusive flux: -(ν/Δx) * ∂u/∂x
        F_diff[I,j,i] = -ν_Δxj * ∂(j, CI(I,i), u)
    ) over I ∈ inside_u(N,j)

    # Boundary fluxes (one-sided stencils)
    compute_boundary_flux!(...)
end
```

Key functions:
- `ϕ(i,I,u)` — Interpolates velocity component `i` to face location
- `ϕu(j,I,u,u_face,λ)` — Computes upwind flux using scheme `λ` (quick, vanLeer, cds)
- `∂(j,I,u)` — Central difference $u^I - u^{I-\delta_j}$

#### Applying Fluxes Conservatively

```julia
# From apply_fluxes! in src/Flow.jl
for i ∈ 1:n, j ∈ 1:n
    F_total = F_conv[I,j,i] + F_diff[I,j,i]

    # Lower boundary: only flux INTO domain
    @loop r[I,i] += F_total over I ∈ slice(N,2,j,2)

    # Interior: flux enters I, leaves I-δ (CONSERVATIVE!)
    @loop r[I,i] += F_total over I ∈ inside_u(N,j)
    @loop r[I-δ(j,I),i] -= F_total over I ∈ inside_u(N,j)

    # Upper boundary: only flux OUT OF domain
    @loop r[I-δ(j,I),i] -= F_total over I ∈ slice(N,N[j],j,2)
end
```

### Enabling FVM Mode

To use explicit flux storage and verification:

```julia
# Enable FVM with flux storage
sim = Simulation((nx, ny), (Lx, Ly);
                 store_fluxes = true,  # Enable FVM mode
                 ν = 0.01)

# Run simulation
sim_step!(sim)

# Access stored fluxes for analysis
F_conv = sim.flow.F_conv  # Convective fluxes
F_diff = sim.flow.F_diff  # Diffusive fluxes

# Verify conservation (sum of internal fluxes = 0)
```

When `store_fluxes=false` (default), the original method is used which computes fluxes on-the-fly without storing them.

### Boundary Flux Treatment

At domain boundaries, fluxes are handled specially since there's no neighbor cell outside:

| Boundary | Treatment | Stencil |
|----------|-----------|---------|
| Lower (index 2) | One-sided upwind | `ϕuL` |
| Upper (index N) | One-sided upwind | `ϕuR` |
| Periodic | Wrap-around | `ϕuP` |

```julia
# Lower boundary: use left-biased stencil
F_conv[I,j,i] = ϕuL(j, I, u, u_face, λ)

# Upper boundary: use right-biased stencil
F_conv[I,j,i] = ϕuR(j, I, u, u_face, λ)

# Periodic: wrap to opposite boundary
F_conv[I,j,i] = ϕuP(j, I_wrapped, I, u, u_face, λ)
```

### Conservation Verification

The FVM ensures exact momentum conservation. For a closed system with no external forces:

```math
\frac{d}{dt} \sum_I u_i^I \cdot \Delta V = \sum_{\text{boundaries}} F_{i}^{boundary}
```

Interior fluxes cancel exactly because each internal face contributes:
- $+F$ to cell $I$
- $-F$ to cell $I-\delta$

This property is crucial for accurate long-time simulations and proper vortex dynamics.

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

## Immersed Boundary Method (BDIM)

BioFlows.jl implements the **Boundary Data Immersion Method (BDIM)** for handling complex geometries, including moving and deforming bodies. This section provides the complete mathematical formulation.

### Signed Distance Function (SDF)

Bodies are defined implicitly through signed distance functions:

```math
\phi(\mathbf{x}, t) < 0 \quad \text{inside body}
```
```math
\phi(\mathbf{x}, t) = 0 \quad \text{on boundary}
```
```math
\phi(\mathbf{x}, t) > 0 \quad \text{in fluid}
```

The SDF provides:
- **Distance to surface**: $d = \phi(\mathbf{x}, t)$
- **Surface normal**: $\mathbf{n} = \nabla\phi / |\nabla\phi|$
- **Curvature**: $\kappa = \nabla \cdot \mathbf{n}$

### BDIM Kernel Functions

The BDIM uses a smooth kernel to transition from fluid to solid over a band of width $2\epsilon$ centered at the body surface. The kernel is a raised cosine:

```math
K(\xi) = \frac{1}{2} + \frac{1}{2}\cos(\pi\xi), \quad \xi \in [-1, 1]
```

where $\xi = d/\epsilon$ is the normalized distance.

#### Zeroth Moment (Volume Fraction)

The volume fraction $\mu_0$ represents how much of a cell is fluid:

```math
\mu_0(d, \epsilon) = \int_{-1}^{d/\epsilon} K(s) \, ds = \frac{1}{2} + \frac{d}{2\epsilon} + \frac{1}{2\pi}\sin\left(\frac{\pi d}{\epsilon}\right)
```

Properties:
- $\mu_0 = 0$ when $d \leq -\epsilon$ (fully inside solid)
- $\mu_0 = 1$ when $d \geq +\epsilon$ (fully in fluid)
- $\mu_0 = 0.5$ at $d = 0$ (on surface)

#### First Moment (Gradient Correction)

The first moment $\mu_1$ provides gradient information for boundary layer resolution:

```math
\mu_1(d, \epsilon) = \epsilon \int_{-1}^{d/\epsilon} s \cdot K(s) \, ds = \epsilon \left[ \frac{1}{4}\left(1 - \frac{d^2}{\epsilon^2}\right) - \frac{1}{2\pi^2}\left(\frac{d}{\epsilon}\sin\frac{\pi d}{\epsilon} + \frac{1 + \cos\frac{\pi d}{\epsilon}}{\pi}\right) \right]
```

### BDIM Velocity Update

The BDIM enforces no-slip/no-penetration at immersed boundaries by blending the fluid velocity $\mathbf{u}^*$ with the body velocity $\mathbf{V}$:

```math
\mathbf{u} = \mu_0 \cdot \mathbf{f} + \mathbf{V} + \boldsymbol{\mu}_1 \cdot \nabla \mathbf{f}
```

where the correction field is:

```math
\mathbf{f} = \mathbf{u}^0 + \Delta t \cdot \mathbf{RHS} - \mathbf{V}
```

and:
- $\mathbf{u}^0$ = velocity at previous time step
- $\mathbf{RHS}$ = convection + diffusion terms
- $\mathbf{V}$ = body velocity at each point
- $\boldsymbol{\mu}_1 = \mu_1 \cdot \mathbf{n}$ = directional first moment

This formulation:
1. Smoothly transitions from fluid velocity to body velocity
2. Maintains proper boundary layer behavior
3. Conserves momentum at the interface

### Reference

Maertens, A.P. and Weymouth, G.D. (2015). "Accurate Cartesian-grid simulations of near-body flows at intermediate Reynolds numbers." *Computer Methods in Applied Mechanics and Engineering*, 283, 106-129. doi:[10.1016/j.cma.2014.09.007](https://doi.org/10.1016/j.cma.2014.09.007)

## Flexible Body Kinematics

BioFlows supports time-varying body geometries through two mechanisms:
1. **Coordinate mapping** for rigid body motion (translation, rotation)
2. **Time-dependent SDF** for flexible/deforming bodies

### Coordinate Mapping (Rigid Motion)

For rigid bodies, the SDF shape is fixed but the body moves via a coordinate mapping $\mathbf{m}(\mathbf{x}, t)$:

```math
\phi(\mathbf{x}, t) = \phi_0(\mathbf{m}(\mathbf{x}, t))
```

where $\phi_0$ is the reference SDF and $\mathbf{m}$ maps world coordinates to body-fixed coordinates.

#### Body Velocity from Mapping

The body velocity is computed from the coordinate mapping using:

```math
\mathbf{V} = -\mathbf{J}^{-1} \cdot \frac{\partial \mathbf{m}}{\partial t}
```

where $\mathbf{J} = \partial \mathbf{m} / \partial \mathbf{x}$ is the Jacobian of the mapping.

**Derivation**: For a material point $\boldsymbol{\xi} = \mathbf{m}(\mathbf{x}, t)$ fixed in the body frame, we have $D\boldsymbol{\xi}/Dt = 0$. Using the chain rule:

```math
\frac{\partial \mathbf{m}}{\partial t} + \mathbf{J} \cdot \dot{\mathbf{x}} = 0
```

Solving for the velocity $\dot{\mathbf{x}} = \mathbf{V}$:

```math
\mathbf{V} = -\mathbf{J}^{-1} \cdot \frac{\partial \mathbf{m}}{\partial t}
```

#### Examples

**Oscillating Cylinder** (vertical sinusoidal motion):

```math
\mathbf{m}(\mathbf{x}, t) = \mathbf{x} - \begin{pmatrix} 0 \\ A\sin(\omega t) \end{pmatrix}
```

Body velocity:

```math
\mathbf{V} = \begin{pmatrix} 0 \\ A\omega\cos(\omega t) \end{pmatrix}
```

**Rotating Body** (angular velocity $\Omega$):

```math
\mathbf{m}(\mathbf{x}, t) = \mathbf{R}(-\Omega t) \cdot (\mathbf{x} - \mathbf{x}_c)
```

where $\mathbf{R}(\theta)$ is the rotation matrix and $\mathbf{x}_c$ is the center of rotation.

### Time-Dependent SDF (Flexible Bodies)

For flexible bodies where the shape itself changes over time, the SDF is directly time-dependent:

```math
\phi = \phi(\mathbf{x}, t)
```

The surface normal is computed as:

```math
\mathbf{n} = \frac{\nabla \phi}{|\nabla \phi|}
```

For **pseudo-SDFs** (implicit functions where $|\nabla \phi| \neq 1$), the distance is corrected:

```math
d = \frac{\phi}{|\nabla \phi|}
```

## Swimming Fish Kinematics

BioFlows implements flexible swimming bodies using traveling wave motion. The fish body is defined by a time-varying SDF based on the centerline displacement.

### Body Centerline

The fish centerline follows a traveling wave with optional leading edge motion:

```math
y(s, t) = y_{head}(t) + y_{pitch}(s, t) + y_{wave}(s, t)
```

where $s \in [0, L]$ is the arc length from head to tail.

#### 1. Leading Edge Heave

Sinusoidal vertical oscillation at the head:

```math
y_{head}(t) = h_{heave} \sin(\omega t + \phi_{heave})
```

- $h_{heave}$ = heave amplitude
- $\omega = 2\pi f$ = angular frequency
- $\phi_{heave}$ = phase offset

#### 2. Leading Edge Pitch

Angular oscillation at the head pivot:

```math
y_{pitch}(s, t) = s \cdot \sin(\theta(t))
```

where the pitch angle is:

```math
\theta(t) = \theta_{max} \sin(\omega t + \phi_{pitch})
```

- $\theta_{max}$ = maximum pitch angle (radians)
- $\phi_{pitch}$ = pitch phase (typically $\pi/2$ for optimal thrust)

#### 3. Traveling Wave

Body undulation with position-dependent amplitude:

```math
y_{wave}(s, t) = A(s) \sin(ks - \omega t + \phi)
```

- $k = 2\pi/\lambda$ = wave number
- $\lambda$ = wavelength (relative to body length)
- $A(s)$ = amplitude envelope
- $\phi$ = phase offset

### Amplitude Envelopes

Different swimming modes have characteristic amplitude distributions:

#### Carangiform (Tail-Dominated)

Typical of tuna, mackerel — stiff body with tail propulsion:

```math
A(s) = A_{tail} \left(\frac{s}{L}\right)^2
```

#### Anguilliform (Whole-Body)

Typical of eels, lampreys — entire body participates:

```math
A(s) = A_{head} + (A_{tail} - A_{head}) \frac{s}{L}
```

#### Subcarangiform (Intermediate)

Typical of trout, carp — between carangiform and anguilliform:

```math
A(s) = A_{tail} \left(\frac{s}{L}\right)^{1.5}
```

### Fish Body SDF

The fish body is modeled as a NACA-like profile with the centerline as its axis:

```math
\phi(\mathbf{x}, t) = |z - z_{body}(x, t)| - h(s)
```

where:
- $z_{body}(x, t) = z_{center} + y(s, t)$ is the centerline position
- $h(s) = h_{max} \cdot 4 \frac{s}{L}\left(1 - \frac{s}{L}\right)$ is the local half-thickness (NACA profile)
- $s = x - x_{head}$ is the position along the body

### Complete Kinematics

Combining all components, the centerline displacement is:

```math
y(s, t) = h_{heave}\sin(\omega t + \phi_{heave}) + s\sin\left(\theta_{max}\sin(\omega t + \phi_{pitch})\right) + A(s)\sin(ks - \omega t + \phi)
```

The body velocity at each point is obtained by taking the time derivative:

```math
\frac{\partial y}{\partial t} = h_{heave}\omega\cos(\omega t + \phi_{heave}) + s\theta_{max}\omega\cos(\omega t + \phi_{pitch})\cos(\theta(t)) - A(s)\omega\cos(ks - \omega t + \phi)
```

### Strouhal Number

The Strouhal number characterizes the swimming efficiency:

```math
St = \frac{f \cdot A_{tail}}{U}
```

where:
- $f$ = oscillation frequency
- $A_{tail}$ = tail amplitude
- $U$ = swimming velocity

Optimal propulsion typically occurs at $St \approx 0.2 - 0.4$.

### Reynolds Number

The Reynolds number based on fish length:

```math
Re = \frac{U \cdot L}{\nu}
```

where:
- $U$ = characteristic velocity (inflow or swimming speed)
- $L$ = fish body length
- $\nu$ = kinematic viscosity

## Fish School Kinematics

For multiple fish, each fish has its own phase offset:

```math
y_i(s, t) = y_{head,i}(t) + y_{pitch,i}(s, t) + A(s)\sin(ks - \omega t + \phi_i)
```

The combined SDF for the school uses the union operation:

```math
\phi_{school}(\mathbf{x}, t) = \min_i \phi_i(\mathbf{x}, t)
```

### Phase Relationships

Different phase relationships create different schooling behaviors:

| Pattern | Phase Offset | Description |
|---------|--------------|-------------|
| Synchronized | $\phi_i = 0$ | All fish in phase |
| Wave | $\phi_i = i \cdot \Delta\phi$ | Progressive phase delay |
| Alternating | $\phi_i = (i \mod 2) \cdot \pi$ | Adjacent fish anti-phase |

## Boundary Conditions

BioFlows.jl supports three types of boundary conditions for the domain boundaries (not to be confused with immersed body boundaries handled by BDIM).

### Domain Boundary Overview

```
                    Top boundary (j = nz)
                    ─────────────────────
                    │                   │
                    │                   │
    Inlet           │                   │    Outlet
    (i = 1)         │     Domain        │    (i = nx)
    inletBC         │                   │    outletBC
                    │                   │
                    │                   │
                    ─────────────────────
                    Bottom boundary (j = 1)
```

### 1. Inlet Boundary Condition (`inletBC`)

The inlet boundary (at $x = 0$) uses a **Dirichlet condition** where velocity is prescribed.

#### Constant Inlet

For uniform inflow, specify a tuple:

```julia
inletBC = (U, 0.0)  # u = U, v = 0 at inlet
```

This sets:
```math
u(0, y, t) = U, \quad v(0, y, t) = 0
```

#### Spatially-Varying Inlet

For non-uniform profiles (e.g., parabolic channel flow), use a function:

```julia
# Parabolic profile: u(y) = U_max * (1 - (y-H)²/H²)
H = Ly / 2  # channel half-height
U_max = 1.5
inletBC(i, x, t) = i == 1 ? U_max * (1 - ((x[2] - H) / H)^2) : 0.0
```

The function signature is `inletBC(i, x, t)` where:
- `i` = velocity component (1 = x, 2 = y/z)
- `x` = position vector
- `t` = time

#### Time-Varying Inlet

For pulsatile or oscillating inflow:

```julia
# Oscillating inlet: u(t) = U₀(1 + A·sin(ωt))
inletBC(i, x, t) = i == 1 ? U₀ * (1 + 0.1*sin(2π*t)) : 0.0
```

!!! note "Velocity Scale Required"
    When using a function for `inletBC`, you must specify `U` (velocity scale) explicitly since it cannot be auto-computed.

### 2. Convective Outlet Boundary Condition (`outletBC`)

The outlet boundary (at $x = L_x$) is the most challenging because **we don't know the flow state there in advance**. Simple conditions like zero-gradient ($\partial u/\partial x = 0$) cause **spurious reflections** — pressure waves bounce back into the domain and contaminate the solution.

#### The Convective BC Approach

The convective (or advective) outlet condition assumes flow structures are **transported out** of the domain at a convection velocity $U_c$:

```math
\frac{\partial u}{\partial t} + U_c \frac{\partial u}{\partial x} = 0
```

This is a 1D wave equation that advects the local velocity pattern out of the domain.

#### Discretization

Using first-order upwind differencing:

```math
u_i^{n+1} = u_i^n - U_c \Delta t \frac{u_i^n - u_{i-1}^n}{\Delta x}
```

In BioFlows, $U_c$ is taken as the mean inlet velocity to ensure mass conservation.

#### Mass Conservation Correction

After applying the convective BC, a correction ensures global mass conservation:

```math
\oint u \, dA = 0 \quad \text{(for incompressible flow)}
```

The outlet velocity is adjusted so that mass flux out equals mass flux in:

```julia
# From src/util.jl - exitBC!
U = mean(u_inlet)           # Average inlet flux
u_outlet = u_outlet - Δt * U * ∂u/∂x  # Convective update
correction = mean(u_outlet) - U       # Mass imbalance
u_outlet = u_outlet - correction      # Enforce conservation
```

#### Why Convective BC Works

```
Without Convective BC:              With Convective BC:

    ────────────────────┐              ────────────────────→
    Vortex → → → ↩ ↩ ↩  │              Vortex → → → → → →
    ────────────────────┘              ────────────────────→
                ↑                                  ↑
          Reflection!                      Passes through
```

The convective BC allows vortices, wakes, and other flow structures to exit smoothly without generating artificial reflections.

#### Usage

```julia
sim = Simulation((nx, nz), (Lx, Lz);
                 inletBC = (1.0, 0.0),
                 outletBC = true)      # Enable convective outlet
```

### 3. Periodic Boundary Condition (`perdir`)

Periodic boundaries make the domain wrap around — flow exiting one side re-enters from the opposite side.

```math
u(x, 0, t) = u(x, L_y, t), \quad v(x, 0, t) = v(x, L_y, t)
```

#### When to Use Periodic BC

| Scenario | Direction | Example |
|----------|-----------|---------|
| Infinite span | z (spanwise) | Flow past cylinder |
| Channel flow | x (streamwise) | Fully-developed pipe flow |
| Homogeneous turbulence | All | Isotropic turbulence box |

#### Usage

```julia
# Periodic in z-direction (direction 2)
sim = Simulation((nx, nz), (Lx, Lz);
                 inletBC = (1.0, 0.0),
                 perdir = (2,))

# Periodic in both y and z (3D)
sim = Simulation((nx, ny, nz), (Lx, Ly, Lz);
                 inletBC = (1.0, 0.0, 0.0),
                 perdir = (2, 3))
```

### 4. Default (No-Flux) Boundaries

Boundaries not explicitly set use a **zero normal gradient** (Neumann) condition:

```math
\frac{\partial u}{\partial n} = 0
```

This is appropriate for:
- Slip walls (free-slip, no penetration)
- Symmetry planes
- Far-field boundaries (approximate)

### Boundary Condition Summary

| Parameter | Condition | Mathematical Form | Use Case |
|-----------|-----------|-------------------|----------|
| `inletBC` | Dirichlet | $u = u_{prescribed}$ | Inflow boundaries |
| `outletBC=true` | Convective | $\partial_t u + U \partial_x u = 0$ | Outflow (prevents reflections) |
| `perdir=(d,)` | Periodic | $u(0) = u(L)$ | Infinite/repeating domains |
| (default) | Neumann | $\partial_n u = 0$ | Slip walls, symmetry |

### Common Configurations

#### External Flow (Wake Problems)

```julia
# Flow past cylinder: inlet + convective outlet + periodic spanwise
sim = Simulation((nx, nz), (Lx, Lz);
                 inletBC = (U, 0.0),
                 outletBC = true,
                 perdir = (2,),
                 body = AutoBody(sdf))
```

#### Channel Flow

```julia
# Fully-developed channel: periodic streamwise + no-slip walls
sim = Simulation((nx, nz), (Lx, Lz);
                 inletBC = (U, 0.0),
                 perdir = (1,))  # Periodic in x (streamwise)
```

#### Closed Cavity

```julia
# Lid-driven cavity: no outlet, no periodic
sim = Simulation((nx, nz), (Lx, Lz);
                 inletBC = (U, 0.0))  # Top wall moves at U
```

## References

1. Harlow, F.H. and Welch, J.E. (1965). "Numerical calculation of time-dependent viscous incompressible flow of fluid with free surface." *Physics of Fluids*, 8(12), 2182-2189.

2. Weymouth, G.D. and Yue, D.K.P. (2011). "Boundary data immersion method for Cartesian-grid simulations of fluid-body interaction problems." *Journal of Computational Physics*, 230(16), 6233-6247.

3. Orlanski, I. (1976). "A simple boundary condition for unbounded hyperbolic flows." *Journal of Computational Physics*, 21(3), 251-269.

4. Leonard, B.P. (1979). "A stable and accurate convective modelling procedure based on quadratic upstream interpolation." *Computer Methods in Applied Mechanics and Engineering*, 19(1), 59-98. (QUICK scheme)

5. Van Leer, B. (1979). "Towards the ultimate conservative difference scheme. V. A second-order sequel to Godunov's method." *Journal of Computational Physics*, 32(1), 101-136. (Van Leer limiter)

6. Versteeg, H.K. and Malalasekera, W. (2007). *An Introduction to Computational Fluid Dynamics: The Finite Volume Method*. Pearson Education.
