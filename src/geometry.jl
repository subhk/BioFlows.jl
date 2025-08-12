abstract type Geometry end

struct Cylinder <: Geometry
    center::NTuple{2,Float64}
    radius::Float64
end

struct Disk <: Geometry
    center::NTuple{2,Float64}
    radius::Float64
end

struct Sphere <: Geometry
    center::NTuple{3,Float64}
    radius::Float64
end

# Signed distance functions (phi < 0 inside)
φ(g::Cylinder, x, y) = sqrt((x-g.center[1])^2 + (y-g.center[2])^2) - g.radius
φ(g::Disk, x, y) = sqrt((x-g.center[1])^2 + (y-g.center[2])^2) - g.radius
φ(g::Sphere, x, y, z) = sqrt((x-g.center[1])^2 + (y-g.center[2])^2 + (z-g.center[3])^2) - g.radius

# Smooth Heaviside Hε for ε ~ grid spacing
Hε(ϕ, ε) = ϕ > ε ? 1.0 : (ϕ < -ε ? 0.0 : 0.5*(1 + ϕ/ε + sinpi(ϕ/ε)/π))

# Body velocity fields; default zero (stationary)
ubody(::Geometry, x,y,t) = (0.0, 0.0)
ubody(::Geometry, x,y,z,t) = (0.0, 0.0, 0.0)

