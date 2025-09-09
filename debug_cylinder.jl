#!/usr/bin/env julia

"""
Debug script to understand why the cylinder simulation is zeroing all velocities
"""

println("🔍 DEBUGGING CYLINDER FORCING")
println("="^60)

# Simulate the cylinder setup from flow_past_cylinder_2d_serial.jl
xc, zc = 1.2, 1.0  # Cylinder center
R = 0.1            # Cylinder radius
nx, nz = 32, 16    # Grid size
Lx, Lz = 6.0, 2.0  # Domain size

# Calculate grid spacing
dx = Lx / nx
dz = Lz / nz

println("Domain setup:")
println("  Domain: $Lx × $Lz")
println("  Grid: $nx × $nz")
println("  Grid spacing: dx=$dx, dz=$dz")
println("  Cylinder center: ($xc, $zc)")
println("  Cylinder radius: $R")
println()

# Create grid coordinate arrays (simplified)
x_centers = range(dx/2, Lx - dx/2, length=nx)
z_centers = range(dz/2, Lz - dz/2, length=nz)
x_u_faces = range(0.0, Lx, length=nx+1)  # u-velocity faces
z_w_faces = range(0.0, Lz, length=nz+1)  # w-velocity faces

println("Grid coordinates:")
println("  x_centers range: $(x_centers[1]) to $(x_centers[end])")
println("  z_centers range: $(z_centers[1]) to $(z_centers[end])")
println("  x_u_faces range: $(x_u_faces[1]) to $(x_u_faces[end])")
println("  z_w_faces range: $(z_w_faces[1]) to $(z_w_faces[end])")
println()

# Test cylinder detection function
function is_inside_cylinder_simple(center_x, center_z, radius, x, z)
    dx = x - center_x
    dz = z - center_z
    return (dx^2 + dz^2) <= radius^2
end

# Check how many grid points are inside the cylinder
global inside_count_u = 0
global inside_count_w = 0
total_u_points = (nx+1) * nz
total_w_points = nx * (nz+1)

println("Checking u-velocity points ($(nx+1) × $nz = $total_u_points points):")
for j = 1:nz, i = 1:(nx+1)
    x = x_u_faces[i]
    z = z_centers[j]
    if is_inside_cylinder_simple(xc, zc, R, x, z)
        global inside_count_u += 1
        if inside_count_u <= 5  # Show first few
            println("  u[$i,$j] at ($x, $z) is INSIDE cylinder")
        end
    end
end

println("Checking w-velocity points ($nx × $(nz+1) = $total_w_points points):")
for j = 1:(nz+1), i = 1:nx
    x = x_centers[i]
    z = z_w_faces[j]
    if is_inside_cylinder_simple(xc, zc, R, x, z)
        global inside_count_w += 1
        if inside_count_w <= 5  # Show first few
            println("  w[$i,$j] at ($x, $z) is INSIDE cylinder")
        end
    end
end

println()
println("Summary:")
println("  u-velocity points inside cylinder: $inside_count_u / $total_u_points")
println("  w-velocity points inside cylinder: $inside_count_w / $total_w_points")

if inside_count_u == 0 && inside_count_w == 0
    println("  ❌ NO POINTS INSIDE CYLINDER - cylinder may be outside domain!")
elseif inside_count_u > total_u_points/2 || inside_count_w > total_w_points/2
    println("  ❌ TOO MANY POINTS INSIDE - cylinder may be too large or coordinates wrong!")
else
    println("  ✅ Reasonable number of points inside cylinder")
end

# Check if cylinder center is within domain
if xc >= 0 && xc <= Lx && zc >= 0 && zc <= Lz
    println("  ✅ Cylinder center is within domain")
else
    println("  ❌ Cylinder center is OUTSIDE domain!")
    println("     Domain: (0,0) to ($Lx,$Lz)")
    println("     Cylinder center: ($xc,$zc)")
end

# Check inlet boundary - are we setting inlet velocity correctly?
println()
println("🔍 CHECKING INLET BOUNDARY CONDITIONS")
println("-"^40)

# Inlet should be at x=0, with velocity Uin=1.0
Uin = 1.0
print("Inlet velocity points (x=0): ")
global inlet_points = 0
for j = 1:nz
    x = x_u_faces[1]  # First u-face is at x=0
    z = z_centers[j]
    if x == 0.0
        global inlet_points += 1
        if inlet_points <= 3
            print("u[1,$j] ")
        end
    end
end
println("\nTotal inlet velocity points: $inlet_points")

if inlet_points == 0
    println("❌ NO INLET POINTS FOUND - boundary conditions may not be applied!")
else
    println("✅ Inlet points found")
end

println()
println("🎯 DIAGNOSIS:")
if inside_count_u == 0 && inside_count_w == 0
    println("The cylinder is not affecting any grid points.")
    println("This suggests either:")
    println("  1. Cylinder is outside the computational domain")
    println("  2. Cylinder is too small relative to grid resolution")
    println("  3. Coordinate system mismatch")
else
    println("The cylinder is affecting grid points correctly.")
    println("Zero velocities likely due to:")
    println("  1. Inlet boundary conditions not being applied")
    println("  2. Too aggressive forcing zeroing the entire field")
    println("  3. Missing or incorrect initial conditions")
end