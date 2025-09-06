# Create an animation of 2D vorticity (XZ plane) from NetCDF snapshots
# Overlays rigid bodies (e.g., cylinder) as filled shapes.
#
# Usage:
#   julia --project examples/animate_vorticity_cylinder.jl path/to/run.nc
#   (Also detects rollover files: run_1.nc, run_2.nc, ...)

using NetCDF
using Plots

# -------------------- Helpers --------------------

function find_sequence_files(basefile::String)
    files = String[]
    if isfile(basefile)
        push!(files, basefile)
    else
        error("File not found: $(basefile)")
    end
    i = 1
    while true
        base, ext = splitext(basefile)
        f = string(base, "_", i, ext)
        if isfile(f)
            push!(files, f)
            i += 1
        else
            break
        end
    end
    return files
end

function central_diff_x(f::AbstractMatrix, dx::Float64)
    nx, nz = size(f)
    df = similar(f)
    @inbounds for j in 1:nz
        df[1, j] = (f[2, j] - f[1, j]) / dx
        for i in 2:nx-1
            df[i, j] = (f[i+1, j] - f[i-1, j]) / (2dx)
        end
        df[nx, j] = (f[nx, j] - f[nx-1, j]) / dx
    end
    return df
end

function central_diff_z(f::AbstractMatrix, dz::Float64)
    nx, nz = size(f)
    df = similar(f)
    @inbounds for i in 1:nx
        df[i, 1] = (f[i, 2] - f[i, 1]) / dz
        for j in 2:nz-1
            df[i, j] = (f[i, j+1] - f[i, j-1]) / (2dz)
        end
        df[i, nz] = (f[i, nz] - f[i, nz-1]) / dz
    end
    return df
end

interpolate_to_centers_u(u::AbstractMatrix) = (nxp1, nz = size(u); nx = nxp1-1; [0.5*(u[i,j]+u[i+1,j]) for i=1:nx, j=1:nz])
interpolate_to_centers_w(w::AbstractMatrix) = (nx, nzp1 = size(w); nz = nzp1-1; [0.5*(w[i,j]+w[i,j+1]) for i=1:nx, j=1:nz])

function read_coords_and_times(nc)
    x = NetCDF.readvar(nc, "x")
    z = NetCDF.readvar(nc, "z")
    time = haskey(nc.vars, "time") ? NetCDF.readvar(nc, "time") : collect(1:size(NetCDF.readvar(nc, "u"), ndims(NetCDF.readvar(nc, "u"))))
    return x, z, time
end

function vorticity_frame(nc, tidx::Int)
    x, z, _ = read_coords_and_times(nc)
    u = NetCDF.readvar(nc, "u")
    w = haskey(nc.vars, "w") ? NetCDF.readvar(nc, "w") : error("2D vertical velocity 'w' not found")
    u2 = ndims(u) == 3 ? @view u[:, :, tidx] : u
    w2 = ndims(w) == 3 ? @view w[:, :, tidx] : w
    u_cc = interpolate_to_centers_u(u2)
    w_cc = interpolate_to_centers_w(w2)
    dx = mean(diff(x)); dz = mean(diff(z))
    ω = central_diff_x(w_cc, dx) .- central_diff_z(u_cc, dz)
    return x, z, ω
end

function read_rigid_bodies_metadata(nc)
    bodies = []
    nrigid = haskey(nc.atts, ("global","rigid_bodies")) ? NetCDF.readatt(nc, "global", "rigid_bodies") : 0
    for i in 1:nrigid
        typ = haskey(nc.atts, ("global","body_$(i)_type")) ? NetCDF.readatt(nc, "global", "body_$(i)_type") : ""
        cx = haskey(nc.atts, ("global","body_$(i)_center_x")) ? NetCDF.readatt(nc, "global", "body_$(i)_center_x") : nothing
        cz = haskey(nc.atts, ("global","body_$(i)_center_z")) ? NetCDF.readatt(nc, "global", "body_$(i)_center_z") : nothing
        ang = haskey(nc.atts, ("global","body_$(i)_angle")) ? NetCDF.readatt(nc, "global", "body_$(i)_angle") : 0.0
        if typ == "Circle"
            r = NetCDF.readatt(nc, "global", "body_$(i)_radius")
            push!(bodies, (; typ=typ, cx=cx, cz=cz, r=r, ang=ang))
        elseif typ == "Square"
            side = NetCDF.readatt(nc, "global", "body_$(i)_side")
            push!(bodies, (; typ=typ, cx=cx, cz=cz, side=side, ang=ang))
        elseif typ == "Rectangle"
            w = NetCDF.readatt(nc, "global", "body_$(i)_width")
            h = NetCDF.readatt(nc, "global", "body_$(i)_height")
            push!(bodies, (; typ=typ, cx=cx, cz=cz, w=w, h=h, ang=ang))
        end
    end
    return bodies
end

function overlay_bodies!(bodies)
    for b in bodies
        if b.typ == "Circle"
            θ = range(0, 2π, length=200)
            px = b.cx .+ b.r .* cos.(θ)
            pz = b.cz .+ b.r .* sin.(θ)
            plot!(px, pz, seriestype=:shape, fillcolor=:black, linecolor=:black, alpha=1.0)
        elseif b.typ == "Square"
            s = b.side/2
            pts = [(-s,-s), (s,-s), (s,s), (-s,s), (-s,-s)]
            c = cos(b.ang); sθ = sin(b.ang)
            px = Float64[]; pz = Float64[]
            for (lx,lz) in pts
                xr = c*lx - sθ*lz; zr = sθ*lx + c*lz
                push!(px, b.cx + xr); push!(pz, b.cz + zr)
            end
            plot!(px, pz, seriestype=:shape, fillcolor=:black, linecolor=:black, alpha=1.0)
        elseif b.typ == "Rectangle"
            s1 = b.w/2; s2 = b.h/2
            pts = [(-s1,-s2), (s1,-s2), (s1,s2), (-s1,s2), (-s1,-s2)]
            c = cos(b.ang); sθ = sin(b.ang)
            px = Float64[]; pz = Float64[]
            for (lx,lz) in pts
                xr = c*lx - sθ*lz; zr = sθ*lx + c*lz
                push!(px, b.cx + xr); push!(pz, b.cz + zr)
            end
            plot!(px, pz, seriestype=:shape, fillcolor=:black, linecolor=:black, alpha=1.0)
        end
    end
end

function mask_bodies!(ω::AbstractMatrix, x::AbstractVector, z::AbstractVector, bodies)
    # Set ω to NaN inside rigid bodies to avoid drawing vorticity within solids
    nx, nz = size(ω)
    for b in bodies
        if b.typ == "Circle"
            @inbounds for j in 1:nz, i in 1:nx
                if (x[i]-b.cx)^2 + (z[j]-b.cz)^2 <= b.r^2
                    ω[i,j] = NaN
                end
            end
        elseif b.typ == "Square"
            s = b.side/2; c = cos(b.ang); sθ = sin(b.ang)
            @inbounds for j in 1:nz, i in 1:nx
                dx = x[i]-b.cx; dz = z[j]-b.cz
                xr = c*dx - sθ*dz; zr = sθ*dx + c*dz
                if abs(xr) <= s && abs(zr) <= s
                    ω[i,j] = NaN
                end
            end
        elseif b.typ == "Rectangle"
            s1 = b.w/2; s2 = b.h/2; c = cos(b.ang); sθ = sin(b.ang)
            @inbounds for j in 1:nz, i in 1:nx
                dx = x[i]-b.cx; dz = z[j]-b.cz
                xr = c*dx - sθ*dz; zr = sθ*dx + c*dz
                if abs(xr) <= s1 && abs(zr) <= s2
                    ω[i,j] = NaN
                end
            end
        end
    end
    return ω
end

function collect_frames_and_range(files)
    frames = Vector{Tuple{String,Int,Float64}}()
    ωmin = Inf; ωmax = -Inf
    for f in files
        nc = NetCDF.open(f)
        _, _, time = read_coords_and_times(nc)
        nt = length(time)
        for t = 1:nt
            x, z, ω = vorticity_frame(nc, t)
            ωmin = min(ωmin, minimum(ω)); ωmax = max(ωmax, maximum(ω))
            push!(frames, (f, t, time[t]))
        end
        NetCDF.close(nc)
    end
    return frames, ωmin, ωmax
end

# -------------------- Main --------------------

function main()
    if isempty(ARGS)
        println("Usage: julia --project examples/animate_vorticity_cylinder.jl run.nc")
        return
    end
    basefile = ARGS[1]
    files = find_sequence_files(basefile)
    frames, omin, omax = collect_frames_and_range(files)
    # Use symmetric color range around 0 for vorticity
    oabs = max(abs(omin), abs(omax))
    crange = (-oabs, oabs)

    # Prepare first file bodies metadata for overlay
    nc0 = NetCDF.open(files[1])
    bodies = read_rigid_bodies_metadata(nc0)
    NetCDF.close(nc0)

    # Animation
    # Optional palette via ENV VORTICITY_PALETTE (default :balance)
    pal_sym = try
        Symbol(get(ENV, "VORTICITY_PALETTE", "balance"))
    catch
        :balance
    end

    anim = @animate for (f, tidx, tval) in frames
        nc = NetCDF.open(f)
        x, z, ω = vorticity_frame(nc, tidx)
        NetCDF.close(nc)
        # Mask body interiors
        mask_bodies!(ω, x, z, bodies)
        ttl = @sprintf("Vorticity t=%.3f (%s)", tval, basename(f))
        heatmap(x, z, ω'; aspect_ratio=:equal, color=pal_sym, clims=crange,
                xlab="x", ylab="z", title=ttl)
        overlay_bodies!(bodies)
    end

    # Save GIF next to input
    base, _ = splitext(basefile)
    fps_gif = try parse(Int, get(ENV, "VORT_FPS_GIF", "15")) catch; 15 end
    gif(anim, string(base, "_vorticity.gif"); fps=fps_gif)
    println("Saved animation: ", string(base, "_vorticity.gif"))
    # Try MP4 as well if ffmpeg is available
    try
        mp4path = string(base, "_vorticity.mp4")
        fps_mp4 = try parse(Int, get(ENV, "VORT_FPS_MP4", "30")) catch; 30 end
        mp4(anim, mp4path; fps=fps_mp4)
        println("Saved animation: ", mp4path)
    catch e
        @warn "Could not save MP4 animation (ffmpeg missing or error): $e"
    end
end

main()
