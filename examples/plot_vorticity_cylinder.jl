# Plot 2D vorticity from BioFlows NetCDF output and overlay a cylinder
# Usage examples:
#   julia --project examples/plot_vorticity_cylinder.jl cylinder2d_const_inlet.nc
#   julia --project examples/plot_vorticity_cylinder.jl cylinder2d_const_inlet.nc --time last --xc 1.2 --zc 1.0 --radius 0.1
#   julia --project examples/plot_vorticity_cylinder.jl output_dir/cylinder2d_const_inlet_2.nc --time 10

using NetCDF
using Plots

function parse_args()
    # Minimal argument parser
    args = copy(ARGS)
    if isempty(args)
        error("Provide path to a NetCDF file (e.g., cylinder2d_const_inlet.nc)")
    end
    filepath = args[1]
    # Defaults
    tsel = :last
    xc = nothing
    zc = nothing
    R = nothing

    i = 2
    while i <= length(args)
        arg = args[i]
        if arg == "--time"
            i += 1
            val = args[i]
            if val == "last"
                tsel = :last
            else
                tsel = parse(Int, val)
            end
        elseif arg == "--xc"
            i += 1
            xc = parse(Float64, args[i])
        elseif arg == "--zc"
            i += 1
            zc = parse(Float64, args[i])
        elseif arg == "--radius"
            i += 1
            R = parse(Float64, args[i])
        else
            @warn "Unknown argument: $arg"
        end
        i += 1
    end
    return filepath, tsel, xc, zc, R
end

function read_field(nc, name)
    haskey(nc.vars, name) ? NetCDF.readvar(nc, name) : nothing
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

function interpolate_to_centers_u(u::AbstractArray)
    # u: (nx+1, nz) -> centers: (nx, nz)
    nxp1, nz = size(u)
    nx = nxp1 - 1
    ucc = zeros(eltype(u), nx, nz)
    @inbounds for j in 1:nz, i in 1:nx
        ucc[i, j] = 0.5 * (u[i, j] + u[i+1, j])
    end
    return ucc
end

function interpolate_to_centers_w(w::AbstractArray)
    # w: (nx, nz+1) -> centers: (nx, nz)
    nx, nzp1 = size(w)
    nz = nzp1 - 1
    wcc = zeros(eltype(w), nx, nz)
    @inbounds for j in 1:nz, i in 1:nx
        wcc[i, j] = 0.5 * (w[i, j] + w[i, j+1])
    end
    return wcc
end

function compute_vorticity_2d(ncfile::String; tsel=:last)
    nc = NetCDF.open(ncfile)
    # Coordinates
    x = NetCDF.readvar(nc, "x")
    z = NetCDF.readvar(nc, "z")
    # Velocity fields (2D files store vertical velocity as "w")
    u = read_field(nc, "u")
    w = read_field(nc, "w")
    time = haskey(nc.vars, "time") ? NetCDF.readvar(nc, "time") : nothing

    if u === nothing || w === nothing
        NetCDF.close(nc)
        error("NetCDF file does not contain required 2D variables 'u' and 'w'")
    end

    # Select time index
    nt = size(u, ndims(u))
    tidx = tsel === :last ? nt : Int(tsel)
    if tidx < 1 || tidx > nt
        NetCDF.close(nc)
        error("Invalid time index $(tidx); file has $(nt) snapshots")
    end

    # Extract 2D slices
    u2 = ndims(u) == 3 ? @view u[:, :, tidx] : u
    w2 = ndims(w) == 3 ? @view w[:, :, tidx] : w

    # Interpolate to cell centers
    u_cc = interpolate_to_centers_u(u2)
    w_cc = interpolate_to_centers_w(w2)

    # Grid spacing (assumed uniform)
    dx = mean(diff(x))
    dz = mean(diff(z))

    # Vorticity (y-component in XZ plane): ω = ∂w/∂x - ∂u/∂z
    dw_dx = central_diff_x(w_cc, dx)
    du_dz = central_diff_z(u_cc, dz)
    ω = dw_dx .- du_dz

    NetCDF.close(nc)
    return x, z, ω
end

function plot_vorticity(ncfile::String; tsel=:last, xc::Union{Nothing,Float64}=nothing,
                        zc::Union{Nothing,Float64}=nothing, R::Union{Nothing,Float64}=nothing)
    x, z, ω = compute_vorticity_2d(ncfile; tsel=tsel)

    # Heatmap expects matrices with y as rows; we transpose for Plots default
    hm = heatmap(x, z, ω', aspect_ratio=:equal, color=:balance,
                 xlab="x", ylab="z", title="Vorticity (t=$(tsel))")

    # Overlay cylinder if provided
    if xc !== nothing && zc !== nothing && R !== nothing
        θ = range(0, 2π, length=200)
        cx = xc .+ R .* cos.(θ)
        cz = zc .+ R .* sin.(θ)
        plot!(cx, cz, fillrange=zc .+ 0*θ, seriestype=:shape, fillcolor=:black, linecolor=:black, alpha=1.0)
    end

    pngfile = replace(ncfile, ".nc" => "_vorticity.png")
    savefig(hm, pngfile)
    println("Saved vorticity plot: $(pngfile)")
end

function main()
    filepath, tsel, xc, zc, R = parse_args()
    plot_vorticity(filepath; tsel=tsel, xc=xc, zc=zc, R=R)
end

main()

