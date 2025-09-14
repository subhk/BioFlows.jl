module JLD2Output

using ..BioFlows
import ..BioFlows: StaggeredGrid, SolutionState, SolutionState2D, SolutionState3D,
                   MPISolutionState2D, MPISolutionState3D,
                   RigidBodyCollection, FlexibleBodyCollection,
                   FluidProperties, ConstantDensity,
                   RefinedGrid,
                   interpolate_to_cell_centers_xz,
                   interpolate_to_cell_centers
using MPI
using JLD2
using Dates
using Printf: @sprintf

# Reuse NetCDFConfig for save cadence/settings to avoid new config surface
const OutputConfig = BioFlows.NetCDFConfig

mutable struct JLD2Writer
    filepath::String
    grid::StaggeredGrid
    config::OutputConfig
    current_snapshot::Int
    last_save_time::Float64
    last_save_iteration::Int
    # Stack-only buffers
    times::Vector{Float64}
    iterations::Vector{Int}
    u_slices::Vector{Any}
    v_slices::Vector{Any}
    w_slices::Vector{Any}
    p_slices::Vector{Any}
    centers::Vector{Any}
    coeff_Cd::Vector{Vector{Float64}}
    coeff_Cl::Vector{Vector{Float64}}
    coeff_Fx::Vector{Vector{Float64}}
    coeff_Fz::Vector{Vector{Float64}}
    coeff_Cdp::Vector{Vector{Float64}}
    coeff_Cdv::Vector{Vector{Float64}}
    amr_maps::Vector{Any}
    # Periodic flush controls
    flush_every::Int
    block_index::Int
    flush_time::Float64  # simulation time window to flush (seconds); 0 disables
    # Flexible body buffers (per-sample, per-body)
    flex_X::Vector{Vector{Array{Float64,2}}}
    flex_F::Vector{Vector{Array{Float64,2}}}
    flex_T::Vector{Vector{Vector{Float64}}}
    flex_K::Vector{Vector{Vector{Float64}}}
    flex_V::Vector{Vector{Array{Float64,2}}}   # optional velocity
    flex_A::Vector{Vector{Array{Float64,2}}}   # optional acceleration
    # Flexible body metadata (per body, set on first encounter)
    flex_lengths::Vector{Float64}
    flex_npoints::Vector{Int}
    flex_svecs::Vector{Vector{Float64}}
    
    function JLD2Writer(filepath::String, grid::StaggeredGrid, config::OutputConfig)
        # Ensure directory exists
        try
            outdir = dirname(filepath)
            if !isempty(outdir) && outdir != "." && !isdir(outdir)
                mkpath(outdir)
            end
        catch
        end
        flush_every = try parse(Int, get(ENV, "BIOFLOWS_JLD2_FLUSH_EVERY", "0")) catch; 0 end
        flush_time = try parse(Float64, get(ENV, "BIOFLOWS_JLD2_FLUSH_TIME", "0")) catch; 0.0 end
        new(filepath, grid, config, 0, 0.0, 0,
            Float64[], Int[], Any[], Any[], Any[], Any[], Any[],
            Vector{Float64}[], Vector{Float64}[], Vector{Float64}[], Vector{Float64}[], Vector{Float64}[], Vector{Float64}[],
            Any[], flush_every, 0, flush_time,
            Vector{Vector{Array{Float64,2}}}(), Vector{Vector{Array{Float64,2}}}(), Vector{Vector{Vector{Float64}}}(), Vector{Vector{Vector{Float64}}}(),
            Vector{Vector{Array{Float64,2}}}(), Vector{Vector{Array{Float64,2}}}(),
            Float64[], Int[], Vector{Vector{Float64}}())
    end
end

import ..BioFlows: write_solution!, close!

function should_save(writer::JLD2Writer, current_time::Float64, current_iteration::Int)
    cfg = writer.config
    # Always save the very first sample (initial condition)
    if writer.current_snapshot == 0
        return true
    end
    time_condition = (current_time - writer.last_save_time) >= cfg.time_interval
    iter_condition = (current_iteration - writer.last_save_iteration) >= cfg.iteration_interval
    if cfg.save_mode == :time_interval
        return time_condition
    elseif cfg.save_mode == :iteration_interval
        return iter_condition
    else
        return time_condition || iter_condition
    end
end

"""
    finalize_stacked!(filepath::String; save_centers=true, save_coeffs=true, save_amr=true)

Post-process a JLD2 file that contains per-snapshot groups into stacked arrays:
- stacked/time, stacked/iteration
- stacked/u, stacked/w, stacked/p
- stacked/coefficients/{Cd,Cl,Fx,Fz,Cd_pressure,Cd_viscous} (if present)
- stacked/rigid_body_centers_xz (if present)
- stacked/amr/refinement_level (if present)
"""
function finalize_stacked!(filepath::String; save_centers::Bool=true, save_coeffs::Bool=true, save_amr::Bool=true)
    fpath = endswith(filepath, ".jld2") ? filepath : string(filepath, ".jld2")
    jldopen(fpath, "a") do f
        # Collect snapshot groups and sort by index
        snap_names = String[]
        for k in keys(f)
            s = String(k)
            if startswith(s, "snapshot_")
                push!(snap_names, s)
            end
        end
        isempty(snap_names) && return
        parse_step(s) = try parse(Int, split(s, "_")[end]) catch; typemax(Int) end
        sort!(snap_names, by=parse_step)

        # Time and iteration vectors
        times = Float64[]; iters = Int[]
        for g in snap_names
            push!(times, f["$g/time"]) ; push!(iters, f["$g/iteration"])
        end
        f["stacked/time"] = times
        f["stacked/iteration"] = iters

        # Flow fields stacked (support center-only mode)
        center_first = haskey(f, "$(snap_names[1])/u_cc")
        u0 = f[center_first ? "$(snap_names[1])/u_cc" : "$(snap_names[1])/u"]
        w0 = f[center_first ? "$(snap_names[1])/w_cc" : "$(snap_names[1])/w"]
        p0 = f["$(snap_names[1])/p"]
        has_v = haskey(f, center_first ? "$(snap_names[1])/v_cc" : "$(snap_names[1])/v")
        Nu = ndims(u0) ; Nw = ndims(w0) ; Np = ndims(p0)
        Nt = length(snap_names)
        if Nu == 2 && Nw == 2 && Np == 2
            U = Array{Float64,3}(undef, size(u0,1), size(u0,2), Nt)
            W = Array{Float64,3}(undef, size(w0,1), size(w0,2), Nt)
            P = Array{Float64,3}(undef, size(p0,1), size(p0,2), Nt)
            for (t, g) in enumerate(snap_names)
                U[:,:,t] = f[center_first ? "$g/u_cc" : "$g/u"]
                W[:,:,t] = f[center_first ? "$g/w_cc" : "$g/w"]
                P[:,:,t] = f["$g/p"]
            end
            f["stacked/$(center_first ? "u_cc" : "u")"] = U
            f["stacked/$(center_first ? "w_cc" : "w")"] = W
            f["stacked/p"] = P
            # Derived: 2D vorticity ω = ∂u/∂z - ∂w/∂x at cell centers
            dx = f["grid/dx"]; dz = f["grid/dz"]
            nx = size(P,1); nz = size(P,2)
            Omega = Array{Float64,3}(undef, nx, nz, Nt)
            for t in 1:Nt
                # Cell-centered velocities if available, else reconstruct from faces
                Ucc = center_first ? view(U, :,:, t) : Array{Float64,2}(undef, nx, nz)
                Wcc = center_first ? view(W, :,:, t) : Array{Float64,2}(undef, nx, nz)
                if !center_first
                    @inbounds for j=1:nz, i=1:nx
                        Ucc[i,j] = 0.5*(U[i,j,t] + U[i+1,j,t])
                        Wcc[i,j] = 0.5*(W[i,j,t] + W[i,j+1,t])
                    end
                end
                # central differences interior
                @inbounds for j=1:nz, i=1:nx
                    du_dz = (Ucc[i, clamp(j+1,1,nz)] - Ucc[i, clamp(j-1,1,nz)])/( (j==1 || j==nz) ? dz : (2dz))
                    dw_dx = (Wcc[clamp(i+1,1,nx), j] - Wcc[clamp(i-1,1,nx), j])/( (i==1 || i==nx) ? dx : (2dx))
                    if (j>1 && j<nz); du_dz = (Ucc[i,j+1]-Ucc[i,j-1])/(2dz); end
                    if (i>1 && i<nx); dw_dx = (Wcc[i+1,j]-Wcc[i-1,j])/(2dx); end
                    Omega[i,j,t] = du_dz - dw_dx
                end
            end
            f["stacked/omega"] = Omega
        elseif Nu == 3 && Nw == 3 && Np == 3
            # 3D single-process path (rare); stack as 4D
            U = Array{Float64,4}(undef, size(u0,1), size(u0,2), size(u0,3), Nt)
            V = has_v ? Array{Float64,4}(undef, size(f[center_first ? "$(snap_names[1])/v_cc" : "$(snap_names[1])/v"],1), size(f[center_first ? "$(snap_names[1])/v_cc" : "$(snap_names[1])/v"],2), size(f[center_first ? "$(snap_names[1])/v_cc" : "$(snap_names[1])/v"],3), Nt) : nothing
            W = Array{Float64,4}(undef, size(w0,1), size(w0,2), size(w0,3), Nt)
            P = Array{Float64,4}(undef, size(p0,1), size(p0,2), size(p0,3), Nt)
            for (t, g) in enumerate(snap_names)
                U[:,:,:,t] = f[center_first ? "$g/u_cc" : "$g/u"]
                if V !== nothing; V[:,:,:,t] = f[center_first ? "$g/v_cc" : "$g/v"]; end
                W[:,:,:,t] = f[center_first ? "$g/w_cc" : "$g/w"]; P[:,:,:,t] = f["$g/p"]
            end
            f["stacked/$(center_first ? "u_cc" : "u")"] = U
            if V !== nothing; f["stacked/$(center_first ? "v_cc" : "v")"] = V; end
            f["stacked/$(center_first ? "w_cc" : "w")"] = W
            f["stacked/p"] = P
            # Derived: 3D vorticity magnitude at cell centers
            if V !== nothing
                dx = f["grid/dx"]; dy = f["grid/dy"]; dz = f["grid/dz"]
                nx = size(P,1); ny = size(P,2); nz = size(P,3)
                Om = Array{Float64,4}(undef, nx, ny, nz, Nt)
                # helper for central diff
                for t in 1:Nt
                    if center_first
                        Ucc = view(U, :,:,:, t); Vcc = view(V, :,:,:, t); Wcc = view(W, :,:,:, t)
                    else
                        Ucc = Array{Float64,3}(undef, nx, ny, nz)
                        Vcc = Array{Float64,3}(undef, nx, ny, nz)
                        Wcc = Array{Float64,3}(undef, nx, ny, nz)
                        @inbounds for k=1:nz, j=1:ny, i=1:nx
                            Ucc[i,j,k] = 0.5*(U[i,j,k,t] + U[i+1,j,k,t])
                            Vcc[i,j,k] = 0.5*(V[i,j,k,t] + V[i,j+1,k,t])
                            Wcc[i,j,k] = 0.5*(W[i,j,k,t] + W[i,j,k+1,t])
                        end
                    end
                    @inbounds for k=1:nz, j=1:ny, i=1:nx
                        dW_dy = (Wcc[i, clamp(j+1,1,ny), k] - Wcc[i, clamp(j-1,1,ny), k]) / ((j==1 || j==ny) ? dy : 2dy)
                        dV_dz = (Vcc[i, j, clamp(k+1,1,nz)] - Vcc[i, j, clamp(k-1,1,nz)]) / ((k==1 || k==nz) ? dz : 2dz)
                        dU_dz = (Ucc[i, j, clamp(k+1,1,nz)] - Ucc[i, j, clamp(k-1,1,nz)]) / ((k==1 || k==nz) ? dz : 2dz)
                        dW_dx = (Wcc[clamp(i+1,1,nx), j, k] - Wcc[clamp(i-1,1,nx), j, k]) / ((i==1 || i==nx) ? dx : 2dx)
                        dV_dx = (Vcc[clamp(i+1,1,nx), j, k] - Vcc[clamp(i-1,1,nx), j, k]) / ((i==1 || i==nx) ? dx : 2dx)
                        dU_dy = (Ucc[i, clamp(j+1,1,ny), k] - Ucc[i, clamp(j-1,1,ny), k]) / ((j==1 || j==ny) ? dy : 2dy)
                        if (j>1 && j<ny); dW_dy = (Wcc[i,j+1,k]-Wcc[i,j-1,k])/(2dy); dU_dy = (Ucc[i,j+1,k]-Ucc[i,j-1,k])/(2dy); end
                        if (k>1 && k<nz); dV_dz = (Vcc[i,j,k+1]-Vcc[i,j,k-1])/(2dz); dU_dz = (Ucc[i,j,k+1]-Ucc[i,j,k-1])/(2dz); end
                        if (i>1 && i<nx); dW_dx = (Wcc[i+1,j,k]-Wcc[i-1,j,k])/(2dx); dV_dx = (Vcc[i+1,j,k]-Vcc[i-1,j,k])/(2dx); end
                        wx = dW_dy - dV_dz
                        wy = dU_dz - dW_dx
                        wz = dV_dx - dU_dy
                        Om[i,j,k,t] = sqrt(wx^2 + wy^2 + wz^2)
                    end
                end
                f["stacked/omega_mag"] = Om
            end
        end

        # Rigid-body centers stacked (if present)
        if save_centers && haskey(f, "$(snap_names[1])/rigid_body_centers_xz")
            c0 = f["$(snap_names[1])/rigid_body_centers_xz"]
            C = Array{Float64,3}(undef, size(c0,1), size(c0,2), Nt)
            for (t, g) in enumerate(snap_names)
                if haskey(f, "$g/rigid_body_centers_xz")
                    C[:,:,t] = f["$g/rigid_body_centers_xz"]
                end
            end
            f["stacked/rigid_body_centers_xz"] = C
        end

        # Coefficients stacked (if present)
        if save_coeffs && haskey(f, "$(snap_names[1])/coefficients/Cd")
            Cd0 = f["$(snap_names[1])/coefficients/Cd"]
            nb = length(Cd0)
            Cd = Array{Float64,2}(undef, nb, Nt)
            Cl = similar(Cd); Fx = similar(Cd); Fz = similar(Cd); Cdp = similar(Cd); Cdv = similar(Cd)
            for (t, g) in enumerate(snap_names)
                if haskey(f, "$g/coefficients/Cd")
                    Cd[:,t]  = f["$g/coefficients/Cd"]
                    Cl[:,t]  = get(f, "$g/coefficients/Cl", zeros(nb))
                    Fx[:,t]  = get(f, "$g/coefficients/Fx", zeros(nb))
                    Fz[:,t]  = get(f, "$g/coefficients/Fz", zeros(nb))
                    Cdp[:,t] = get(f, "$g/coefficients/Cd_pressure", zeros(nb))
                    Cdv[:,t] = get(f, "$g/coefficients/Cd_viscous", zeros(nb))
                end
            end
            f["stacked/coefficients/Cd"] = Cd
            f["stacked/coefficients/Cl"] = Cl
            f["stacked/coefficients/Fx"] = Fx
            f["stacked/coefficients/Fz"] = Fz
            f["stacked/coefficients/Cd_pressure"] = Cdp
            f["stacked/coefficients/Cd_viscous"] = Cdv
        end

        # AMR refinement level stacked
        if save_amr
            amr_names = String[]
            for k in keys(f)
                s = String(k)
                if startswith(s, "amr_snapshot_")
                    push!(amr_names, s)
                end
            end
            if !isempty(amr_names)
                sort!(amr_names, by=parse_step)
                L0 = f["$(amr_names[1])/refinement_level"]
                if ndims(L0) == 2
                    nx, nz = size(L0); NtA = length(amr_names)
                    L = Array{Int,3}(undef, nx, nz, NtA)
                    for (t, g) in enumerate(amr_names)
                        L[:,:,t] = f["$g/refinement_level"]
                    end
                    f["stacked/amr/refinement_level"] = L
                else
                    nx, ny, nz = size(L0); NtA = length(amr_names)
                    L = Array{Int,4}(undef, nx, ny, nz, NtA)
                    for (t, g) in enumerate(amr_names)
                        L[:,:,:,t] = f["$g/refinement_level"]
                    end
                    f["stacked/amr/refinement_level"] = L
                end
            end
        end

        # Flexible-body data stacked (if present)
        # Stacks per-body arrays over time: X (n_points,2,Nt), force (n_points,2,Nt),
        # tension (n_points,Nt), curvature (n_points,Nt), and optionally velocity/acceleration.
        if haskey(f, "$(snap_names[1])/flexible_bodies_count")
            nflex = Int(f["$(snap_names[1])/flexible_bodies_count"])::Int
            for bi in 1:nflex
                # Check first snapshot has this body
                bfirst = "$(snap_names[1])/flexible/body_$(bi)"
                if !haskey(f, bfirst * "/X")
                    continue
                end
                # Read static info from first snapshot
                npts = Int(f[bfirst * "/n_points"])::Int
                svec = f[bfirst * "/s"]
                len  = f[bfirst * "/length"]
                # Allocate stacks
                Xs = Array{Float64,3}(undef, npts, 2, Nt)
                Fs = Array{Float64,3}(undef, npts, 2, Nt)
                Ts = Array{Float64,2}(undef, npts, Nt)
                Ks = Array{Float64,2}(undef, npts, Nt)
                has_vel = haskey(f, bfirst * "/velocity")
                has_acc = haskey(f, bfirst * "/acceleration")
                Vs = has_vel ? Array{Float64,3}(undef, npts, 2, Nt) : nothing
                As = has_acc ? Array{Float64,3}(undef, npts, 2, Nt) : nothing
                # Fill stacks
                for (t, g) in enumerate(snap_names)
                    bgrp = "$g/flexible/body_$(bi)"
                    if !haskey(f, bgrp * "/X")
                        # Missing body in this snapshot; fill zeros
                        Xs[:,:,t] .= 0; Fs[:,:,t] .= 0; Ts[:,t] .= 0; Ks[:,t] .= 0
                        if Vs !== nothing; Vs[:,:,t] .= 0; end
                        if As !== nothing; As[:,:,t] .= 0; end
                        continue
                    end
                    Xs[:,:,t] = f[bgrp * "/X"]
                    Fs[:,:,t] = f[bgrp * "/force"]
                    Ts[:,t]   = f[bgrp * "/tension"]
                    Ks[:,t]   = f[bgrp * "/curvature"]
                    if Vs !== nothing && haskey(f, bgrp * "/velocity")
                        Vs[:,:,t] = f[bgrp * "/velocity"]
                    end
                    if As !== nothing && haskey(f, bgrp * "/acceleration")
                        As[:,:,t] = f[bgrp * "/acceleration"]
                    end
                end
                # Write stacked outputs for this body under stacked/flexible/body_i
                outb = "stacked/flexible/body_$(bi)"
                f[outb * "/length"] = len
                f[outb * "/n_points"] = npts
                f[outb * "/s"] = svec
                f[outb * "/X"] = Xs
                f[outb * "/force"] = Fs
                f[outb * "/tension"] = Ts
                f[outb * "/curvature"] = Ks
                if Vs !== nothing; f[outb * "/velocity"] = Vs; end
                if As !== nothing; f[outb * "/acceleration"] = As; end
            end
        end

        # Convenience meshgrid at cell centers for plotting
        # 2D: stacked/grid/Xcc (nx,nz), stacked/grid/Zcc (nx,nz)
        # 3D: stacked/grid/Xcc (nx,ny,nz), Ycc, Zcc
        has_x = haskey(f, "grid/x"); has_z = haskey(f, "grid/z")
        has_y = haskey(f, "grid/y")
        if has_x && has_z
            x = f["grid/x"]; z = f["grid/z"]
            nx = length(x); nz = length(z)
            if !haskey(f, "stacked/grid/Xcc") || !haskey(f, "stacked/grid/Zcc")
                if !has_y
                    Xcc = Array{Float64,2}(undef, nx, nz)
                    Zcc = Array{Float64,2}(undef, nx, nz)
                    @inbounds for j=1:nz, i=1:nx
                        Xcc[i,j] = x[i]
                        Zcc[i,j] = z[j]
                    end
                    f["stacked/grid/Xcc"] = Xcc
                    f["stacked/grid/Zcc"] = Zcc
                else
                    y = f["grid/y"]; ny = length(y)
                    Xcc = Array{Float64,3}(undef, nx, ny, nz)
                    Ycc = Array{Float64,3}(undef, nx, ny, nz)
                    Zcc = Array{Float64,3}(undef, nx, ny, nz)
                    @inbounds for k=1:nz, j=1:ny, i=1:nx
                        Xcc[i,j,k] = x[i]
                        Ycc[i,j,k] = y[j]
                        Zcc[i,j,k] = z[k]
                    end
                    f["stacked/grid/Xcc"] = Xcc
                    f["stacked/grid/Ycc"] = Ycc
                    f["stacked/grid/Zcc"] = Zcc
                end
            end
        end
    end
    println("Finalized stacked arrays → $fpath")
    return true
end

export finalize_stacked!

"""
    write_solution!(writer::JLD2Writer, state::MPISolutionState2D, ...)

Gather 2D MPI local fields to root and write a global JLD2 snapshot.
"""
function write_solution!(writer::JLD2Writer,
                        state::MPISolutionState2D,
                        bodies::Union{Nothing, RigidBodyCollection, FlexibleBodyCollection},
                        grid::StaggeredGrid,
                        fluid::FluidProperties,
                        current_time::Float64,
                        current_iteration::Int; kwargs...)
    decomp = state.decomp
    comm = decomp.comm
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # Local interior indices
    ils, ile = decomp.i_local_start, decomp.i_local_end
    jls, jle = decomp.j_local_start, decomp.j_local_end
    is, ie = decomp.i_start, decomp.i_end
    js, je = decomp.j_start, decomp.j_end
    nxg, nzg = decomp.nx_global, decomp.nz_global

    # Include domain boundary faces (staggered)
    u_i_hi = ile + (ie == nxg ? 1 : 0)
    v_j_hi = jle + (je == nzg ? 1 : 0)

    u_blk = @view state.u[ils:u_i_hi, jls:jle]
    v_blk = @view state.w[ils:ile, jls:v_j_hi]
    p_blk = @view state.p[ils:ile, jls:jle]

    if rank == 0
        u_glob = zeros(Float64, writer.grid.nx + 1, writer.grid.nz)
        v_glob = zeros(Float64, writer.grid.nx, writer.grid.nz + 1)
        p_glob = zeros(Float64, writer.grid.nx, writer.grid.nz)
        # Place rank 0
        u_glob[is:ie + (ie == nxg ? 1 : 0), js:je] .= u_blk
        v_glob[is:ie, js:je + (je == nzg ? 1 : 0)] .= v_blk
        p_glob[is:ie, js:je] .= p_blk
        # Receive others
        for src in 1:size-1
            hdr = Array{Int}(undef, 4)
            MPI.Recv!(hdr, src, 9100, comm)
            isrc, iesrc, jsrc, jesrc = hdr
            u_count_i = iesrc - isrc + 1 + (iesrc == nxg ? 1 : 0)
            v_count_j = jesrc - jsrc + 1 + (jesrc == nzg ? 1 : 0)
            u_recv = Array{Float64}(undef, u_count_i, jesrc - jsrc + 1)
            v_recv = Array{Float64}(undef, iesrc - isrc + 1, v_count_j)
            p_recv = Array{Float64}(undef, iesrc - isrc + 1, jesrc - jsrc + 1)
            MPI.Recv!(u_recv, src, 9101, comm)
            MPI.Recv!(v_recv, src, 9102, comm)
            MPI.Recv!(p_recv, src, 9103, comm)
            u_glob[isrc:iesrc + (iesrc == nxg ? 1 : 0), jsrc:jesrc] .= u_recv
            v_glob[isrc:iesrc, jsrc:jesrc + (jesrc == nzg ? 1 : 0)] .= v_recv
            p_glob[isrc:iesrc, jsrc:jesrc] .= p_recv
        end
        # Build global state and write
        gstate = SolutionState2D(writer.grid.nx, writer.grid.nz)
        gstate.u .= u_glob
        gstate.w .= v_glob
        gstate.p .= p_glob
        gstate.t = current_time
        gstate.step = current_iteration
        return write_solution!(writer, gstate, bodies, writer.grid, fluid, current_time, current_iteration; kwargs...)
    else
        hdr = Int[is, ie, js, je]
        MPI.Send(hdr, 0, 9100, comm)
        MPI.Send(Array(u_blk), 0, 9101, comm)
        MPI.Send(Array(v_blk), 0, 9102, comm)
        MPI.Send(Array(p_blk), 0, 9103, comm)
        return true
    end
end

"""
    write_solution!(writer::JLD2Writer, state::MPISolutionState3D, ...)
"""
function write_solution!(writer::JLD2Writer,
                        state::MPISolutionState3D,
                        bodies::Union{Nothing, RigidBodyCollection, FlexibleBodyCollection},
                        grid::StaggeredGrid,
                        fluid::FluidProperties,
                        current_time::Float64,
                        current_iteration::Int; kwargs...)
    decomp = state.decomp
    comm = decomp.comm
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    # Global ranges
    is, ie = decomp.i_start, decomp.i_end
    js, je = decomp.j_start, decomp.j_end
    ks, ke = decomp.k_start, decomp.k_end
    nxg, nyg, nzg = decomp.nx_global, decomp.ny_global, decomp.nz_global
    nx_loc, ny_loc, nz_loc = size(state.p)
    u_i_hi_loc = nx_loc + (ie == nxg ? 1 : 0)
    v_j_hi_loc = ny_loc + (je == nyg ? 1 : 0)
    w_k_hi_loc = nz_loc + (ke == nzg ? 1 : 0)
    u_blk = @view state.u[1:u_i_hi_loc, 1:ny_loc, 1:nz_loc]
    v_blk = @view state.v[1:nx_loc, 1:v_j_hi_loc, 1:nz_loc]
    w_blk = @view state.w[1:nx_loc, 1:ny_loc, 1:w_k_hi_loc]
    p_blk = @view state.p[1:nx_loc, 1:ny_loc, 1:nz_loc]
    if rank == 0
        u_glob = zeros(Float64, writer.grid.nx + 1, writer.grid.ny, writer.grid.nz)
        v_glob = zeros(Float64, writer.grid.nx, writer.grid.ny + 1, writer.grid.nz)
        w_glob = zeros(Float64, writer.grid.nx, writer.grid.ny, writer.grid.nz + 1)
        p_glob = zeros(Float64, writer.grid.nx, writer.grid.ny, writer.grid.nz)
        u_glob[is:ie + (ie == nxg ? 1 : 0), js:je, ks:ke] .= u_blk
        v_glob[is:ie, js:je + (je == nyg ? 1 : 0), ks:ke] .= v_blk
        w_glob[is:ie, js:je, ks:ke + (ke == nzg ? 1 : 0)] .= w_blk
        p_glob[is:ie, js:je, ks:ke] .= p_blk
        for src in 1:size-1
            hdr = Array{Int}(undef, 6)
            MPI.Recv!(hdr, src, 9200, comm)
            isrc, iesrc, jsrc, jesrc, ksrc, kesrc = hdr
            u_count_i = iesrc - isrc + 1 + (iesrc == nxg ? 1 : 0)
            v_count_j = jesrc - jsrc + 1 + (jesrc == nyg ? 1 : 0)
            w_count_k = kesrc - ksrc + 1 + (kesrc == nzg ? 1 : 0)
            u_recv = Array{Float64}(undef, u_count_i, jesrc - jsrc + 1, kesrc - ksrc + 1)
            v_recv = Array{Float64}(undef, iesrc - isrc + 1, v_count_j, kesrc - ksrc + 1)
            w_recv = Array{Float64}(undef, iesrc - isrc + 1, jesrc - jsrc + 1, w_count_k)
            p_recv = Array{Float64}(undef, iesrc - isrc + 1, jesrc - jsrc + 1, kesrc - ksrc + 1)
            MPI.Recv!(u_recv, src, 9201, comm)
            MPI.Recv!(v_recv, src, 9202, comm)
            MPI.Recv!(w_recv, src, 9203, comm)
            MPI.Recv!(p_recv, src, 9204, comm)
            u_glob[isrc:iesrc + (iesrc == nxg ? 1 : 0), jsrc:jesrc, ksrc:kesrc] .= u_recv
            v_glob[isrc:iesrc, jsrc:jesrc + (jesrc == nyg ? 1 : 0), ksrc:kesrc] .= v_recv
            w_glob[isrc:iesrc, jsrc:jesrc, ksrc:kesrc + (kesrc == nzg ? 1 : 0)] .= w_recv
            p_glob[isrc:iesrc, jsrc:jesrc, ksrc:kesrc] .= p_recv
        end
        gstate = SolutionState3D(writer.grid.nx, writer.grid.ny, writer.grid.nz)
        gstate.u .= u_glob
        gstate.v .= v_glob
        gstate.w .= w_glob
        gstate.p .= p_glob
        gstate.t = current_time
        gstate.step = current_iteration
        return write_solution!(writer, gstate, bodies, writer.grid, fluid, current_time, current_iteration; kwargs...)
    else
        hdr = Int[is, ie, js, je, ks, ke]
        MPI.Send(hdr, 0, 9200, comm)
        MPI.Send(Array(u_blk), 0, 9201, comm)
        MPI.Send(Array(v_blk), 0, 9202, comm)
        MPI.Send(Array(w_blk), 0, 9203, comm)
        MPI.Send(Array(p_blk), 0, 9204, comm)
        return true
    end
end

"""
    save_amr_to_output!(writer::JLD2Writer, refined_grid::RefinedGrid, state::SolutionState, step::Int, time::Float64; bodies=nothing)
"""
function save_amr_to_output!(writer::JLD2Writer,
                            refined_grid::RefinedGrid,
                            state::SolutionState,
                            step::Int,
                            time::Float64; bodies=nothing)
    output_state, metadata = BioFlows.prepare_amr_for_netcdf_output(refined_grid, state, "amr_flow", step, time)
    write_solution!(writer, output_state, bodies, writer.grid, BioFlows.FluidProperties(0.0, BioFlows.ConstantDensity(1.0), 1.0), time, step)
    # Also write refinement map
    fpath = endswith(writer.filepath, ".jld2") ? writer.filepath : string(writer.filepath, ".jld2")
    jldopen(fpath, "a") do f
        grp = "amr_snapshot_$(step)"
        if refined_grid.base_grid.grid_type == BioFlows.TwoDimensional
            nx, nz = refined_grid.base_grid.nx, refined_grid.base_grid.nz
            lvl = zeros(Int, nx, nz)
            for ((i,j), val) in refined_grid.refined_cells_2d
                lvl[i,j] = val
            end
            f["$grp/refinement_level"] = lvl
        else
            nx, ny, nz = refined_grid.base_grid.nx, refined_grid.base_grid.ny, refined_grid.base_grid.nz
            lvl = zeros(Int, nx, ny, nz)
            for ((i,j,k), val) in refined_grid.refined_cells_3d
                lvl[i,j,k] = val
            end
            f["$grp/refinement_level"] = lvl
        end
        for (k,v) in metadata
            f["$grp/metadata/$k"] = v
        end
    end
    return true
end

function close!(writer::JLD2Writer)
    # Flush any remaining buffer as the final block
    if !isempty(writer.p_slices)
        write_block!(writer)
    end
    # Assemble stacked arrays from all blocks
    fpath = endswith(writer.filepath, ".jld2") ? writer.filepath : string(writer.filepath, ".jld2")
    try
        jldopen(fpath, "a") do f
        # Collect blocks sorted by index (handle nested group under "blocks/")
        block_keys = String[]
        if haskey(f, "blocks")
            for b in keys(f["blocks"])  # names like "block_0001"
                push!(block_keys, "blocks/" * String(b))
            end
            sort!(block_keys)
        end
            # Gather times/iterations
            all_times = Float64[]; all_iters = Int[]
            for bk in block_keys
                append!(all_times, f[bk * "/time"])
                append!(all_iters, f[bk * "/iteration"])
            end
            f["stacked/time"] = all_times
            f["stacked/iteration"] = all_iters
            # Fields
            if writer.grid.grid_type == BioFlows.TwoDimensional
                U = nothing; W = nothing; P = nothing
                for bk in block_keys
                    Ub = f[bk * "/u_cc"]; Wb = f[bk * "/w_cc"]; Pb = f[bk * "/p"]
                    if U === nothing
                        U = Ub; W = Wb; P = Pb
                    else
                        U = cat(U, Ub; dims=3)
                        W = cat(W, Wb; dims=3)
                        P = cat(P, Pb; dims=3)
                    end
                end
                if U !== nothing
                    f["stacked/u_cc"] = U
                    f["stacked/w_cc"] = W
                    f["stacked/p"] = P
                    # Derived vorticity
                    nx, nz, Nt = size(P)
                    dx = writer.grid.dx; dz = writer.grid.dz
                    Omega = Array{Float64,3}(undef, nx, nz, Nt)
                    @inbounds for t=1:Nt
                        Ucc = view(U, :,:, t); Wcc = view(W, :,:, t)
                        for j=1:nz, i=1:nx
                            du_dz = (Ucc[i, clamp(j+1,1,nz)] - Ucc[i, clamp(j-1,1,nz)]) / ((j==1 || j==nz) ? dz : 2dz)
                            dw_dx = (Wcc[clamp(i+1,1,nx), j] - Wcc[clamp(i-1,1,nx), j]) / ((i==1 || i==nx) ? dx : 2dx)
                            if (j>1 && j<nz); du_dz = (Ucc[i,j+1]-Ucc[i,j-1])/(2dz); end
                            if (i>1 && i<nx); dw_dx = (Wcc[i+1,j]-Wcc[i-1,j])/(2dx); end
                            Omega[i,j,t] = du_dz - dw_dx
                        end
                    end
                    f["stacked/omega"] = Omega
                end
            else
                U = nothing; V = nothing; W = nothing; P = nothing
                for bk in block_keys
                    Ub = f[bk * "/u_cc"]; Vb = f[bk * "/v_cc"]; Wb = f[bk * "/w_cc"]; Pb = f[bk * "/p"]
                    if U === nothing
                        U = Ub; V = Vb; W = Wb; P = Pb
                    else
                        U = cat(U, Ub; dims=4)
                        V = cat(V, Vb; dims=4)
                        W = cat(W, Wb; dims=4)
                        P = cat(P, Pb; dims=4)
                    end
                end
                if U !== nothing
                    f["stacked/u_cc"] = U
                    f["stacked/v_cc"] = V
                    f["stacked/w_cc"] = W
                    f["stacked/p"] = P
                    # Derived vorticity magnitude
                    nx, ny, nz, Nt = size(P)
                    dx = writer.grid.dx; dy = writer.grid.dy; dz = writer.grid.dz
                    Om = Array{Float64,4}(undef, nx, ny, nz, Nt)
                    @inbounds for t=1:Nt
                        Ucc = view(U, :,:,:, t); Vcc = view(V, :,:,:, t); Wcc = view(W, :,:,:, t)
                        for k=1:nz, j=1:ny, i=1:nx
                            dW_dy = (Wcc[i, clamp(j+1,1,ny), k] - Wcc[i, clamp(j-1,1,ny), k]) / ((j==1 || j==ny) ? dy : 2dy)
                            dV_dz = (Vcc[i, j, clamp(k+1,1,nz)] - Vcc[i, j, clamp(k-1,1,nz)]) / ((k==1 || k==nz) ? dz : 2dz)
                            dU_dz = (Ucc[i, j, clamp(k+1,1,nz)] - Ucc[i, j, clamp(k-1,1,nz)]) / ((k==1 || k==nz) ? dz : 2dz)
                            dW_dx = (Wcc[clamp(i+1,1,nx), j, k] - Wcc[clamp(i-1,1,nx), j, k]) / ((i==1 || i==nx) ? dx : 2dx)
                            dV_dx = (Vcc[clamp(i+1,1,nx), j, k] - Vcc[clamp(i-1,1,nx), j, k]) / ((i==1 || i==nx) ? dx : 2dx)
                            dU_dy = (Ucc[i, clamp(j+1,1,ny), k] - Ucc[i, clamp(j-1,1,ny), k]) / ((j==1 || j==ny) ? dy : 2dy)
                            if (j>1 && j<ny); dW_dy = (Wcc[i,j+1,k]-Wcc[i,j-1,k])/(2dy); dU_dy = (Ucc[i,j+1,k]-Ucc[i,j-1,k])/(2dy); end
                            if (k>1 && k<nz); dV_dz = (Vcc[i,j,k+1]-Vcc[i,j,k-1])/(2dz); dU_dz = (Ucc[i,j,k+1]-Ucc[i,j,k-1])/(2dz); end
                            if (i>1 && i<nx); dW_dx = (Wcc[i+1,j,k]-Wcc[i-1,j,k])/(2dx); dV_dx = (Vcc[i+1,j,k]-Vcc[i-1,j,k])/(2dx); end
                            wx = dW_dy - dV_dz
                            wy = dU_dz - dW_dx
                            wz = dV_dx - dU_dy
                            Om[i,j,k,t] = sqrt(wx^2 + wy^2 + wz^2)
                        end
                    end
                    f["stacked/omega_mag"] = Om
                end
            end
            # Stacked centers
            center_blocks = [bk for bk in block_keys if haskey(f, bk * "/rigid_body_centers_xz")]
            if !isempty(center_blocks)
                C = nothing
                for bk in center_blocks
                    Cb = f[bk * "/rigid_body_centers_xz"]
                    C = (C === nothing) ? Cb : cat(C, Cb; dims=3)
                end
                if C !== nothing; f["stacked/rigid_body_centers_xz"] = C; end
            end
            # Stacked coefficients
            coeff_blocks = [bk for bk in block_keys if haskey(f, bk * "/coefficients/Cd")]
            if !isempty(coeff_blocks)
                Cd = nothing; Cl = nothing; Fx = nothing; Fz = nothing; Cdp = nothing; Cdv = nothing
                for bk in coeff_blocks
                    Cd_b = f[bk * "/coefficients/Cd"];  Cl_b = f[bk * "/coefficients/Cl"]
                    Fx_b = f[bk * "/coefficients/Fx"]; Fz_b = f[bk * "/coefficients/Fz"]
                    Cdp_b = f[bk * "/coefficients/Cd_pressure"]; Cdv_b = f[bk * "/coefficients/Cd_viscous"]
                    Cd = (Cd === nothing) ? Cd_b : hcat(Cd, Cd_b)
                    Cl = (Cl === nothing) ? Cl_b : hcat(Cl, Cl_b)
                    Fx = (Fx === nothing) ? Fx_b : hcat(Fx, Fx_b)
                    Fz = (Fz === nothing) ? Fz_b : hcat(Fz, Fz_b)
                    Cdp = (Cdp === nothing) ? Cdp_b : hcat(Cdp, Cdp_b)
                    Cdv = (Cdv === nothing) ? Cdv_b : hcat(Cdv, Cdv_b)
                end
                f["stacked/coefficients/Cd"] = Cd
                f["stacked/coefficients/Cl"] = Cl
                f["stacked/coefficients/Fx"] = Fx
                f["stacked/coefficients/Fz"] = Fz
                f["stacked/coefficients/Cd_pressure"] = Cdp
                f["stacked/coefficients/Cd_viscous"] = Cdv
            end
            # Convenience cc meshgrid
            has_x = haskey(f, "grid/x"); has_z = haskey(f, "grid/z"); has_y = haskey(f, "grid/y")
            if has_x && has_z
                x = f["grid/x"]; z = f["grid/z"]; nx = length(x); nz = length(z)
                if !haskey(f, "stacked/grid/Xcc") || !haskey(f, "stacked/grid/Zcc")
                    if !has_y
                        Xcc = Array{Float64,2}(undef, nx, nz)
                        Zcc = Array{Float64,2}(undef, nx, nz)
                        @inbounds for j=1:nz, i=1:nx
                            Xcc[i,j] = x[i]; Zcc[i,j] = z[j]
                        end
                        f["stacked/grid/Xcc"] = Xcc
                        f["stacked/grid/Zcc"] = Zcc
                    else
                        y = f["grid/y"]; ny = length(y)
                        Xcc = Array{Float64,3}(undef, nx, ny, nz)
                        Ycc = Array{Float64,3}(undef, nx, ny, nz)
                        Zcc = Array{Float64,3}(undef, nx, ny, nz)
                        @inbounds for k=1:nz, j=1:ny, i=1:nx
                            Xcc[i,j,k] = x[i]; Ycc[i,j,k] = y[j]; Zcc[i,j,k] = z[k]
                        end
                        f["stacked/grid/Xcc"] = Xcc
                        f["stacked/grid/Ycc"] = Ycc
                        f["stacked/grid/Zcc"] = Zcc
                    end
                end
            end
            # Optionally remove block data to keep file compact
            keep_blocks = lowercase(get(ENV, "BIOFLOWS_JLD2_KEEP_BLOCKS", "")) in ("1","true","yes")
            if !keep_blocks && !isempty(block_keys)
                try
                    delete!(f, "blocks")
                catch e
                    @warn "Could not remove intermediate JLD2 blocks: $e"
                end
            end
            # Optionally remove block data to keep file compact
            keep_blocks = lowercase(get(ENV, "BIOFLOWS_JLD2_KEEP_BLOCKS", "")) in ("1","true","yes")
            if !keep_blocks && !isempty(block_keys)
                try
                    delete!(f, "blocks")
                catch e
                    @warn "Could not remove intermediate JLD2 blocks: $e"
                end
            end
        end
    catch e
        @warn "Failed to write stacked JLD2 output: $e"
    end
    return nothing
end

function write_solution!(writer::JLD2Writer,
                        state::SolutionState,
                        bodies::Union{Nothing, RigidBodyCollection, FlexibleBodyCollection},
                        grid::StaggeredGrid,
                        fluid::FluidProperties,
                        current_time::Float64,
                        current_iteration::Int; kwargs...)
    if !should_save(writer, current_time, current_iteration)
        return false
    end
    writer.current_snapshot += 1
    writer.last_save_time = current_time
    writer.last_save_iteration = current_iteration
    push!(writer.times, current_time)
    push!(writer.iterations, current_iteration)
    # Buffer cell-centered velocities and pressure
    if grid.grid_type == BioFlows.TwoDimensional
        ucc, wcc = interpolate_to_cell_centers_xz(state.u, state.w, grid)
        push!(writer.u_slices, ucc)
        push!(writer.w_slices, wcc)
    else
        ucc, vcc, wcc = interpolate_to_cell_centers(state.u, state.v, state.w, grid)
        push!(writer.u_slices, ucc)
        push!(writer.v_slices, vcc)
        push!(writer.w_slices, wcc)
    end
    push!(writer.p_slices, Array(state.p))
    # Rigid-body centers and coefficients (buffered)
    if bodies !== nothing && writer.config.save_body_positions && bodies isa RigidBodyCollection
        centers = reduce(hcat, [[b.center[1], length(b.center)>2 ? b.center[3] : b.center[2]] for b in bodies.bodies])
        push!(writer.centers, centers)
    end
    # Flexible bodies: buffer per-sample per-body arrays
    if bodies !== nothing && bodies isa FlexibleBodyCollection
        nb = bodies.n_bodies
        # Initialize meta on first encounter
        if isempty(writer.flex_lengths)
            for b in bodies.bodies
                push!(writer.flex_lengths, b.length)
                push!(writer.flex_npoints, b.n_points)
                push!(writer.flex_svecs, copy(b.s))
            end
        end
        # Build per-sample vectors
        Xs = Vector{Array{Float64,2}}(undef, nb)
        Fs = Vector{Array{Float64,2}}(undef, nb)
        Ts = Vector{Vector{Float64}}(undef, nb)
        Ks = Vector{Vector{Float64}}(undef, nb)
        save_vels = lowercase(get(ENV, "BIOFLOWS_SAVE_FLEX_VELS", "")) in ("1","true","yes")
        dt_kw = haskey(kwargs, :dt) ? kwargs[:dt] : nothing
        Vs = Vector{Array{Float64,2}}(undef, nb)
        As = Vector{Array{Float64,2}}(undef, nb)
        for (i, b) in enumerate(bodies.bodies)
            Xs[i] = copy(b.X)
            Fs[i] = copy(b.force)
            Ts[i] = copy(b.tension)
            Ks[i] = copy(b.curvature)
            if save_vels && dt_kw !== nothing
                try
                    Vs[i] = (b.X .- b.X_old) ./ dt_kw
                    As[i] = (b.X .- 2 .* b.X_old .+ b.X_prev) ./ (dt_kw^2)
                catch
                    Vs[i] = zeros(size(b.X)); As[i] = zeros(size(b.X))
                end
            else
                Vs[i] = Array{Float64,2}(undef, 0, 0)
                As[i] = Array{Float64,2}(undef, 0, 0)
            end
        end
        push!(writer.flex_X, Xs)
        push!(writer.flex_F, Fs)
        push!(writer.flex_T, Ts)
        push!(writer.flex_K, Ks)
        push!(writer.flex_V, Vs)
        push!(writer.flex_A, As)
    end
    if bodies !== nothing && writer.config.save_force_coefficients && bodies isa RigidBodyCollection
        n = bodies.n_bodies
        Cd = zeros(n); Cl = zeros(n); Fx = zeros(n); Fz = zeros(n)
        Cd_p = zeros(n); Cd_v = zeros(n)
        for (i, body) in enumerate(bodies.bodies)
            coeffs = BioFlows.compute_drag_lift_coefficients(body, grid, state, fluid;
                reference_velocity=writer.config.reference_velocity,
                flow_direction=writer.config.flow_direction)
            Cd[i] = coeffs.Cd; Cl[i] = coeffs.Cl; Fx[i] = coeffs.Fx; Fz[i] = coeffs.Fz
            Cd_p[i] = coeffs.Cd_pressure; Cd_v[i] = coeffs.Cd_viscous
        end
        push!(writer.coeff_Cd, Cd)
        push!(writer.coeff_Cl, Cl)
        push!(writer.coeff_Fx, Fx)
        push!(writer.coeff_Fz, Fz)
        push!(writer.coeff_Cdp, Cd_p)
        push!(writer.coeff_Cdv, Cd_v)
    end
    # Periodic flush
    if writer.flush_every > 0 && length(writer.p_slices) >= writer.flush_every
        write_block!(writer)
    end
    # Time-based flush window (simulation time)
    if writer.flush_time > 0 && length(writer.times) > 0
        if (writer.times[end] - writer.times[1]) >= writer.flush_time
            write_block!(writer)
        end
    end
    return true
end

"""
    write_block!(writer)

Flush the current in-memory buffers as a block under `blocks/block_XXXX` in the JLD2 file,
then clear the buffers.
"""
function write_block!(writer::JLD2Writer)
    Nt = length(writer.p_slices)
    Nt == 0 && return false
    writer.block_index += 1
    block_name = @sprintf("blocks/block_%04d", writer.block_index)
    fpath = endswith(writer.filepath, ".jld2") ? writer.filepath : string(writer.filepath, ".jld2")
    jldopen(fpath, "a") do f
        # Grid metadata (once)
        if !haskey(f, "grid/nx")
            f["grid/nx"] = writer.grid.nx
            f["grid/ny"] = writer.grid.grid_type == BioFlows.ThreeDimensional ? writer.grid.ny : 0
            f["grid/nz"] = writer.grid.nz
            f["grid/dx"] = writer.grid.dx
            if writer.grid.grid_type == BioFlows.ThreeDimensional
                f["grid/dy"] = writer.grid.dy
            end
            f["grid/dz"] = writer.grid.dz
            f["grid/type"] = writer.grid.grid_type == BioFlows.ThreeDimensional ? "3D" : "2D"
            try f["grid/Lx"] = getfield(writer.grid, :Lx) catch; end
            try f["grid/Ly"] = getfield(writer.grid, :Ly) catch; end
            try f["grid/Lz"] = getfield(writer.grid, :Lz) catch; end
            try
                org = getfield(writer.grid, :origin)
                if org !== nothing
                    f["grid/origin"] = org
                end
            catch; end
            try f["grid/x"] = writer.grid.x catch; end
            if writer.grid.grid_type == BioFlows.ThreeDimensional
                try f["grid/y"] = writer.grid.y catch; end
            end
            try f["grid/z"] = writer.grid.z catch; end
        end
        # Write block time and iteration
        f[block_name * "/time"] = writer.times
        f[block_name * "/iteration"] = writer.iterations
        # Stack and write fields for this block
        if writer.grid.grid_type == BioFlows.TwoDimensional
            nx, nz = size(writer.p_slices[1])
            U = Array{Float64,3}(undef, nx, nz, Nt)
            W = Array{Float64,3}(undef, nx, nz, Nt)
            P = Array{Float64,3}(undef, nx, nz, Nt)
            @inbounds for t=1:Nt
                U[:,:,t] = writer.u_slices[t]
                W[:,:,t] = writer.w_slices[t]
                P[:,:,t] = writer.p_slices[t]
            end
            f[block_name * "/u_cc"] = U
            f[block_name * "/w_cc"] = W
            f[block_name * "/p"] = P
            # Flexible bodies (if any buffered in this block)
            if !isempty(writer.flex_X)
                nb = length(writer.flex_X[1])
                Ntloc = length(writer.flex_X)
                for bi in 1:nb
                    # Determine n_points for this body
                    npts = writer.flex_npoints[bi]
                    Xstk = Array{Float64,3}(undef, npts, 2, Ntloc)
                    Fstk = Array{Float64,3}(undef, npts, 2, Ntloc)
                    Tstk = Array{Float64,2}(undef, npts, Ntloc)
                    Kstk = Array{Float64,2}(undef, npts, Ntloc)
                    hasV = size(writer.flex_V[1][bi],1) > 0
                    Vstk = hasV ? Array{Float64,3}(undef, npts, 2, Ntloc) : nothing
                    Astk = hasV ? Array{Float64,3}(undef, npts, 2, Ntloc) : nothing
                    for t in 1:Ntloc
                        Xstk[:,:,t] = writer.flex_X[t][bi]
                        Fstk[:,:,t] = writer.flex_F[t][bi]
                        Tstk[:,t]   = writer.flex_T[t][bi]
                        Kstk[:,t]   = writer.flex_K[t][bi]
                        if hasV
                            Vstk[:,:,t] = writer.flex_V[t][bi]
                            Astk[:,:,t] = writer.flex_A[t][bi]
                        end
                    end
                    bgrp = block_name * "/flexible/body_" * string(bi)
                    f[bgrp * "/length"] = writer.flex_lengths[bi]
                    f[bgrp * "/n_points"] = writer.flex_npoints[bi]
                    f[bgrp * "/s"] = writer.flex_svecs[bi]
                    f[bgrp * "/X"] = Xstk
                    f[bgrp * "/force"] = Fstk
                    f[bgrp * "/tension"] = Tstk
                    f[bgrp * "/curvature"] = Kstk
                    if hasV
                        f[bgrp * "/velocity"] = Vstk
                        f[bgrp * "/acceleration"] = Astk
                    end
                end
            end
        else
            nx, ny, nz = size(writer.p_slices[1])
            U = Array{Float64,4}(undef, nx, ny, nz, Nt)
            V = Array{Float64,4}(undef, nx, ny, nz, Nt)
            W = Array{Float64,4}(undef, nx, ny, nz, Nt)
            P = Array{Float64,4}(undef, nx, ny, nz, Nt)
            @inbounds for t=1:Nt
                U[:,:,:,t] = writer.u_slices[t]
                V[:,:,:,t] = writer.v_slices[t]
                W[:,:,:,t] = writer.w_slices[t]
                P[:,:,:,t] = writer.p_slices[t]
            end
            f[block_name * "/u_cc"] = U
            f[block_name * "/v_cc"] = V
            f[block_name * "/w_cc"] = W
            f[block_name * "/p"] = P
            # Flexible bodies (3D variant could be 2D filaments; save same per-point arrays)
            if !isempty(writer.flex_X)
                nb = length(writer.flex_X[1])
                Ntloc = length(writer.flex_X)
                for bi in 1:nb
                    npts = writer.flex_npoints[bi]
                    Xstk = Array{Float64,3}(undef, npts, 2, Ntloc)
                    Fstk = Array{Float64,3}(undef, npts, 2, Ntloc)
                    Tstk = Array{Float64,2}(undef, npts, Ntloc)
                    Kstk = Array{Float64,2}(undef, npts, Ntloc)
                    hasV = size(writer.flex_V[1][bi],1) > 0
                    Vstk = hasV ? Array{Float64,3}(undef, npts, 2, Ntloc) : nothing
                    Astk = hasV ? Array{Float64,3}(undef, npts, 2, Ntloc) : nothing
                    for t in 1:Ntloc
                        Xstk[:,:,t] = writer.flex_X[t][bi]
                        Fstk[:,:,t] = writer.flex_F[t][bi]
                        Tstk[:,t]   = writer.flex_T[t][bi]
                        Kstk[:,t]   = writer.flex_K[t][bi]
                        if hasV
                            Vstk[:,:,t] = writer.flex_V[t][bi]
                            Astk[:,:,t] = writer.flex_A[t][bi]
                        end
                    end
                    bgrp = block_name * "/flexible/body_" * string(bi)
                    f[bgrp * "/length"] = writer.flex_lengths[bi]
                    f[bgrp * "/n_points"] = writer.flex_npoints[bi]
                    f[bgrp * "/s"] = writer.flex_svecs[bi]
                    f[bgrp * "/X"] = Xstk
                    f[bgrp * "/force"] = Fstk
                    f[bgrp * "/tension"] = Tstk
                    f[bgrp * "/curvature"] = Kstk
                    if hasV
                        f[bgrp * "/velocity"] = Vstk
                        f[bgrp * "/acceleration"] = Astk
                    end
                end
            end
        end
        # Rigid-body centers/coefficients if any
        if !isempty(writer.centers)
            C = writer.centers
            nb = size(C[1],2)
            CC = Array{Float64,3}(undef, 2, nb, length(C))
            for t=1:length(C); CC[:,:,t] = C[t]; end
            f[block_name * "/rigid_body_centers_xz"] = CC
        end
        if !isempty(writer.coeff_Cd)
            NtK = length(writer.coeff_Cd)
            nb = length(writer.coeff_Cd[1])
            Cd = Array{Float64,2}(undef, nb, NtK)
            Cl = similar(Cd); Fx = similar(Cd); Fz = similar(Cd); Cdp = similar(Cd); Cdv = similar(Cd)
            for t=1:NtK
                Cd[:,t] = writer.coeff_Cd[t]
                Cl[:,t] = writer.coeff_Cl[t]
                Fx[:,t] = writer.coeff_Fx[t]
                Fz[:,t] = writer.coeff_Fz[t]
                Cdp[:,t] = writer.coeff_Cdp[t]
                Cdv[:,t] = writer.coeff_Cdv[t]
            end
            f[block_name * "/coefficients/Cd"] = Cd
            f[block_name * "/coefficients/Cl"] = Cl
            f[block_name * "/coefficients/Fx"] = Fx
            f[block_name * "/coefficients/Fz"] = Fz
            f[block_name * "/coefficients/Cd_pressure"] = Cdp
            f[block_name * "/coefficients/Cd_viscous"] = Cdv
        end
    end
    # Clear buffers
    empty!(writer.times); empty!(writer.iterations)
    empty!(writer.u_slices); empty!(writer.v_slices); empty!(writer.w_slices); empty!(writer.p_slices)
    empty!(writer.centers)
    empty!(writer.coeff_Cd); empty!(writer.coeff_Cl); empty!(writer.coeff_Fx); empty!(writer.coeff_Fz); empty!(writer.coeff_Cdp); empty!(writer.coeff_Cdv)
    empty!(writer.flex_X); empty!(writer.flex_F); empty!(writer.flex_T); empty!(writer.flex_K); empty!(writer.flex_V); empty!(writer.flex_A)
    return true
end

end # module
