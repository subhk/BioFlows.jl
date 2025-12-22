### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 00000000-0000-0000-0000-000000000001
begin
    using JLD2
    using CairoMakie
    using PlutoUI
end

# ╔═╡ 00000000-0000-0000-0000-000000000002
md"""
# Cylinder Vorticity Viewer

This Pluto notebook visualises the vorticity field stored in a `cylinder_shedding_XXX.jld2` snapshot.

Select the snapshot file (default: `cylinder_shedding_005.jld2`) and use the sliders to choose the time index if stacked data are present.
"""

# ╔═╡ 9a92c9c2-3f24-4a58-91f7-1c8e3e9e6ee0
@bind snapshot_path TextField(
    default = "cylinder_shedding_005.jld2",
    placeholder = "Path to cylinder_shedding_XXX.jld2",
    width = 400,
)

# ╔═╡ a76193de-1a9d-4605-b4be-b2832b0bb1c4
data_keys = begin
    isfile(snapshot_path) || return String[]
    JLD2.jldopen(snapshot_path, "r") do f
        collect(keys(f))
    end
end

# ╔═╡ 994e8308-87b4-4474-a772-83fc10d3f081
md"""
### Available top-level datasets
$(join(string.(data_keys), ", "))
"""

# ╔═╡ 9c7f7b48-0ec9-47a1-8863-4e74877d2ef4
begin
    function read_dataset(path::String, name::String)
        JLD2.jldopen(path, "r") do f
            haskey(f, name) ? f[name] : nothing
        end
    end

    # Try stacked first, then base-level datasets
    omega_raw = begin
        haskey = n -> (JLD2.jldopen(snapshot_path, "r") do f; haskey(f, n) end)
        if isfile(snapshot_path)
            if haskey("stacked/omega")
                read_dataset(snapshot_path, "stacked/omega")
            elseif haskey("omega")
                read_dataset(snapshot_path, "omega")
            elseif haskey("stacked/omega_mag")
                read_dataset(snapshot_path, "stacked/omega_mag")
            elseif haskey("omega_mag")
                read_dataset(snapshot_path, "omega_mag")
            else
                nothing
            end
        else
            nothing
        end
    end
end

# ╔═╡ 072de91c-a329-4c38-b445-0cfdf3934a51
omega_is_stacked = omega_raw isa Array{<:Real,3} || omega_raw isa Array{<:Real,4}

# ╔═╡ 65d4d1ce-5a61-4e26-9dce-71f54dfd774c
@bind time_index Slider(
    omega_is_stacked ? 1:size(omega_raw, ndims(omega_raw)) : 1:1,
    default = 1,
    show_value = true,
)

# ╔═╡ 04f9f620-0f46-4f40-a46a-1245c8878b51
grid_x = begin
    isfile(snapshot_path) || return nothing
    read_dataset(snapshot_path, "grid/x")
end

# ╔═╡ aa6a6c19-1dc1-4ac4-8592-5a8c329edbe3
grid_z = begin
    isfile(snapshot_path) || return nothing
    read_dataset(snapshot_path, "grid/z")
end

# ╔═╡ 0088bdd8-6f91-4d70-8d06-5bdba1406a3f
omega_field = begin
    omega_raw === nothing && return nothing
    nd = ndims(omega_raw)
    if nd == 3
        view(omega_raw, :, :, time_index)
    elseif nd == 4
        # take magnitude slice assuming omega_raw is magnitude array (nx,ny,nz,Nt)
        # For a 3D magnitude we reduce over the middle dimension by averaging to a 2D presentation.
        slice = view(omega_raw, :, :, :, time_index)
        dropdims(mean(slice; dims = 2), dims = 2)
    else
        omega_raw
    end
end

# ╔═╡ fdf6b34d-4b00-4860-9d3f-7253ccf84374
md"""
### Vorticity heatmap
"""

# ╔═╡ c775f653-8f72-4c5c-9ebf-235c70222f9a
let
    if omega_field === nothing || grid_x === nothing || grid_z === nothing
        md"_Snapshot file not found or vorticity dataset missing._"
    else
        fig = Figure(resolution = (700, 500))
        ax = Axis(fig[1, 1],
            xlabel = "x",
            ylabel = "z",
            title = "Vorticity slice (time index = $time_index)"
        )
        heatmap!(ax, grid_x, grid_z, transpose(omega_field))
        Colorbar(fig[1, 2], ax, label = "ω")
        fig
    end
end

# ╔═╡ Cell order:
# ╠═00000000-0000-0000-0000-000000000001
# ╠═00000000-0000-0000-0000-000000000002
# ╠═9a92c9c2-3f24-4a58-91f7-1c8e3e9e6ee0
# ╠═a76193de-1a9d-4605-b4be-b2832b0bb1c4
# ╠═994e8308-87b4-4474-a772-83fc10d3f081
# ╠═9c7f7b48-0ec9-47a1-8863-4e74877d2ef4
# ╠═072de91c-a329-4c38-b445-0cfdf3934a51
# ╠═65d4d1ce-5a61-4e26-9dce-71f54dfd774c
# ╠═04f9f620-0f46-4f40-a46a-1245c8878b51
# ╠═aa6a6c19-1dc1-4ac4-8592-5a8c329edbe3
# ╠═0088bdd8-6f91-4d70-8d06-5bdba1406a3f
# ╠═fdf6b34d-4b00-4860-9d3f-7253ccf84374
# ╠═c775f653-8f72-4c5c-9ebf-235c70222f9a
# ╚═00000000-0000-0000-0000-000000000001
