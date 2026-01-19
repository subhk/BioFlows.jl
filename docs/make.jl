using Documenter
using BioFlows

makedocs(
    sitename = "BioFlows.jl",
    authors = "Subhajit Kar, Dibyendu Ghosh",
    modules = [BioFlows],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://subhk.github.io/BioFlows.jl",
        assets = ["assets/custom.css"],
        sidebar_sitename = true,
        collapselevel = 2,
        highlights = ["julia"],
        footer = "BioFlows.jl — High-Performance CFD in Julia",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "GPU Acceleration" => "gpu.md",
        "Core Types" => "core_types.md",
        "Adaptive Mesh Refinement" => "amr.md",
        "Diagnostics" => "diagnostics.md",
        "Examples" => "examples.md",
        "Numerical Methods" => "numerical_methods.md",
        "Codebase Structure" => "codebase_structure.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
    warnonly = true,
)

deploydocs(
    repo = "github.com/subhk/BioFlows.jl.git",
    devbranch = "main",
    push_preview = true,
)
