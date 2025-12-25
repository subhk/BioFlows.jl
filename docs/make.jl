using Documenter
using BioFlows

makedocs(
    sitename = "BioFlows.jl",
    authors = "Subhajit Kar, Dibyendu Ghosh",
    modules = [BioFlows],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://subhk.github.io/BioFlows.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Codebase Structure" => "codebase_structure.md",
        "Numerical Methods" => "numerical_methods.md",
        "Core Types" => "core_types.md",
        "Adaptive Mesh Refinement" => "amr.md",
        "Diagnostics" => "diagnostics.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,  # Only check exported symbols
    warnonly = true,       # Treat all errors as warnings
)

deploydocs(
    repo = "github.com/subhk/BioFlows.jl.git",
    devbranch = "main",
    push_preview = true,
)
