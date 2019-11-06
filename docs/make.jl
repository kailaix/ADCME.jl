using PyCall 
using Pkg; Pkg.add("PyPlot")

using Documenter, ADCME
makedocs(sitename="ADCME", modules=[ADCME],
pages = Any[
    "index.md",
    "inverse_modeling.md",
    "Manual" => ["array.md", "sparse.md", "newton_raphson.md"],
    "Resources" => ["customop.md", "while_loop.md",
            "julia_customop.md", "pytorchnn.md", "extra.md"],
    "Applications" => ["apps_ana.md", "apps_levy.md", "apps_constitutive_law.md"],
    "api.md"
],
authors = "Kailai Xu")

deploydocs(
    repo = "github.com/kailaix/ADCME.jl.git",
)