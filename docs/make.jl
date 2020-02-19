using PyCall 
using Pkg; Pkg.add("PyPlot")

using Documenter, ADCME
makedocs(sitename="ADCME", modules=[ADCME],
pages = Any[
    "index.md",
    "inverse_modeling.md",
    "Tutorial" => ["tutorial.md"]
    "Manual" => ["inverse_impl.md", "array.md", "sparse.md", "newton_raphson.md", "parallel.md", "ode.md"],
    "Resources" => ["customop.md", "global.md", "while_loop.md",
            "julia_customop.md", "pytorchnn.md", "extra.md", "ot.md", "resource_manager.md"],
    "Applications" => ["apps_ana.md", "apps_levy.md", "apps_constitutive_law.md", "apps_ad.md"],
    "api.md"
],
authors = "Kailai Xu")

deploydocs(
    repo = "github.com/kailaix/ADCME.jl.git",
)