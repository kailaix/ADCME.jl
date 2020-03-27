using PyCall 
using Pkg; Pkg.add("PyPlot")

using Documenter, ADCME
makedocs(sitename="ADCME", modules=[ADCME],
pages = Any[
    "index.md",
    "Tutorial"=>["tutorial.md", "tu_whatis.md", "tu_basic.md", "tu_sparse.md", "tu_fd.md", "tu_fem.md",
        "tu_inv.md", "tu_recipe.md", "tu_implicit.md", "exercise.md"],
    "Resources" => ["newton_raphson.md", "parallel.md", "ode.md", "customop.md", "global.md", 
            "julia_customop.md", "nn.md", "extra.md", "ot.md", "resource_manager.md"],
    "Applications" => ["apps.md", "apps_ana.md", "apps_levy.md", 
            "apps_constitutive_law.md", "apps_ad.md", "apps_adseismic.md"],
    "api.md"
],
authors = "Kailai Xu")

deploydocs(
    repo = "github.com/kailaix/ADCME.jl.git",
)