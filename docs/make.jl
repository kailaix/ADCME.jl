cd(@__DIR__)
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Documenter, ADCME
makedocs(sitename="ADCME", modules=[ADCME],
pages = Any[
    "index.md",
    "Tutorial"=>["tutorial.md","resources.md", "tu_whatis.md", "tu_basic.md", "tu_optimization.md", "tu_sparse.md", "tu_fd.md", "tu_fem.md",
        "tu_inv.md", "tu_recipe.md", "tu_nn.md", "tu_implicit.md", "tu_customop.md","tu_debug.md", "exercise.md"],
    "Resources" => ["newton_raphson.md", "parallel.md", "optimizers.md", "ode.md", "global.md", 
            "julia_customop.md", "nn.md", "ot.md", "resource_manager.md", "alphascheme.md", "factorization.md", "customopt.md",
            "options.md", "smt.md"],
    "Deep Learning Schemes" => ["vae.md", "flow.md", "convnet.md", "bnn.md"],
    "Applications" => ["apps.md", "apps_ana.md", "apps_levy.md", 
            "apps_constitutive_law.md", "apps_ad.md", "apps_adseismic.md", "apps_nnfem.md"],
    "api.md"
],
authors = "Kailai Xu")

deploydocs(
    repo = "github.com/kailaix/ADCME.jl.git",
)