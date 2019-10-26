using PyCall 
using Pkg; Pkg.add("PyPlot")

using Documenter, ADCME
makedocs(sitename="ADCME", modules=[ADCME],
pages = Any[
    "index.md",
    "Inverse Modeling" => ["inverse_modeling.md"],
    "Automatic Differentiation" => ["four_types.md"],
    "Resources" => ["customop.md", "while_loop.md", "julia_customop.md", "pytorchnn.md", "extra.md"],
    "Applications" => [],
    "api.md"
],
authors = "Kailai Xu")

deploydocs(
    repo = "github.com/kailaix/ADCME.jl.git",
)