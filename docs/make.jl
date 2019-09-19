
try
    run(`which pip`)
    if haskey(ENV, "REINSTALL_PIP")
        error("Force Reinstall Pip...")
    end
catch
    run(`wget -O get-pip.py https://bootstrap.pypa.io/get-pip.py`)
    run(`python get-pip.py --user`)
    rm("get-pip.py")
end
run(`pip install --user -U -r $(@__DIR__)/requirements.txt`)

using PyCall 
using Pkg; Pkg.add("PyPlot")

using Documenter, ADCME
makedocs(sitename="ADCME", modules=[ADCME],
pages = Any[
    "Getting Started" => "index.md",
    "Additional Tools" => "extra.md"
],
authors = "Kailai Xu")

deploydocs(
    repo = "github.com/kailaix/ADCME.jl.git",
)