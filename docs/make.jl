
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
run(`pip install --user tensorflow==1.14`)
run(`pip install --user tensorflow_probability==0.7`)
run(`pip install --user -U numpy`)
run(`pip install --user -U scipy`)
run(`pip install --user -U matplotlib`)


using PyCall 
using Pkg; Pkg.add("PyPlot")
if PyCall.python!=readlines(`which python`)[1]
    error("Python version and PyCall Python version does not match. Please reinstall PyCall with the default Python version.
PyCall Python: $(PyCall.python)
System Python: $(readlines(`which python`)[1])
Instruction: 
julia -e 'ENV[\"PYTHON\"] = readlines(`which python`)[1]; using Pkg; Pkg.build(\"PyCall\")")
end

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