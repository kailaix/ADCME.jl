using PyCall
using Documenter
using ADCME

function install_packages()
    try 
        run(`$(PyCall.python) -m pip --version`)
    catch
        @warn "pip is not installed, downloading and installing pip..."
        download("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")
        run(`$(PyCall.python) get-pip.py --user`)
        rm("get-pip.py")
    end
    run(`$(PyCall.python) -m pip install --user scipy`)
    run(`$(PyCall.python) -m pip install --user matplotlib`)
end

install_packages()


makedocs(sitename="ADCME", modules=[ADCME],
pages = Any["index.md"],
authors = "Kailai Xu")


deploydocs(
    repo = "github.com/kailaix/ADCME.jl.git",
)