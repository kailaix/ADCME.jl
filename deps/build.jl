using PyCall

if Sys.iswindows()
    @warn "PyTorch plugin is still under construction for Windows platform and will be disabled for the current version."
end

function install_tensorflow()
    try 
        run(`$(PyCall.python) -m pip --version`)
    catch
        @warn "pip is not installed, downloading and installing pip..."
        download("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")
        run(`$(PyCall.python) get-pip.py --user`)
        rm("get-pip.py")
    end
    run(`$(PyCall.python) -m pip install --user -U numpy`)
    run(`$(PyCall.python) -m pip install --user tensorflow==1.14`)
    run(`$(PyCall.python) -m pip install --user tensorflow_probability==0.7`)
end

try
    tf = pyimport("tensorflow")
    tf = pyimport("tensorflow_probability")
catch ee
    install_tensorflow()
end