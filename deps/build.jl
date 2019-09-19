using PyCall

if Sys.iswindows()
    @warn "PyTorch plugin is still under construction for Windows platform and will be disabled for the current version."
end

function package_exist(s::String)
py"""
import pkgutil; 
exist_ = True if pkgutil.find_loader($s) else False
"""
py"exist_"
end

function mksymlink()
    tf = pyimport("tensorflow")
    if Sys.isapple()
        ext = "dylib"
    elseif Sys.iswindows()
        ext = "dll"
    elseif Sys.islinux()
        ext = "so"
    end
    tfdir = splitdir(tf.__file__)[1] 
    if !isfile(joinpath(tfdir, "libtensorflow_framework.$ext"))
        for f in readdir(tfdir)
            if occursin("libtensorflow_framework", f)
                name = joinpath(tfdir, f)
                link = joinpath(tfdir, "libtensorflow_framework.$ext")
                @info "Creating symbolic link $link-->$name"
                symlink(name, link)
            end
        end
    end
end

function install_tensorflow()
    if haskey(ENV,"REINSTALL_PIP")
        @info "Reinstall pip..."
        download("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")
        run(`$(PyCall.python) get-pip.py --user`)
        rm("get-pip.py")
    end
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

if !(package_exist("tensorflow") && package_exist("tensorflow_probability"))
    install_tensorflow()
end
mksymlink()