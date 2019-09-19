# install requirements.txt
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
if PyCall.python!=readlines(`which python`)[1]
    error("Python version and PyCall Python version does not match. Please reinstall PyCall with the default Python version.
PyCall Python: $(PyCall.python)
System Python: $(readlines(`which python`)[1])
Instruction: 
julia> ENV[\"PYTHON\"] = $(readlines(`which python`)[1])
julia> using Pkg; Pkg.build(\"PyCall\")")
end

if Sys.iswindows()
    @warn "PyTorch plugin is still under construction for Windows platform and will be disabled for the current version."
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

mksymlink()