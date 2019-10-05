import Conda
using PyCall
using Pkg
pkgs = Conda._installed_packages()


@warn "Installing binary dependencies..."
if !("python" in pkgs) && 
    Conda._installed_packages_dict()["python"][1]>v"3.6"
    Conda.add("python=3.6")
end
# Conda.add("gcc", channel="anaconda")
to_install = ""
for pkg in ["make", "cmake", "zip", "unzip", "matplotlib", "numpy", "scipy", "tensorflow-probability=0.7"]
    global to_install
    if split(pkg, "=")[1] in pkgs; continue; end
    Conda.add(pkg)
end

@warn "Downloading python dependencies..."
PYTHON = joinpath(Conda.BINDIR, "python")
PIP = joinpath(Conda.BINDIR, "pip")
ZIP = joinpath(Conda.BINDIR, "zip")
UNZIP = joinpath(Conda.BINDIR, "unzip")
run(`$PIP install tensorflow==1.14`)
if Sys.islinux()
    run(`$PIP install tensorflow-gpu==1.14`)
end

function install_custom_op_dependency()
    LIBDIR = "$(Conda.LIBDIR)/Libraries"

    # Install Eigen3 library
    if !isdir(LIBDIR)
        @warn "Downloading dependencies to $LIBDIR..."
        mkdir(LIBDIR)
    end

    if !isfile("$LIBDIR/eigen.zip")
        download("http://bitbucket.org/eigen/eigen/get/3.3.7.zip","$LIBDIR/eigen.zip")
    end

    if !isdir("$LIBDIR/eigen3")    
        run(`$UNZIP $LIBDIR/eigen.zip`)
        mv("eigen-eigen-323c052e1731", "$LIBDIR/eigen3", force=true)
    end

    # Install Torch library
    #=
    if Sys.isapple()
        if !isfile("$LIBDIR/libtorch.zip")
            download("https://download.pytorch.org/libtorch/cpu/libtorch-macos-latest.zip","$LIBDIR/libtorch.zip")
        end
        if !isdir("$LIBDIR/libtorch")
            run(`$UNZIP $LIBDIR/libtorch.zip`)
            mv("libtorch", "$LIBDIR/libtorch", force=true)
            if !isdir("$LIBDIR/libtorch/lib/")
                mkdir("$LIBDIR/libtorch/lib/")
            end
            download("https://github.com/intel/mkl-dnn/releases/download/v0.19/mklml_mac_2019.0.5.20190502.tgz","$LIBDIR/mklml_mac_2019.0.5.20190502.tgz")
            run(`tar -xvzf $LIBDIR/mklml_mac_2019.0.5.20190502.tgz`)
            mv("mklml_mac_2019.0.5.20190502/lib/libiomp5.dylib","$LIBDIR/libtorch/lib/libiomp5.dylib", force=true)
            mv("mklml_mac_2019.0.5.20190502/lib/libmklml.dylib","$LIBDIR/libtorch/lib/libmklml.dylib", force=true)
            rm("mklml_mac_2019.0.5.20190502/", force=true, recursive=true)
        end
    elseif Sys.islinux()
        if !isfile("$LIBDIR/libtorch.zip")
            download("https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip","$LIBDIR/libtorch.zip")
        end
        if !isdir("$LIBDIR/libtorch")
            run(`$UNZIP $LIBDIR/libtorch.zip`)
            mv("libtorch", "$LIBDIR/libtorch")
            if !isdir("$LIBDIR/libtorch/lib/")
                mkdir("$LIBDIR/libtorch/lib/")
            end
            download("https://github.com/intel/mkl-dnn/releases/download/v0.19/mklml_lnx_2019.0.5.20190502.tgz","$LIBDIR/mklml_lnx_2019.0.5.20190502.tgz")
            run(`tar -xvzf $LIBDIR/mklml_lnx_2019.0.5.20190502.tgz`)
            mv("mklml_lnx_2019.0.5.20190502/lib/libiomp5.so", "$LIBDIR/libtorch/lib/libiomp5.so", force=true)
            mv("mklml_lnx_2019.0.5.20190502/lib/libmklml_gnu.so", "$LIBDIR/libtorch/lib/libmklml_gnu.so", force=true)
            mv("mklml_lnx_2019.0.5.20190502/lib/libmklml_intel.so", "$LIBDIR/libtorch/lib/libmklml_intel.so", force=true)
            rm("mklml_lnx_2019.0.5.20190502/", force=true, recursive=true)
        end
    end
    =#
end

install_custom_op_dependency()

if haskey(ENV, "LD_LIBRARY_PATH")
    run(setenv(`$PYTHON build.py`, "LD_LIBRARY_PATH"=>ENV["LD_LIBRARY_PATH"]*"$(Conda.LIBDIR)"))
else
    run(setenv(`$PYTHON build.py`, "LD_LIBRARY_PATH"=>"$(Conda.LIBDIR)"))
end

@warn "Fix Python Version"
if PYTHON!=PyCall.python
    ENV["PYTHON"] = "$PYTHON"
    Pkg.build("PyCall")
end
