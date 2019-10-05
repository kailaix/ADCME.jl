import Conda 
using PyCall
pkgs = Conda._installed_packages()

@warn "Installing binary dependencies..."
for pkg in ["make", "cmake", "zip", "python", "unzip"]
    if pkg in pkgs; continue; end
    Conda.add(pkg)
end
if !("gcc" in pkgs)
    Conda.add("gcc", channel="anaconda")
end


@warn "Downloading python dependencies..."
PYTHON = joinpath(Conda.BINDIR, "python")
PIP = joinpath(Conda.BINDIR, "pip")
ZIP = joinpath(Conda.BINDIR, "zip")
UNZIP = joinpath(Conda.BINDIR, "unzip")
GCC = joinpath(Conda.BINDIR, "")
run(`$PIP install -U -r $(@__DIR__)/requirements.txt`)
if Sys.islinux()
    run(`$PIP install -U tensorflow-gpu==1.14`)
end


function install_custom_op_dependency()
    LIBDIR = "$(@__DIR__)/Libraries"

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
    if Sys.isapple()
        if !isfile("$LIBDIR/libtorch.zip")
            download("https://download.pytorch.org/libtorch/cpu/libtorch-macos-latest.zip","$LIBDIR/libtorch.zip")
        end
        if !isdir("$LIBDIR/libtorch")
            run(`$UNZIP $LIBDIR/libtorch.zip`)
            mv("libtorch", "$LIBDIR/libtorch", force=true)
            mkdir("$LIBDIR/libtorch/lib/")
            download("https://github.com/intel/mkl-dnn/releases/download/v0.19/mklml_mac_2019.0.5.20190502.tgz","$LIBDIR/mklml_mac_2019.0.5.20190502.tgz")
            run(`tar -xvzf $LIBDIR/mklml_mac_2019.0.5.20190502.tgz`)
            mv("mklml_mac_2019.0.5.20190502/lib/libiomp5.dylib","$LIBDIR/libtorch/lib/", force=true)
            mv("mklml_mac_2019.0.5.20190502/lib/libmklml.dylib","$LIBDIR/libtorch/lib/", force=true)
            rm("mklml_mac_2019.0.5.20190502/", force=true, recursive=true)
        end
    elseif Sys.islinux()
        if !isfile("$LIBDIR/libtorch.zip")
            download("https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip","$LIBDIR/libtorch.zip")
        end
        if !isdir("$LIBDIR/libtorch")
            run(`$UNZIP $LIBDIR/libtorch.zip`)
            run(`mv libtorch $LIBDIR/libtorch`)
            mkdir("$LIBDIR/libtorch/lib/")
            download("https://github.com/intel/mkl-dnn/releases/download/v0.19/mklml_lnx_2019.0.5.20190502.tgz","$LIBDIR/mklml_lnx_2019.0.5.20190502.tgz")
            run(`tar -xvzf $LIBDIR/mklml_lnx_2019.0.5.20190502.tgz`)
            mv("mklml_lnx_2019.0.5.20190502/lib/libiomp5.so", "$LIBDIR/libtorch/lib/", force=true)
            mv("mklml_lnx_2019.0.5.20190502/lib/libmklml_gnu.so", "$LIBDIR/libtorch/lib/", force=true)
            mv("mklml_lnx_2019.0.5.20190502/lib/libmklml_intel.so", "$LIBDIR/libtorch/lib/", force=true)
            rm("mklml_lnx_2019.0.5.20190502/", force=true, recursive=true)
        end
    end
end

function mksymlink()
    tf = pyimport_conda("tensorflow", "tensorflow")
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

install_custom_op_dependency()
mksymlink()