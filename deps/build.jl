import Conda
using PyCall
# assumption: PyCall and Conda are installed, and PyCall.conda==true 

tf_ver = "1.14"
PYTHON = joinpath(Conda.BINDIR, "python")
PIP = joinpath(Conda.BINDIR, "pip")
ZIP = joinpath(Conda.BINDIR, "zip")
UNZIP = joinpath(Conda.BINDIR, "unzip")

if !PyCall.conda
    error("""ADCME requires that PyCall use the Conda.jl Python.
Rebuild PyCall with

julia> ENV["PYTHON"] = "$PYTHON"
julia> using Pkg; Pkg.build("PyCall")

""")
end

@info "Install CONDA dependencies..."
pkgs = Conda._installed_packages()
for pkg in ["zip", "unzip", "make", "cmake", "tensorflow=$tf_ver", "tensorflow-probability=0.7",
            "matplotlib"]
    if split(pkg,"=")[1] in pkgs; continue; end 
    Conda.add(pkg)
end

if Sys.islinux() && haskey(ENV, "GPU") && !("tensorflow-gpu" in pkgs)
    @info "Add tensorflow-gpu"
    # Conda.add("tensorflow-gpu=$tf_ver")
    run(`$PIP install tensorflow-gpu==$tf_ver`)
end

@info "Fix libtensorflow_framework.so..."
if haskey(ENV, "LD_LIBRARY_PATH")
    run(setenv(`$PYTHON build.py`, "LD_LIBRARY_PATH"=>ENV["LD_LIBRARY_PATH"]*":$(Conda.LIBDIR)"))
else
    run(setenv(`$PYTHON build.py`, "LD_LIBRARY_PATH"=>"$(Conda.LIBDIR)"))
end

function install_custom_op_dependency()
    LIBDIR = "$(Conda.LIBDIR)/Libraries"

    # Install Eigen3 library
    if !isdir(LIBDIR)
        @info "Downloading dependencies to $LIBDIR..."
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


# useful command for debug
# readelf -p .comment libtensorflow_framework.so 
# strings libstdc++.so.6 | grep GLIBCXX
if Sys.islinux() 
    tflib = nothing
    if occursin("3.6", PyCall.libpython)
        tflib = joinpath(Conda.LIBDIR, "python3.6/site-packages/tensorflow/libtensorflow_framework.so")
    elseif occursin("3.7", PyCall.libpython)
        tflib = joinpath(Conda.LIBDIR, "python3.7/site-packages/tensorflow/libtensorflow_framework.so")
    end
    verinfo = read(`readelf -p .comment $tflib`, String)
    Conda.clean()
    if occursin("5.4", verinfo)
        if !("gcc-5" in Conda._installed_packages())
            try
                # a workaround
                Conda.add("mpfr", channel="anaconda")
                if !isfile(joinpath(Conda.LIBDIR, "libmpfr.so.4")) && isfile(joinpath(Conda.LIBDIR, "libmpfr.so.6"))
                    symlink(joinpath(Conda.LIBDIR, "libmpfr.so.6"), joinpath(Conda.LIBDIR, "libmpfr.so.4"))
                end
                Conda.add("gcc-5", channel="psi4")
                Conda.add("libgcc")
            catch
                error("Installation of GCC failed. Follow the documentation for instructions.")
            end
        end
    elseif occursin("4.8.", verinfo)
        if !("gcc" in Conda._installed_packages())
            Conda.add("gcc", channel="anaconda")
            Conda.add("libgcc")
        end
    else
        error("The GCC version which TensorFlow was compiled is not officially supported by ADCME. You have the following choices
1. Continue using ADCME. But you are responsible for the compatible issue of GCC versions for custom operators.
2. Report to the author of ADCME by opening an issue in https://github.com/kailaix/ADCME.jl/
Compiler information:
$verinfo
")
    end
    rm(joinpath(Conda.LIBDIR,"libstdc++.so.6"), force=true)
    @info "Making a symbolic link for libgcc"
    symlink(joinpath(Conda.LIBDIR,"libstdc++.so.6.0.26"), joinpath(Conda.LIBDIR,"libstdc++.so.6"))
end