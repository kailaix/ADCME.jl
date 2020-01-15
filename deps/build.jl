push!(LOAD_PATH, "@stdlib")
using Pkg
using Conda
PYTHON = joinpath(Conda.BINDIR, "python")
if !haskey(Pkg.installed(), "PyCall")
    ENV["PYTHON"]=PYTHON
    Pkg.add("PyCall"); Pkg.build("PyCall")
end
using PyCall
println("PyCall Python: $(PyCall.python)
Conda Python: $(PYTHON)")

tf_ver = "1.14"
PIP = joinpath(Conda.BINDIR, "pip")
ZIP = joinpath(Conda.BINDIR, "zip")
UNZIP = joinpath(Conda.BINDIR, "unzip")

@info "Install CONDA dependencies..."
pkgs = Conda._installed_packages()
for pkg in ["zip", "unzip", "make", "cmake", "tensorflow=$tf_ver", "tensorflow-probability=0.7",
            "matplotlib", "git"]
    if split(pkg,"=")[1] in pkgs; continue; end 
    Conda.add(pkg)
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

try
    Conda.clean()
catch
    @warn "Conda.clean() failed"
end