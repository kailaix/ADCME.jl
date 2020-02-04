begin 
if haskey(ENV, "manual")
    include("build2.jl")
    @goto writedeps
end

push!(LOAD_PATH, "@stdlib")
using Pkg
using Conda

PYTHON = joinpath(Conda.BINDIR, "python")
!haskey(Pkg.installed(), "PyCall") && Pkg.add("PyCall")
ENV["PYTHON"]=PYTHON
Pkg.build("PyCall")

using PyCall
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
!("hdf5" in pkgs) && Conda.add("hdf5", channel="anaconda")


@info "Preparing environment for custom operators"
tf = pyimport("tensorflow")
lib = readdir(splitdir(tf.__file__)[1])
tflib = joinpath(splitdir(tf.__file__)[1],lib[findall(occursin.("libtensorflow_framework", lib))[1]])
TF_INC = tf.sysconfig.get_compile_flags()[1][3:end]
TF_ABI = tf.sysconfig.get_compile_flags()[2][end:end]

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


s = ""
t = []
function adding(k, v)
    global s 
    s *= "$k = \"$v\"\n"
    push!(t, "$k")
end
adding("BINDIR", Conda.BINDIR)
adding("LIBDIR", Conda.LIBDIR)
adding("TF_INC", TF_INC)
adding("TF_ABI", TF_ABI)
adding("EIGEN_INC", joinpath(Conda.LIBDIR,"Libraries"))
adding("CC", joinpath(Conda.BINDIR, "gcc"))
adding("CXX", joinpath(Conda.BINDIR, "g++"))
adding("CMAKE", joinpath(Conda.BINDIR, "cmake"))
adding("MAKE", joinpath(Conda.BINDIR, "make"))
adding("GIT", joinpath(Conda.BINDIR, "git"))
adding("PYTHON", PyCall.python)
adding("TF_LIB_FILE", tflib)

t = "join(["*join(t, ",")*"], \";\")"
s *= "__STR__ = $t"
open("deps.jl", "w") do io 
    write(io, s)
end

@label writedeps  


end