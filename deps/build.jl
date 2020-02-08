begin 
if haskey(ENV, "MANUAL")
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
@info """
PyCall Python version: $(PyCall.python)
Conda Python version: $PYTHON
"""

easy_get = pkg->begin
    try
        strip(read(pipeline(`which $pkg`), String))
    catch
        Conda.add(pkg)
        joinpath(Conda.BINDIR, pkg)
    end
end

pkgs = Conda._installed_packages()

@info "Install binaries"
ZIP = easy_get("zip")
UNZIP = easy_get("unzip")
GIT = easy_get("git")

@info "Install CONDA dependencies..."
for pkg in ["make", "cmake", "tensorflow=1.14", "tensorflow-probability=0.7",
            "matplotlib"]
    if split(pkg,"=")[1] in pkgs; continue; end 
    Conda.add(pkg)
end
!("hdf5" in pkgs) && Conda.add("hdf5", channel="anaconda")


function enable_gpu()
    pkgs = Conda._installed_packages()

    if !("tensorflow-gpu" in pkgs)
        Conda.add("tensorflow-gpu=1.14")
    end
    
    if !("cudatoolkit" in pkgs)
        Conda.add("cudatoolkit", channel="anaconda")
    end

    gpus = joinpath(splitdir(tf.__file__)[1], "include/third_party/gpus")
    if !isdir(gpus)
    mkdir(gpus)
    end
    gpus = joinpath(gpus, "cuda")
    if !isdir(gpus)
    mkdir(gpus)
    end
    incpath = joinpath(splitdir(strip(read(`which nvcc`, String)))[1], "../include/")
    if !isdir(joinpath(gpus, "include"))
        cp(incpath, joinpath(gpus, "include"))
    end

    pth = joinpath(Conda.ROOTENV, "pkgs/cudatoolkit-10.1.168-0/lib/")
    # compatible 
    files = readdir(pth)
    for f in files
        if f[end-2:end]==".10" && !isfile(joinpath(pth, f*".0"))
            symlink(joinpath(pth, f), joinpath(pth, f*".0"))
        end
        if f[end-4:end]==".10.1" && !isfile(joinpath(pth, f[1:end-2]*".0"))
            symlink(joinpath(pth, f), joinpath(pth, f[1:end-2]*".0"))
        end
    end
    
    @info("Run the following command in shell

    echo 'export LD_LIBRARY_PATH=$pth:\$LD_LIBRARY_PATH' >> ~/.bashrc")
end

if haskey(ENV, "GPU")
    enable_gpu()
end


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
        @info("The GCC version which TensorFlow was compiled is not officially supported by ADCME. You have the following choices
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
adding("GIT", GIT)
adding("PYTHON", PyCall.python)
adding("TF_LIB_FILE", tflib)

t = "join(["*join(t, ",")*"], \";\")"
s *= "__STR__ = $t"
open("deps.jl", "w") do io 
    write(io, s)
end

@label writedeps  


end