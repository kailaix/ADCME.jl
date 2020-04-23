if VERSION>=v"1.4" && Sys.isapple() && !(haskey(ENV, "DOCUMENTER_KEY"))
    error("""Your Julia version is ≥1.4, and your operation system is MacOSX. 
Currently, there is a compatibility issue for this combination. 
Please downgrade your Julia version.""")
end

begin 

if haskey(ENV, "GPU") && ENV["GPU"]=="1" && !(Sys.isapple())
    try 
        run(`which nvcc`)
    catch
        error("""You specified ENV["GPU"]=1 but nvcc cannot be found (`which nvcc` failed.
Make sure `nvcc` is available.""")
    end
    s = join(readlines(pipeline(`nvcc --version`)), " ")
    ver = parse(Int64, match(r"V(\d+)\.\d", s)[1])
    if ver!=10
        error("TensorFlow backend of ADCME requires CUDA 10. But you have CUDA $ver")
    end
end


if haskey(ENV, "MANUAL") && ENV["MANUAL"] == "1"
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

pkgs_dict = Conda._installed_packages_dict()
pkgs = collect(keys(pkgs_dict))

@info " ########### Install binaries ########### "
ZIP = easy_get("zip")
UNZIP = easy_get("unzip")
GIT = "LibGit2"


@info " ########### Install CONDA dependencies ########### "
!("make" in pkgs) && Conda.add("make")
!("cmake" in pkgs) && Conda.add("cmake")
!("matplotlib" in pkgs) && Conda.add("matplotlib")

if haskey(ENV, "GPU") && ENV["GPU"]=="1" && Sys.isapple()
    @warn "MacOSX does not support Tensorflow GPU, ignoring GPU..."
end

if haskey(ENV, "GPU") && ENV["GPU"]=="1" && !(Sys.isapple())
    !("tensorflow-gpu" in pkgs) && Conda.add("tensorflow-gpu=1.15")
else 
    if !("tensorflow" in pkgs || "tensorflow-gpu" in pkgs)
        Conda.add("tensorflow=1.15")
    end
end
!("tensorflow-probability" in pkgs) && Conda.add("tensorflow-probability=0.7")
!("hdf5" in pkgs) && Conda.add("hdf5", channel="anaconda")

@info " ########### Preparing environment for custom operators ########### "
tf = pyimport("tensorflow")
core_path = abspath(joinpath(tf.sysconfig.get_compile_flags()[1][3:end], ".."))
lib = readdir(core_path)
TF_LIB_FILE = joinpath(core_path,lib[findall(occursin.("libtensorflow_framework", lib))[end]])
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
        run(`$UNZIP -qq $LIBDIR/eigen.zip`)
        mv("eigen-eigen-323c052e1731", "$LIBDIR/eigen3", force=true)
    end
end

install_custom_op_dependency()

# useful command for debug
# readelf -p .comment libtensorflow_framework.so 
# strings libstdc++.so.6 | grep GLIBCXX
if Sys.islinux() 
    verinfo = read(`readelf -p .comment $TF_LIB_FILE`, String)
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

LIBCUDA = ""
if haskey(ENV, "GPU") && ENV["GPU"]=="1" && !(Sys.isapple())
    @info " ########### CUDA dynamic libraries  ########### "
    pkg_dir = joinpath(Conda.ROOTENV, "pkgs/")
    files = readdir(pkg_dir)
    libpath = filter(x->startswith(x, "cudatoolkit") && isdir(joinpath(pkg_dir,x)), files)
    if length(libpath)==0
        error("cudatoolkit* not found in $pkg_dir")
    elseif length(libpath)>1
        @warn "more than 1 cudatoolkit found, use $(libpath[1]) by default"
    end
    LIBCUDA = joinpath(pkg_dir, libpath[1], "lib")

    libpath = filter(x->startswith(x, "cudnn") && isdir(joinpath(pkg_dir,x)), files)
    if length(libpath)==0
        error("cudnn* not found in $pkg_dir")
    elseif length(libpath)>1
        @warn "more than 1 cudatoolkit found, use $(libpath[1]) by default"
    end
    LIBCUDA = LIBCUDA*":"*joinpath(pkg_dir, libpath[1], "lib")

    @info " ########### CUDA include headers  ########### "
    cudnn = joinpath(pkg_dir, libpath[1], "include", "cudnn.h")
    cp(cudnn, joinpath(TF_INC, "cudnn.h"), force=true)

    NVCC = readlines(pipeline(`which nvcc`))[1]
    NVCC_INC = joinpath(splitdir(splitdir(NVCC)[1])[1], "include")
    TF_INC = TF_INC*":"*NVCC_INC
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
adding("TF_LIB_FILE", TF_LIB_FILE)
adding("LIBCUDA", LIBCUDA)


t = "join(["*join(t, ",")*"], \";\")"
s *= "__STR__ = $t"
open("deps.jl", "w") do io 
    write(io, s)
end

@label writedeps  


end