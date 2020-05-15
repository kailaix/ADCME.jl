if VERSION>=v"1.4" && Sys.isapple() && !(haskey(ENV, "DOCUMENTER_KEY"))
    error("""Your Julia version is ≥1.4, and your operation system is MacOSX. 
Currently, there is a compatibility issue for this combination. 
Please downgrade your Julia version.""")
end

if haskey(ENV, "MANUAL") && ENV["MANUAL"]=="1" 
    error("""****** You indicated you want to build ADCME package manually. 
To this end, you need to create a dependency file 
$(joinpath(@__DIR__, "deps.jl"))
and populate it with appropriate binary locations. 
--------------------------------------------------------------------------------------------
BINDIR = ""
LIBDIR = ""
TF_INC = ""
TF_ABI = ""
EIGEN_INC = ""
CC = ""
CXX = ""
CMAKE = ""
MAKE = ""
GIT = ""
PYTHON = ""
TF_LIB_FILE = ""
LIBCUDA = ""
CUDA_INC = ""
__STR__ = join([BINDIR,LIBDIR,TF_INC,TF_ABI,EIGEN_INC,CC,CXX,CMAKE,MAKE,GIT,PYTHON,TF_LIB_FILE,LIBCUDA,CUDA_INC], ";")
--------------------------------------------------------------------------------------------
""")
end




push!(LOAD_PATH, "@stdlib")
using Pkg
using Conda
using CMake


ENVDIR = abspath("$(Conda.ROOTENV)/envs/ADCME")

VER = haskey(Pkg.installed(),"ADCME")  ? Pkg.installed()["ADCME"] : "NOT_INSTALLED"
@info """Your Julia version is $VERSION, current ADCME version is $VER, ADCME env: $ENVDIR"""

@info " --------------- Install Tensorflow Dependencies  --------------- "

if haskey(ENV, "FORCE_REINSTALL_ADCME") && ENV["FORCE_REINSTALL_ADCME"]=="1" && "adcme" in Conda._installed_packages(:ADCME)
    @info " --------------- Remove Existing ADCME Environment  --------------- "
    rm(ENVDIR, force=true, recursive = true)
end

if !("adcme" in Conda._installed_packages(:ADCME))
    Conda.add("adcme", :ADCME, channel="kailaix")
end

BINDIR = Sys.iswindows() ? abspath("$ENVDIR/Scripts") : abspath("$ENVDIR/bin")  

GIT = "LibGit2"
PYTHON = joinpath(BINDIR, "python")

if Sys.iswindows()
    PYTHON = abspath(joinpath(ENVDIR, "python.exe"))
end
@info " --------------- Check Python Version  --------------- "

!haskey(Pkg.installed(), "PyCall") && Pkg.add("PyCall")
ENV["PYTHON"]=PYTHON
Pkg.build("PyCall")
using PyCall
@info """
PyCall Python version: $(PyCall.python)
Conda Python version: $PYTHON
"""

@info " --------------- Looking for TensorFlow Dynamic Libraries --------------- "
tf = pyimport("tensorflow")
core_path = abspath(joinpath(tf.sysconfig.get_compile_flags()[1][3:end], ".."))
lib = readdir(core_path)
if Sys.iswindows()
    global TF_LIB_FILE = abspath(joinpath(core_path, "python/_pywrap_tensorflow_internal.lib"))
else 
    global TF_LIB_FILE = joinpath(core_path,lib[findall(occursin.("libtensorflow_framework", lib))[end]])
end
TF_INC = tf.sysconfig.get_compile_flags()[1][3:end]
TF_ABI = tf.sysconfig.get_compile_flags()[2][end:end]

@info " --------------- Preparing Custom Operator Environment --------------- "
LIBDIR = abspath("$ENVDIR/lib/Libraries")

if !isdir(LIBDIR)
    @info "Downloading dependencies to $LIBDIR..."
    mkdir(LIBDIR)
end

if !isfile("$LIBDIR/eigen.zip")
    download("http://bitbucket.org/eigen/eigen/get/3.3.7.zip","$LIBDIR/eigen.zip")
end

if !isdir("$LIBDIR/eigen3")    
    UNZIP =  joinpath(BINDIR, "unzip")
    if Sys.iswindows()
        if !isfile("$LIBDIR/unzip.exe")
            download("http://stahlworks.com/dev/unzip.exe", joinpath(LIBDIR, "unzip.exe"))
        end
        UNZIP =  joinpath(LIBDIR, "unzip.exe")
    end 
    run(`$UNZIP -qq $LIBDIR/eigen.zip -d $LIBDIR`)
    mv("$LIBDIR/eigen-eigen-323c052e1731", "$LIBDIR/eigen3", force=true)
end


LIBCUDA = ""
CUDA_INC = ""
if haskey(ENV, "GPU") && ENV["GPU"]=="1" && !(Sys.isapple())
    @info " --------------- Installing GPU Dependencies --------------- "
    try 
        run(`which nvcc`)
    catch
        error("""You specified ENV["GPU"]=1 but nvcc cannot be found (`which nvcc`) failed.
Make sure `nvcc` is available.""")
    end
    s = join(readlines(pipeline(`nvcc --version`)), " ")
    ver = match(r"V(\d+\.\d)", s)[1]
    if ver[1:2]!="10"
        error("TensorFlow backend of ADCME requires CUDA 10.0. But you have CUDA $ver")
    end
    if ver[1:4]!="10.0"
        @warn("TensorFlow is compiled using CUDA 10.0, but you have CUDA $ver. This might cause some problems.")
    end

    if !("adcme-gpu" in Conda._installed_packages(:ADCME))
        Conda.add("adcme-gpu", :ADCME, channel="kailaix")
    end
    
    pkg_dir = joinpath(Conda.ROOTENV, "pkgs/")
    files = readdir(pkg_dir)
    libpath = filter(x->startswith(x, "cudatoolkit") && isdir(joinpath(pkg_dir,x)), files)
    if length(libpath)==0
        @warn "cudatoolkit* not found in $pkg_dir"
    elseif length(libpath)>1
        @warn "more than 1 cudatoolkit found, use $(libpath[1]) by default"
    end

    if length(libpath)>=1
        LIBCUDA = abspath(joinpath(pkg_dir, libpath[1], "lib"))
    end
    

    libpath = filter(x->startswith(x, "cudnn") && isdir(joinpath(pkg_dir,x)), files)
    if length(libpath)==0
        @warn "cudnn* not found in $pkg_dir"
    elseif length(libpath)>1
        @warn "more than 1 cudatoolkit found, use $(libpath[1]) by default"
    end

    if length(libpath)>=1
        LIBCUDA = LIBCUDA*":"*abspath(joinpath(pkg_dir, libpath[1], "lib"))
        @info " --------------- CUDA include headers  --------------- "
        cudnn = joinpath(pkg_dir, libpath[1], "include", "cudnn.h")
        cp(cudnn, joinpath(TF_INC, "cudnn.h"), force=true)
    end

    NVCC = readlines(pipeline(`which nvcc`))[1]
    CUDA_INC = joinpath(splitdir(splitdir(NVCC)[1])[1], "include")

else
    @info " --------------- Skipped: Installing GPU Dependencies  --------------- "
end

@info """ --------------- Write Dependency Files  --------------- """

s = ""
t = []
function adding(k, v)
    global s 
    if Sys.iswindows()
        v = replace(v, "\\"=>"\\\\")
    end
    s *= "$k = \"$v\"\n"
    push!(t, "$k")
end
adding("BINDIR", BINDIR)
adding("LIBDIR", abspath(joinpath(ENVDIR, "lib")))
adding("TF_INC", TF_INC)
adding("TF_ABI", TF_ABI)
adding("EIGEN_INC", LIBDIR)
if Sys.isapple()
    adding("CC", joinpath(BINDIR, "clang"))
    adding("CXX", joinpath(BINDIR, "clang++"))
elseif Sys.islinux()
    adding("CC", joinpath(BINDIR, "x86_64-conda_cos6-linux-gnu-gcc"))
    adding("CXX", joinpath(BINDIR, "x86_64-conda_cos6-linux-gnu-g++"))
else
    adding("CC", joinpath(BINDIR, ""))
    adding("CXX", joinpath(BINDIR, ""))
end
adding("CMAKE", cmake)
adding("MAKE", joinpath(BINDIR, "make"))
adding("GIT", GIT)
adding("PYTHON", PyCall.python)
adding("TF_LIB_FILE", TF_LIB_FILE)
adding("LIBCUDA", LIBCUDA)
adding("CUDA_INC", CUDA_INC)

t = "join(["*join(t, ",")*"], \";\")"
s *= "__STR__ = $t"
open("deps.jl", "w") do io 
    write(io, s)
end

@info """ --------------- Finished: $(abspath("deps.jl"))  --------------- """

