if VERSION==v"1.4" && Sys.isapple() && !(haskey(ENV, "DOCUMENTER_KEY"))
    error("""Your Julia version is =1.4, and your operation system is MacOSX. 
Currently, there is a compatibility issue for this combination. 
Please downgrade your Julia version.""")
end

if haskey(ENV, "MANUAL") && ENV["MANUAL"]=="1" 
    error("""****** You indicated you want to build ADCME package manually. 
To this end, you need to create a dependency file 
$(joinpath(@__DIR__, "deps.jl"))
and populate it with appropriate binary locations. 
--------------------------------------------------------------------------------------------
BINDIR = 
LIBDIR = 
TF_INC = 
TF_ABI = 
PREFIXDIR = 
CC = 
CXX = 
CMAKE = 
MAKE = 
GIT = 
PYTHON = 
TF_LIB_FILE = 
LIBCUDA = 
CUDA_INC = 
NINJA = 
INCDIR = 
__STR__ = join([BINDIR,LIBDIR,TF_INC,TF_ABI,PREFIXDIR,CC,CXX,CMAKE,MAKE,GIT,PYTHON,TF_LIB_FILE,LIBCUDA,CUDA_INC,NINJA,INCDIR], ";")
--------------------------------------------------------------------------------------------
""")
end

JULIA_ADCME_DIR = homedir()
if haskey(ENV, "JULIA_ADCME_DIR")
    JULIA_ADCME_DIR = abspath(ENV["JULIA_ADCME_DIR"])
    @info "Found JULIA_ADCME_DIR=$JULIA_ADCME_DIR in environment variables. ADCME will install the dependencies to JULIA_ADCME_DIR."
    if !ispath(joinpath(JULIA_ADCME_DIR, ".julia"))
        mkpath(joinpath(JULIA_ADCME_DIR, ".julia"))
    end
end

push!(LOAD_PATH, "@stdlib")
using Pkg
using CMake
using LibGit2

ENVDIR = joinpath(JULIA_ADCME_DIR, ".julia", "adcme")

VER = haskey(Pkg.installed(),"ADCME")  ? Pkg.installed()["ADCME"] : "NOT_INSTALLED"
@info """Your Julia version is $VERSION, current ADCME version is $VER, ADCME dependencies installation path: $ENVDIR"""

@info " --------------- (1/6) Install Tensorflow Dependencies  --------------- "
FORCE_REINSTALL_ADCME = haskey(ENV, "FORCE_REINSTALL_ADCME") && ENV["FORCE_REINSTALL_ADCME"] in [1, "1"]
include("install_adcme.jl")

BINDIR = Sys.iswindows() ? abspath("$ENVDIR/Scripts") : abspath("$ENVDIR/bin")  

GIT = "LibGit2"
PYTHON = joinpath(BINDIR, "python")

if Sys.iswindows()
    PYTHON = abspath(joinpath(ENVDIR, "python.exe"))
end
@info " --------------- (2/6) Check Python Version  --------------- "

!haskey(Pkg.installed(), "PyCall") && Pkg.add("PyCall")
ENV["PYTHON"]=PYTHON
Pkg.build("PyCall")
using PyCall
@info """
PyCall Python version: $(PyCall.python)
Conda Python version: $PYTHON
"""

@info " --------------- (3/6) Looking for TensorFlow Dynamic Libraries --------------- "
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

if Sys.iswindows()
    if !isdir(joinpath(TF_INC, "tensorflow"))
        @info " --------------- (Windows) Downloading Include Files for Custom Operators --------------- "
        run(`cmd /c rmdir /s /q $TF_INC`)
        LibGit2.clone("https://github.com/kailaix/tensorflow-1.15-include", TF_INC)
    end
end

@info " --------------- (4/6) Preparing Custom Operator Environment --------------- "
LIBDIR = abspath("$ENVDIR/lib/Libraries")

if !isdir(LIBDIR)
    @info "Downloading dependencies to $LIBDIR..."
    mkdir(LIBDIR)
end

if !isfile("$LIBDIR/eigen.zip")
    download("https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip","$LIBDIR/eigen.zip")
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
    mv("$LIBDIR/eigen-3.3.7", "$LIBDIR/eigen3", force=true)
end


CONDA = ""
if Sys.iswindows()
    CONDA = "$(JULIA_ADCME_DIR)/.julia/adcme/Scripts/conda.exe"
else 
    CONDA = "$(JULIA_ADCME_DIR)/.julia/adcme/bin/conda"
end

PIP = ""
if Sys.iswindows()
    PIP = "$(JULIA_ADCME_DIR)/.julia/adcme/Scripts/pip.exe"
else 
    PIP = "$(JULIA_ADCME_DIR)/.julia/adcme/bin/pip"
end

# install matplotlib 
if !occursin("matplotlib", read(`$PIP list`, String))
    run(`$PIP install matplotlib`)
end

# If the system has `nvcc` but "GPU" is not specified, warn the users to build with 
# ENV["GPU"] = 1
if !haskey(ENV, "GPU")
    try 
        if Sys.iswindows()
            run(`cmd /c nvcc --version`)
        else
            run(`which nvcc`)
        end
        @warn("""We detected that you have `nvcc` installed but ENV[\"GPU\"] is not set. 
>>> If you want to install ADCME with GPU capabiity enabled, please set `ENV[\"GPU\"]=1`.""")
    catch
    end
end 

LIBCUDA = ""
CUDA_INC = ""
if Sys.islinux() && haskey(ENV, "GPU") && ENV["GPU"] in ["1", 1]
    @info " --------------- (5/6) Installing GPU Dependencies --------------- "
    
    NVCC = readlines(pipeline(`which nvcc`))[1]
    s = join(readlines(pipeline(`nvcc --version`)), " ")
    ver = match(r"V(\d+\.\d)", s)[1]
    if ver[1:2]!="10"
        error("TensorFlow backend of ADCME requires CUDA 10.0. But you have CUDA $ver")
    end
    if ver[1:4]!="10.0"
        @warn("TensorFlow is compiled using CUDA 10.0, but you have CUDA $ver. This might cause some problems.")
    end
    
    pkg_dir = "$(JULIA_ADCME_DIR)/.julia/adcme/pkgs/"
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
    
    
    libcudatoolkit_path = filter(x->startswith(x, "cudnn") && isdir(joinpath(pkg_dir,x)), files)
    if length(libcudatoolkit_path)==0
        @warn "cudnn* not found in $pkg_dir"
    elseif length(libcudatoolkit_path)>1
        @warn "more than 1 cudatoolkit found, use $(libpath[1]) by default"
    end
    
    if length(libcudatoolkit_path)>=1
        LIBCUDA = LIBCUDA*":"*abspath(joinpath(pkg_dir, libcudatoolkit_path[1], "lib"))
        @info " --------------- CUDA include headers  --------------- "
        cudnn = joinpath(pkg_dir, libcudatoolkit_path[1], "include", "cudnn.h")
        cp(cudnn, joinpath(TF_INC, "cudnn.h"), force=true)
    end
    
    CUDA_INC = joinpath(splitdir(splitdir(NVCC)[1])[1], "include")    

else
    @info " --------------- (5/6) Skipped: Installing GPU Dependencies  --------------- "
end

@info """ --------------- (6/6) Write Dependency Files  --------------- """

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
if Sys.iswindows()
    D = abspath(joinpath(LIBDIR, "lib"))
    !isdir(D) && mkdir(D)
    adding("LIBDIR", D)
else
    adding("LIBDIR", abspath(joinpath(ENVDIR, "lib")))
end
adding("TF_INC", TF_INC)
adding("TF_ABI", TF_ABI)
adding("PREFIXDIR", LIBDIR)
if Sys.isapple()
    adding("CC", joinpath(BINDIR, "clang"))
    adding("CXX", joinpath(BINDIR, "clang++"))
elseif Sys.islinux()
    adding("CC", joinpath(BINDIR, "x86_64-conda_cos6-linux-gnu-gcc"))
    adding("CXX", joinpath(BINDIR, "x86_64-conda_cos6-linux-gnu-g++"))
else
    adding("CC", "")
    adding("CXX", "")
end
if Sys.islinux()
    adding("CMAKE", joinpath(BINDIR, "cmake"))
else
    adding("CMAKE", cmake)
end
if Sys.iswindows()
    adding("MAKE", "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\MSBuild\\15.0\\Bin\\MSBuild.exe")
else 
    adding("MAKE", joinpath(BINDIR, "make"))
end 
adding("GIT", GIT)
adding("PYTHON", PyCall.python)
adding("TF_LIB_FILE", TF_LIB_FILE)
adding("LIBCUDA", LIBCUDA)
adding("CUDA_INC", CUDA_INC)
if Sys.iswindows()
    adding("NINJA", "")
else
    adding("NINJA", joinpath(BINDIR, "ninja"))
end
adding("INCDIR", abspath(joinpath(BINDIR, "..", "include")))

t = "join(["*join(t, ",")*"], \";\")"
s *= "__STR__ = $t"

open("deps.jl", "w") do io 
    write(io, s)
end

@info """ --------------- Finished: $(abspath("deps.jl"))  --------------- """

