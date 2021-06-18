ADCME_DIR = joinpath(homedir(), ".julia/adcme")
if !isdir(ADCME_DIR)
    mkpath(ADCME_DIR)
end

using Pkg 
using PyCall
push!(LOAD_PATH, "@stdlib")
if PyCall.python!="/usr/local/bin/python"
    ENV["PYTHON"] = "/usr/local/bin/python"
    Pkg.build("PyCall")
end

deps = """
BINDIR = "$(joinpath(ADCME_DIR, "bin"))"
LIBDIR = "$(joinpath(ADCME_DIR, "lib"))"
TF_INC = "/usr/local/lib/python3.6/dist-packages/tensorflow_core/include"
TF_ABI = "0"
PREFIXDIR = "$(joinpath(ADCME_DIR, "lib/Libraries"))"
CC = "/usr/bin/gcc"
CXX = "/usr/bin/g++"
CMAKE = "/usr/bin/cmake"
MAKE = "/usr/bin/make"
GIT = "LibGit2"
PYTHON = "/usr/local/bin/python"
TF_LIB_FILE = "/usr/local/lib/python3.6/dist-packages/tensorflow_core/libtensorflow_framework.so.1"
LIBCUDA = ""
CUDA_INC = ""
NINJA = "/usr/bin/ninja"
INCDIR = "$(joinpath(ADCME_DIR, "include"))"
__STR__ = join([BINDIR,LIBDIR,TF_INC,TF_ABI,PREFIXDIR,CC,CXX,CMAKE,MAKE,GIT,PYTHON,TF_LIB_FILE,LIBCUDA,CUDA_INC,NINJA,INCDIR], ";")
"""

open("deps.jl", "w") do io 
    write(io, deps)
end