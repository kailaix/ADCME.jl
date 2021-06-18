ADCME_DIR = joinpath(homedir(), ".julia/adcme")
if !isdir(ADCME_DIR)
    mkpath(ADCME_DIR)
end

if !isdir(joinpath(ADCME_DIR, "bin"))
    mkpath(joinpath(ADCME_DIR, "bin"))
end

if !islink("$(joinpath(ADCME_DIR, "bin/unzip"))")
    symlink("/usr/bin/unzip", "$(joinpath(ADCME_DIR, "bin/unzip"))")
end


if !islink("$(joinpath(ADCME_DIR, "bin/make"))")
    symlink("/usr/bin/make", "$(joinpath(ADCME_DIR, "bin/make"))")
end

if !islink("$(joinpath(ADCME_DIR, "bin/ninja"))")
    symlink("/usr/loca/bin/ninja", "$(joinpath(ADCME_DIR, "bin/ninja"))")
end


if !islink("$(joinpath(ADCME_DIR, "bin/cmake"))")
    symlink("/usr/bin/cmake", "$(joinpath(ADCME_DIR, "bin/cmake"))")
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