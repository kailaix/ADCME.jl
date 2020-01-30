using PyCall
using Conda
easy_get = x->try; strip(read(pipeline(`which $x`), String)); catch; "";end

tf = pyimport("tensorflow")
lib = readdir(splitdir(tf.__file__)[1])
TF_LIB_FILE = joinpath(splitdir(tf.__file__)[1],lib[findall(occursin.("libtensorflow_framework", lib))[1]])
TF_INC = tf.sysconfig.get_compile_flags()[1][3:end]
TF_ABI = tf.sysconfig.get_compile_flags()[2][end:end]

CC = easy_get("gcc")
CXX = easy_get("g++")
CMAKE = easy_get("cmake")
MAKE = easy_get("make")
GIT = easy_get("git")
PYTHON = easy_get("python")
    
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
adding("CC", CC)
adding("CXX", CXX)
adding("CMAKE", CMAKE)
adding("MAKE", MAKE)
adding("GIT", GIT)
adding("PYTHON", PYTHON)
adding("TF_LIB_FILE", TF_LIB_FILE)

t = "join(["*join(t, ",")*"], \";\")"
s *= "__STR__ = $t"
open("deps.jl", "w") do io 
    write(io, s)
end
