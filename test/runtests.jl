using NLopt
using Optim
using ADCME
using Statistics
using LinearAlgebra
using PyCall
using SparseArrays
using Test
using Random

PIP = get_pip()
run_with_env(`$PIP install matplotlib`)
using PyPlot


if has_gpu()
    use_gpu(0)
end
doctor()
sess = Session()

include("layers.jl")
include("sparse.jl")
include("random.jl")
include("io.jl")
include("variable.jl")
include("ops.jl")
include("core.jl")
include("extra.jl")
include("newton.jl")
include("optim.jl")
include("ot.jl")
include("ode.jl")
include("flow.jl")
include("mpi.jl")
include("toolchain.jl")
include("rbf.jl")
include("pcl.jl")
# The default matplotlib backend does not work for MacOSX
if !Sys.isapple()
    include("kit.jl")
end
