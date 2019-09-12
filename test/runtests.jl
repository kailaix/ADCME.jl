push!(LOAD_PATH, "@stdlib")
using Pkg
Pkg.add("NLopt"); Pkg.add("Optim")

using NLopt
using Optim
using ADCME
using Statistics
using LinearAlgebra
using PyCall
using SparseArrays
using Test



sess = Session()


include("sparse.jl")
include("random.jl")
include("io.jl")
include("variable.jl")
include("ops.jl")
include("core.jl")
include("extra.jl")
include("RBF.jl")
include("optim.jl")
