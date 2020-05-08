using NLopt
using Optim
using ADCME
using Statistics
using LinearAlgebra
using PyCall
using SparseArrays
using Test
using Random

doctor()
clean()
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