using ADCME
using Test
using PyCall
sess = Session()

include("io.jl")
include("variable.jl")
include("ops.jl")
include("core.jl")
include("extra.jl")
include("RBF.jl")


