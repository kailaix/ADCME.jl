using Revise
using ADCME
using PyCall
using PyPlot
include("moon.jl")

function sample_moons(n)
    X, _ = nmoons(Float64, n, 2, ε=0.3, d=2, translation=[0.25, -0.25])
    return Array(X')
end

sklearn = pyimport("sklearn")


x = sample_moons(128)
scatter(x[:,1],x[:,2])
axis("equal")

flows = [AffineHalfFlow(2, mod(i,2)==1) for i = 0:8]
model = NormalizingFlowModel(prior, flows)