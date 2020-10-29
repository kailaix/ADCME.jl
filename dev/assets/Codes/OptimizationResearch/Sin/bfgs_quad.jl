using Revise 
using ADCME 
using LinearAlgebra
using LineSearches
using JLD2 
using PyPlot 
include("../optimizers.jl")

using Random; Random.seed!(233)

x = LinRange(0, 1, 500)|>Array
y = sin.(10π*x)
z = Variable(rand(length(y)))

loss = sum((z-y)^4)
sess = Session(); init(sess)
angles = Float64[]
losses = Optimize!(sess, loss; optimizer = BFGSOptimizer(), max_num_iter=2000, angles = angles, restart = true)


close("all")
figure(figsize=(10,4))
subplot(121)
semilogy(losses)
subplot(122)
semilogy(angles)
savefig("quad.png")