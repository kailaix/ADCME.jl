using Revise 
using ADCME 
using LinearAlgebra
using LineSearches
using JLD2 
include("../optimizers.jl")

using Random; Random.seed!(233)

x = rand(10)
y = sin.(x)
θ = Variable(ae_init([1,20,20,20,1]))
z = squeeze(fc(x, [20, 20, 20, 1], θ))

loss = sum((z-y)^2)
sess = Session(); init(sess)
losses = Optimize!(sess, loss; optimizer = BFGSOptimizer(), max_num_iter=2000)

@save "data/bfgs.jld2" losses 


