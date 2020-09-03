using Revise
using ADCME
using PyPlot

A = rand(10,10)
function hfunc(x)
    A*x
end

θ = ones(10)
obs = hfunc(θ) + randn(10)
θ_est = A\obs 

mc  = MCMCSimple(obs, hfunc, 1.0, θ, 0., 2.,  0.1)
sim = simulate(mc, 20000)
diagnose(mc)
plt.suptitle("Step size = 0.1")

mc  = MCMCSimple(obs, hfunc, 1.0, θ, 0., 2.,  0.01)
sim = simulate(mc, 20000)
diagnose(mc)
plt.suptitle("Step size = 0.01")


mc  = MCMCSimple(obs, hfunc, 1.0, θ, 0., 2.,  1.0)
sim = simulate(mc, 20000)
diagnose(mc)
plt.suptitle("Step size = 1.0")
