using ADCME
using JLD2
using PyPlot 

@load "data/bfgs.jld2" losses
losses1 = losses

@load "data/lbfgs.jld2" losses
losses2 = losses

semilogy(losses1, label="BFGS")
semilogy(losses2, label="LBFGS")
legend()
savefig("data/OR_bl.png")