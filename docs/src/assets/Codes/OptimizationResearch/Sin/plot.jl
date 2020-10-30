using PyPlot 
using JLD2

close("all")
@load "data/adam.jld2" losses
semilogy(losses, label = "Adam")


@load "data/bfgs.jld2" losses
semilogy(losses, label = "BFGS")

@load "data/lbfgs.jld2" losses
semilogy(losses, label = "LBFGS")


@load "data/dampled_bfgs.jld2" losses
semilogy(losses, label = "Damped BFGS")

legend()
xlabel("Iterations")
ylabel("Loss")
savefig("data/sinloss.png")
