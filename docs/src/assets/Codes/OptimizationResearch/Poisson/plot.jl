using ADCME
using PyPlot 
using JLD2
using AdFem
using ADCME
using PyPlot 
using JLD2
using Statistics 

function kappa(x, y)
    return 2 + exp(10x) - (10y)^2
end

function f(x, y)
    return sin(2π*10y+π/8)
end

mmesh = Mesh(joinpath(PDATA, "twoholes_large.stl"))

Kappa = eval_f_on_gauss_pts(kappa, mmesh)
xy = gauss_nodes(mmesh)

sess = Session()
MSE = Float64[]

# adam 
@load "data/adam.jld2" losses w
loss1 = losses
KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))

@load "data/bfgs_with_adam_hessian.jld2" losses w
loss2 = losses
KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))


@load "data/bfgs_with_adam_nohessian.jld2" losses w
loss3 = losses
KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))

@load "data/bfgs_with_noadam.jld2" losses w
loss4 = losses
KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))

@load "data/lbfgs_adam.jld2" losses w
loss5 = losses
KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))

@load "data/lbfgs_noadam.jld2" losses w
loss6 = losses
KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))


MSE = round.(MSE, sigdigits=2)

close("all")
semilogy(loss1, label="Adam")
semilogy(loss2, label="BFGS+Adam+Hessian")
semilogy(loss3, label="BFGS+Adam")
semilogy(loss4, label="BFGS")
semilogy(loss5, label="LBFGS+Adam")
semilogy(loss6, label="LBFGS")
legend()
xlabel("Iterations")
ylabel("Loss")
savefig("data/loss300.png")


close("all")
# semilogy(loss1, label="Adam")
# semilogy(loss2, label="BFGS+Adam+Hessian")
# semilogy(loss3, label="BFGS+Adam")
semilogy(loss4, label="BFGS")
# semilogy(loss5, label="LBFGS+Adam")
semilogy(loss6, label="LBFGS")
xlim(0, 20)
legend()
xlabel("Iterations")
ylabel("Loss")
tight_layout()
savefig("data/loss300_zoom.png")
