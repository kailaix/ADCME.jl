using PyPlot 
using ADCME
using JLD2

include("inverse.jl")
E0 = eval_f_on_gauss_pts(f, mmesh)
MSE = Float64[]

sess = Session(); init(sess)

close("all")
@load "data/resultadam/data.jld2" losses THETA1
E1 = run(sess, E, θ1=>THETA1)
push!(MSE, mean((E1 - E0).^2))
semilogy(losses, label = "Adam")

@load "data/resultbfgs_adam_hessian/data.jld2" losses THETA1
E1 = run(sess, E, θ1=>THETA1)
push!(MSE, mean((E1 - E0).^2))
semilogy(losses, label = "BFGS+Adam+Hessian")

@load "data/resultbfgs_adam_nohessian/data.jld2" losses THETA1
E1 = run(sess, E, θ1=>THETA1)
push!(MSE, mean((E1 - E0).^2))
semilogy(losses, label = "BFGS+Adam")

@load "data/resultbfgs_noadam/data.jld2" losses THETA1
E1 = run(sess, E, θ1=>THETA1)
push!(MSE, mean((E1 - E0).^2))
semilogy(losses, label = "BFGS")

@load "data/resultlbfgs_adam/data.jld2" losses THETA1
E1 = run(sess, E, θ1=>THETA1)
push!(MSE, mean((E1 - E0).^2))
semilogy(losses, label = "LBFGS+Adam")

@load "data/resultlbfgs/data.jld2" losses THETA1
E1 = run(sess, E, θ1=>THETA1)
push!(MSE, mean((E1 - E0).^2))
semilogy(losses, label = "LBFGS")

xlabel("Iterations")
ylabel("Loss")
legend()
savefig("data/loss_le.png")



@load "data/resultbfgs_adam_hessian/data.jld2" THETA1
E1 = run(sess, E, θ1=>THETA1)
@load "data/resultadam/data.jld2" THETA1
E2 = run(sess, E, θ1=>THETA1)


MSE = round.(MSE, sigdigits=2)

close("all")
figure(figsize=(10,10))
subplot(221)
visualize_scalar_on_gauss_points(E1, mmesh, vmin=1, vmax=3)
title("BFGS+Adam+Hessian")
subplot(222)
visualize_scalar_on_gauss_points(abs.(E1-E0), mmesh)
title("Error")
subplot(223)
visualize_scalar_on_gauss_points(E2, mmesh, vmin=1, vmax=3)
title("Adam")
subplot(224)
visualize_scalar_on_gauss_points(abs.(E2-E0), mmesh)
title("Error")
savefig("data/compare_le.png")