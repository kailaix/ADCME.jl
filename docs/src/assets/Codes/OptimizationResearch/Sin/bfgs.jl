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
θ = Variable(ae_init([1,20,20,20,1]))
z = squeeze(fc(x, [20, 20, 20, 1], θ))

loss = sum((z-y)^2)
opt = AdamOptimizer().minimize(loss)
sess = Session(); init(sess)
angles = Float64[]
losses = Optimize!(sess, loss; optimizer = BFGSOptimizer(), max_num_iter=1000, angles = angles, restart = true)

for i = 1:1000
    _, l = run(sess, [opt, loss])
    push!(losses, l)
end

close("all")
plot(x, run(sess, z), label = "Adam")
plot(x, y, "--", label = "Reference")
xlabel("x"); ylabel("y")
legend()
savefig("data/bfgs.png")

@save "data/bfgs.jld2" losses 


close("all")
figure(figsize=(10,4))
subplot(121)
semilogy(losses)
subplot(122)
semilogy(angles)
savefig("diag.png")