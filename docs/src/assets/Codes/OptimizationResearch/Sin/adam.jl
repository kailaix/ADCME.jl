# julia adam.jl &
# julia bfgs.jl &
# julia lbfgs.jl &
# wait 
# 

using ADCME
using JLD2
using Optim
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

losses = Float64[]
for i = 1:10000
    _, l = run(sess, [opt, loss])
    push!(losses, l )
    if mod(i, 1000)==1
        @info i 
    end
end
close("all")
plot(x, run(sess, z), label = "Adam")
plot(x, y, "--", label = "Reference")
xlabel("x"); ylabel("y")
legend()
savefig("data/adam.png")
@save "data/adam.jld2" losses 
