using ADCME
using JLD2
using Optim
using Random; Random.seed!(233)

x = rand(10)
y = sin.(x)
θ = Variable(ae_init([1,20,20,20,1]))
z = squeeze(fc(x, [20, 20, 20, 1], θ))
loss = sum((z-y)^2)

opt = AdamOptimizer().minimize(loss)
sess = Session(); init(sess)

losses = Float64[]
for i = 1:2000
    _, l = run(sess, [opt, loss])
    push!(losses, l )
end
@save "data/adam.jld2" losses 
