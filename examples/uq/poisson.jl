using Revise
using ADCME
using PyPlot

function solve_poisson(μ)
    [μ;μ;μ]
end

udata = randn(100,3)*5 .+ 10.0
μm = Variable(1.0)
μc = Variable(1.0)
μ = UQNode(μm, μc)
u = UQOp(solve_poisson)(μ)
μ0, C0 = ml_estimator(udata)
loss = sum((u.loc-μ0)^2) + sum((u.cov-C0)^2)

sess = Session(); init(sess)
ADCME.BFGS!(sess, loss, 100)
@info run(sess, [μm, μc])

# visualize
s = sample(umid, 1000)
samples = run(sess, s)
hist(udata)
hist(samples)