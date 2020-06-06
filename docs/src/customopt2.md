# Custom Optimizers

ADCME provides a function [`UnconstrainedOptimizer`](@ref) that allows users to craft their own optimizers for unconstrained optimization problems. 

```julia
using Optim 

x = Variable(rand(2))
loss = (1-x[1])^2 + 100(x[2]-x[1]^2)^2

sess = Session(); init(sess)
uo = UnconstrainedOptimizer(sess, loss)

function f(x)
    return getLoss(uo, x)
end

function g!(G, x)
    _, g = getLossAndGrad(uo, x)
    G[:] = g  
end

x0 = getInit(uo)
optimize(f, g!, x0, LBFGS())
```