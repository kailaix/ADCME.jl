using LinearAlgebra
using ADCME

n = 101 # number of grid nodes in [0,1]
h = 1/(n-1)
x = LinRange(0,1,n)[2:end-1]

b = Variable(10.0) # create a Variable for `b`
A = diagm(0=>2/h^2*ones(n-2), -1=>-1/h^2*ones(n-3), 1=>-1/h^2*ones(n-3)) # discrete Laplacian matrix
B = b*A + diagm(0=>ones(n-2))  # coefficient matrix
f = @. 4*(2 + x - x^2) # right hand side
u = B\f # solve the equation
ue = u[div(n+1,2)] # extract values at x=0.5

loss = (ue-1.0)^2 # form the loss function

# Optimization
opt = ScipyOptimizerInterface(loss)
sess = Session(); init(sess)
ScipyOptimizerMinimize(sess, opt)

println("Estimated b = ", run(sess, b))