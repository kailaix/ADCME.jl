using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using ForwardDiff
Random.seed!(233)

function twolayer(y, x, w1, w2, b1, b2)
    f = x -> begin
        w1 = reshape(w1, 10, 10)
        w2 = reshape(w2, 10, 10)
        z = w2*tanh.(w1*x+b1)+b2
    end
    y[:] = ForwardDiff.jacobian(f, x)[:]
end
two_layer = load_op("build/libTwoLayer", "two_layer")


w1 = rand(100)
w2 = rand(100)
b1 = rand(10)
b2 = rand(10)
x = rand(10)
J = rand(100)
twolayer(J, x, w1, w2, b1, b2)

y = two_layer(constant(x), constant(w1), constant(b1), constant(w2), constant(b2))
sess = Session(); init(sess)
J0 = run(sess, y)
@show norm(J-J0)