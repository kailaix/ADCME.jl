using ADCME
using PyCall
using LinearAlgebra
using PyPlot
if !(@isdefined initialized)
py"""
import tensorflow as tf
from tensorflow.python.framework import ops
libnonlinear = tf.load_op_library('build/libnonlinear.dylib')
@ops.RegisterGradient("NonLinear")
def _gradcc(op, grad):
    return libnonlinear.non_linear_grad(grad, *op.inputs)
"""
    global initialized = true
end

non_linear = py"libnonlinear.non_linear"

v = constant(rand(10))
w = constant(rand(10,10))
u = non_linear(v, w)
sess = Session()
init(sess)
run(sess, u)
err = maximum(abs.(run(sess, u^3-w*v^2)))
println(err)

# gradient check -- v
function scalar_function(v, w)
    return sum(tanh(non_linear(v, w)))
end

m = constant(rand(20))
v = constant(rand(20))
x = constant(rand(20,20))
y = scalar_function(m, x)
dy = gradients(y, m)
ms = Array{Any}(undef, 5)
ys = Array{Any}(undef, 5)
s = Array{Any}(undef, 5)
w = Array{Any}(undef, 5)
gs =  @. 1 / 10^(1:5)

for i = 1:5
    g = gs[i]
    ms[i] = m + g*v
    ys[i] = scalar_function(ms[i], x)
    s[i] = ys[i] - y 
    w[i] = s[i] - g*sum(v.*dy)
end

sess = Session()
init(sess)
sval = run(sess, s)
wval = run(sess, w)
loglog(gs, abs.(sval), "*-", label="finite difference")
loglog(gs, abs.(wval), "+-", label="automatic differentiation")
loglog(gs, gs.^2 * 0.5*abs(wval[1])/gs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs, gs * 0.5*abs(sval[1])/gs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")


# gradient check -- w
function scalar_function(w, v)
    return sum(tanh(non_linear(v, w)))
end

m = constant(rand(20,20))
v = constant(rand(20,20))
x = constant(rand(20))
y = scalar_function(m, x)
dy = gradients(y, m)
ms = Array{Any}(undef, 5)
ys = Array{Any}(undef, 5)
s = Array{Any}(undef, 5)
w = Array{Any}(undef, 5)
gs =  @. 1 / 10^(1:5)

for i = 1:5
    g = gs[i]
    ms[i] = m + g*v
    ys[i] = scalar_function(ms[i], x)
    s[i] = ys[i] - y 
    w[i] = s[i] - g*sum(v.*dy)
end

sess = Session()
init(sess)
sval = run(sess, s)
wval = run(sess, w)
figure()
loglog(gs, abs.(sval), "*-", label="finite difference")
loglog(gs, abs.(wval), "+-", label="automatic differentiation")
loglog(gs, gs.^2 * 0.5*abs(wval[1])/gs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs, gs * 0.5*abs(sval[1])/gs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")