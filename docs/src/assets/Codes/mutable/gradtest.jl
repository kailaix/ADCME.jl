using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

my_assign = load_op("./build/libMyAssign","my_assign")

# TODO: specify your input parameters
u = Variable([0.1,0.2,0.3])
v = constant(Array{Float64}(1:3))
u2 = u^2
w = my_assign(u,v)
sess = tf.Session()
init(sess)
@show run(sess, u)
@show run(sess, u2)
@show run(sess, w)
@show run(sess, u2)
