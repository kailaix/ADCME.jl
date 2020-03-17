push!(LOAD_PATH, "..")
include("../JuliaOpModule.jl")
using ADCME
import JuliaOpModule:do_it, DoIt!

x = placeholder(rand(100))
y = 2x # or `y = Variable(rand(100))`
u = do_it(y)
sess = tf.Session()
init(sess)
run(sess, u, x=>rand(100))
