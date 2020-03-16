push!(LOAD_PATH, "..")
include("../JuliaOpModule.jl")
using ADCME
import JuliaOpModule:do_it, DoIt!
x = constant(rand(100))
y = 2x # or `y = Variable(rand(100))`
u = do_it(y)
sess = Session()
init(sess)
run(sess, u)