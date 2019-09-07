using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

do_it_op = load_op("build/libDoItOp", "do_it_op")
function DoIt!(x::Array{Float64}, y::Array{Float64})
    @show "print from Julia"
    x[:] = y
end

# TODO: specify your input parameters
x = constant(rand(100))
u = do_it_op(x)
sess = Session()
init(sess)
run(sess, u)


