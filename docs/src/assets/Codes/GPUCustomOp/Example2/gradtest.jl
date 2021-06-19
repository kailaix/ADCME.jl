using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function mat_vec(a,x)
    mat_vec_ = load_op_and_grad("./build/libMatVec","mat_vec")
    a,x = convert_to_tensor(Any[a,x], [Float64,Float64])
    mat_vec_(a,x)
end

# TODO: specify your input parameters
a0 = rand(10,10)
x0 = rand(10)
a = constant(a0)
u = mat_vec(a0,x0)
sess = Session(); init(sess)
@show run(sess, u)≈a0 * x0 
