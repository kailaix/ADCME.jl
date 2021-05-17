using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function compute_sin(input)
    compute_sin_ = load_op_and_grad("./build/libComputeSin","compute_sin")
    input = convert_to_tensor(Any[input], [Float64]); input = input[1]
    compute_sin_(input)
end


# TODO: specify your input parameters
input = rand(10)
u = compute_sin(input)

sess = Session(CPU=1); init(sess)
@show run(sess, u)
