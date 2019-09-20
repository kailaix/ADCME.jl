using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

set_tensor_flow_timer = load_op("build/libTensorFlowTimer", "set_tensor_flow_timer")
get_tensor_flow_timer = load_op("build/libTensorFlowTimer", "get_tensor_flow_timer")

################## End Load Operator ##################

# TODO: specify your input parameters
i = constant(1, dtype=Int32)
A = constant(rand(2000,2000))
u = set_tensor_flow_timer(i)
v = get_tensor_flow_timer(i)
A = bind(A, u)
A, B, C = tf.linalg.svd(A)
A = bind(A, v)
sess = Session()
init(sess)
run(sess, A)
