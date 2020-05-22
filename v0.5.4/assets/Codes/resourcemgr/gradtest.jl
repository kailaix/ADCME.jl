using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

test_resource_manager = load_op_and_grad("./build/libTestResourceManager","test_resource_manager")
# TODO: specify your input parameters
k = constant(0, dtype=Int32)
u1 = test_resource_manager(k) 
u2 = test_resource_manager(k)
u3 = test_resource_manager(k)

control_dependencies([u1, u2, u3]) do 
    global z = test_resource_manager(k)  # use z to read the summation
end


# z = test_resource_manager(k)
sess = Session(); init(sess)
run(sess, z)

# output
# Create a new container
# Current Value=1
# Current Value=2
# Current Value=3
# Current Value=4
# 4