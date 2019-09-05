using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using Cxx
Random.seed!(233)

do_it_op = load_op("build/libDoItOp", "do_it_op")
function DoIt!(x::Array{Float64}, y::Array{Float64})
    println(y)
    x[:] = y
end

function printjulia(p::Int64)
    wait(_cpp_async_cond)
    # p = unsafe_load(Ptr{Float64}(Ptr{Cvoid}(p)))
    # @show p
    # ccall((:Cfunction, "build/libDoItOp.dylib"), Cvoid, ())
    println("JULIA WORKS")
end

# cxx"""
# #include <mutex>
# #include <thread>
# #include <iostream>

# std::mutex mu;
# void foo()
# {
    
#     auto id = std::this_thread::get_id();
 
#     mu.lock();
#     std::cout << "thread " << id << " sleeping...\n";
#     mu.unlock();
 
# }
# """
# @cxx foo()

# TODO: specify your input parameters
# session_conf = tf.ConfigProto(
#       intra_op_parallelism_threads=8,
#       inter_op_parallelism_threads=8)
x = Variable(ones(100))
u = do_it_op(x)
sess = Session()
init(sess)
run(sess, u);


