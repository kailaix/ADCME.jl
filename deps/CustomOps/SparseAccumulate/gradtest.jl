using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

sparse_accumulate = load_op("build/libSparseAccumulate", "sparse_accumulate")
get_sparse_accumulate = load_op("build/libSparseAccumulate", "get_sparse_accumulate")
function clear(n)
    ccall((:initialize_sparse_accumulate, "build/libSparseAccumulate.dylib"), Cvoid, (Cint,), n)
end


# TODO: specify your input parameters
clear(4)
cols = constant([2;3;4], dtype=Int32)
vals = constant([1.;2.;3.])
u1 = sparse_accumulate(constant(1,dtype=Int32),cols,vals)
u2 = sparse_accumulate(constant(2,dtype=Int32),cols,vals)
u3 = sparse_accumulate(constant(2,dtype=Int32),cols,vals)
ii,jj,vv = get_sparse_accumulate()
sess = Session()
init(sess)
run(sess, [u1,u2,u3])
run(sess, [ii, jj, vv])


