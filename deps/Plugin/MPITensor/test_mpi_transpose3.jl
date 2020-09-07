using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using DelimitedFiles
Random.seed!(233)

function mpi_tensor_transpose(row,col,ncol,val,n,rank,nt)
    require_mpi()
    mpi_tensor_transpose_ = load_op_and_grad("./build/libMPITensor.so","mpi_tensor_transpose", multiple=true)
    row,col,ncol,val,n,rank,nt = convert_to_tensor(Any[row,col,ncol,val,n,rank,nt], [Int32,Int32,Int32,Float64,Int64,Int64,Int64])
    indices, vals = mpi_tensor_transpose_(row,col,ncol,val,n,rank,nt)
end

mpi_init()
row = [0;1] .+ mpi_rank()*2
col = [0;1;2;3;4;5;0;1;2;3;4;5]
ncol = [6;6]
val = Array(1:12) .+ 12 * mpi_rank()

n = 2
rank = mpi_rank()
nt = 6



# TODO: specify your input parameters
u = mpi_tensor_transpose(row,col,ncol,val,n,rank,nt)
sess = Session(); init(sess)

indices, vals = run(sess, u)
sleep(mpi_rank())
println("===================================================")
writedlm(stdout, indices)
writedlm(stdout, vals)

mpi_finalize()