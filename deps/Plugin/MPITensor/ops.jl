using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using SparseArrays
Random.seed!(233)

function mpi_create_matrix(indices,values,ilower,iupper)
    mpi_create_matrix_ = load_op_and_grad("./build/libMPITensor","mpi_create_matrix", multiple=true)
    indices,values,ilower,iupper = convert_to_tensor(Any[indices,values,ilower,iupper], [Int64,Float64,Int64,Int64])
    mpi_create_matrix_(indices,values,ilower,iupper)
end

function mpi_get_matrix(rows,ncols,cols,ilower,iupper,values, N)
    mpi_get_matrix_ = load_op_and_grad("./build/libMPITensor","mpi_get_matrix", multiple=true)
    rows,ncols,cols,ilower_,iupper_,values = convert_to_tensor(Any[rows,ncols,cols,ilower,iupper,values], [Int32,Int32,Int32,Int64,Int64,Float64])
    indices, vals = mpi_get_matrix_(rows,ncols,cols,ilower_,iupper_,values)
    SparseTensor(tf.SparseTensor(indices, vals, (iupper-ilower+1, N)), false)
end


function mpi_tensor_solve(rows,ncols,cols,values,rhs,ilower,iupper,solver = "BoomerAMG",printlevel = 2)
    mpi_tensor_solve_ = load_op_and_grad("./build/libMPITensor","mpi_tensor_solve")
    rows,ncols,cols,values,rhs,ilower,iupper,printlevel = convert_to_tensor(Any[rows,ncols,cols,values,rhs,ilower,iupper,printlevel], [Int32,Int32,Int32,Float64,Float64,Int64,Int64,Int64])
    mpi_tensor_solve_(rows,ncols,cols,values,rhs,ilower,iupper,solver,printlevel)
end