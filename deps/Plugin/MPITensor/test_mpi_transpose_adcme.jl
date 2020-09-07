using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using DelimitedFiles
using SparseArrays
Random.seed!(233)


mpi_init()
sp = sprand(10,10,0.3)
SP = mpi_SparseTensor(sp)
SPt = SP'
sess = Session(); init(sess)
Array(run(sess, SP))-Array(run(sess, SPt))'