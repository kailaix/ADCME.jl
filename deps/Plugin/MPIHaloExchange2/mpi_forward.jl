using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using DelimitedFiles
using Random
Random.seed!(233)

function halo_exchange_neighbor_two(u,fill_value,m,n,tag,w)
    halo_exchange_neighbor_two_ = load_op_and_grad("./build/libHaloExchangeNeighborTwo","halo_exchange_neighbor_two")
    u,fill_value,m,n,tag,w = convert_to_tensor(Any[u,fill_value,m,n,tag,w], [Float64,Float64,Int64,Int64,Int64,Float64])
    halo_exchange_neighbor_two_(u,fill_value,m,n,tag,w)
end

mpi_init()
u = reshape(Array(1:36), 6, 6)|>Array

if mpi_rank()==0
    u = u[1:3,1:3]
end
if mpi_rank()==1
    u = u[1:3,4:end]
end

if mpi_rank()==2
    u = u[4:end,1:3]
end
if mpi_rank()==3
    u = u[4:end,4:end]
end

fill_value = 10.0
m = 2
n = 2
tag = 1
w = 1.0
# TODO: specify your input parameters
uext = halo_exchange_neighbor_two(u,fill_value,m,n,tag,w)
sess = Session(); init(sess)
uval =  run(sess, uext)

sleep(mpi_rank())
@info "rank = $(mpi_rank())===================================================="
writedlm(stdout, round.(uval, digits=3))
# uncomment it for testing gradients

mpi_finalize()