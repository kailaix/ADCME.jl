using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using DelimitedFiles
Random.seed!(233)


function halo_exchange_two_d(u,fill_value,m,n)
    halo_exchange_two_d_ = load_op_and_grad("./build/libHaloExchangeTwoD","halo_exchange_two_d")
    u,fill_value,m,n = convert_to_tensor(Any[u,fill_value,m,n], [Float64,Float64,Int64,Int64])
    halo_exchange_two_d_(u,fill_value,m,n)
end

mpi_init()
U = reshape(1:24, 4, 6)'|>Array

m = 3
n = 2
fill_value = 1.0

M = mpi_rank()÷n+1
N = mpi_rank()%n+1
ulocal = U[(M-1)*2 + 1: M * 2, (N-1)*2+1:N*2]
# TODO: specify your input parameters
u = halo_exchange_two_d(ulocal,fill_value,m,n)
sess = Session(); init(sess)
Uval = run(sess, u)

sleep(mpi_rank())
println("=========================== rank = $(mpi_rank()) ======================")
writedlm(stdout, Uval)
mpi_finalize()

