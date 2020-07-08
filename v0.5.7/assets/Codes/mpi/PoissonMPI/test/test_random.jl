# mpirun -n 10 julia test_random.jl

using PoissonMPI
using JLD2

MPI.Init()
comm = MPI.COMM_WORLD
k = MPI.Comm_rank(comm)

M = 2
N = 5
if k==0
    MPIBlockDims(M, N, force=true)
end
MPI.Barrier(comm)

MPIBlockDims(M, N)
if k==0
    @time generate_random_test_case(800,800, 1.0)
end
MPI.Barrier(comm)

@load "test.jld2" list 

l = list[k+1]
U = l[1]
F = l[2]
Ref = l[3]

h = 1.0
u = Poisson(U, F,h)
sess = Session(); init(sess)
U = @time run(sess, u)
@show k, maximum(abs.(U - Ref))
# @show k, U
# @show k, Ref