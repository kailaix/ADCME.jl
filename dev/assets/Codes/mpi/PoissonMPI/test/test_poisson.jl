# launch the script with
# mpirun -n 10 julia test_poisson.jl 

using PoissonMPI
using JLD2


MPI.Init()
comm = MPI.COMM_WORLD
k = MPI.Comm_rank(comm)

# We divide the mesh into 2x5 blocks, each block consists of a grid of size 800x800
M = 2
N = 5
m = 3000
n = 3000

# Compile a custom operator for the above configuration
if k==0
    MPIBlockDims(M, N)
end
MPI.Barrier(comm)
MPIBlockDims(M, N)


u0 = zeros(m, n) # initial guess
f0 = ones(m, n) # right hand size
h = 0.01 # mesh size 
u = PoissonSolver(u0, f0, h)
sess = Session(); init(sess)
res = @timed run(sess, u)
ccall((:report_time, "../deps/Poisson/build/libPoissonOp"), Cvoid, ())
println("<$(div(k, N)) $(mod(k, N))> Total time = $(res[2])")