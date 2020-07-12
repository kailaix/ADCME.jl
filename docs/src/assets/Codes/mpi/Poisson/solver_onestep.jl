
include("poisson.jl")
@load  "data.jld2" Us Fs Ss 

mpi_init()

M = 2
N = 2
r = mpi_rank()
s = mpi_size()
@assert s==M*N 
I = div(r, M)
J = r%M


n, m = size(Us[1,1])
h = 1/(m*M+1)
f = constant(Fs[I+1, J+1])
u = constant(Us[I+1, J+1])
o = Ss[I+1, J+1]

out = update_u(u, f)

sess = Session(); init(sess)
@info mpi_rank(), run(sess, out)-o

mpi_finalize()