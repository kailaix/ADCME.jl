
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
u = poisson_solver(f, 100)

sess = Session(); init(sess)
@info "Start running..."
result = run(sess, u[end])

@save "solution$I$J.jld2" result

mpi_finalize()