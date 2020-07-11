# mpiexec.exe -n 4 julia .\mpisum.jl
using ADCME

mpi_init()
r = mpi_rank()
a = constant(ones(10) * r)
b = mpi_bcast(a, 3)
L = sum(b^2)
L = mpi_sum(L)
g = gradients(L, a)

sess = Session(); init(sess)
v, G = run(sess, [b, G])

@info r, v


# gradient test


# expected [ Info: (3, [24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0])
G = run(sess, g)

@info r, G

mpi_finalize()