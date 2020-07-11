# mpiexec.exe -n 4 julia .\mpisum.jl
using ADCME

mpi_init()
r = mpi_rank()
a = constant(ones(10) * r)
a = mpi_sendrecv(a, 0, 2)

# Equivalently, we can use 
# if r==2
#     global a
#     a = mpi_send(a, 0)
# end
# if r==0
#     global a
#     a = mpi_recv(a,2)
# end

L = sum(a^2)
g = gradients(L, a)

sess = Session(); init(sess)
v, G = run(sess, [a,g])

# processor 0 should have the same `v` as processor 2
@info r, v


G = run(sess, g)

# gradients on processor 0 should be the same as processor 2 
# because `a` is received from 2
@info r, G

mpi_finalize()