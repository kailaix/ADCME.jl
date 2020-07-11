# mpiexec.exe -n 4 julia .\mpisum.jl
using ADCME

mpi_init()
r = mpi_rank()
a = constant(Float64.(Array(1:10) * r))
b = mpi_sum(a)

L = sum(b)
g = gradients(L, a)
sess = Session(); init(sess)
v, G = run(sess, [b,g])


@info r, G
if r==0
    ref = zeros(10)
    for k = 0:mpi_size()-1
        global ref
        ref += Array(1:10)*k 
    end
    @info v, ref
end

mpi_finalize()