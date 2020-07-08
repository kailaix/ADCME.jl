using MPI 

MPI.Init()

v = ccall((:printinfo, "./build/libmtest"), Cint, ())

if MPI.Comm_rank(MPI.COMM_WORLD)==0
    println(v)
end

