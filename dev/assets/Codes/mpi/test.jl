using MPI 

MPI.Init()

v = ccall((:printinfo, "./build/Release/mtest.dll"), Cint, ())
print(v)

