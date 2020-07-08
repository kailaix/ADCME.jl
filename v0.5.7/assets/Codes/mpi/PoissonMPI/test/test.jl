# mpirun -n 2 julia test.jl
using PoissonMPI
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

MPI.Init()
m = 10
n = 20
u = ones(m, n)
f = zeros(m, n)
h = 1.0

# TODO: specify your input parameters
u = Poisson(u,f,h)
sess = Session(); init(sess)
@show run(sess, u)
