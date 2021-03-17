include("../../ccode/mpiops.jl")
using ADOPT 

mpi_init()

function kappa(x, y)
    return 1+x^2+y^2
end

function rhs(x, y)
    -2*x*(1 - x)*(x^2 + y^2 + 1) + 2*x*(-x*y*(1 - y) + y*(1 - x)*(1 - y)) -
             2*y*(1 - y)*(x^2 + y^2 + 1) + 2*y*(-x*y*(1 - x) + x*(1 - x)*(1 - y))
end

function ufunc(x, y)
    x * (1-x) * y * (1-y)
end

n = Int64(round(sqrt(parse(Int64, ARGS[1]))))
mc = MPIConfig(1800÷n)
global_to_local, local_to_global = dofmap(mc)
X, Y = get_xy(mc)
f_local = rhs.(X, Y)
using Random; Random.seed!(233)
θ = Variable(fc_init([2,20,20,20,1]))
θ_shared = mpi_bcast(θ)
κ_local = abs(fc([X'[:] Y'[:]], [20,20,20,1], θ_shared) + 5.0)|>squeeze
κ_local = reshape(κ_local, (mc.n, mc.n))
u_local = poisson_solver(κ_local, f_local, mc)

loss = sum(mpi_sum((u_local)^2))
g = gradients(loss, θ)


sess = Session(); init(sess)



function calculate_loss(x)
    L = run(sess, loss, θ=>x)
    L 
end

function calculate_gradients(x)
    run(sess, g, θ=>x)
end

x = run(sess, θ)
calculate_loss(x)
@info "HERE"
ccall((:MPITensor_Solve_Timer_SetZero, "../../../../../../../deps/Plugin/MPITensor/build/libMPITensor.so"), Cvoid, ())
stats = @timed begin 
    for i = 1:3
        calculate_loss(x)
    end
end 
t0 = stats[2]/3
s0 = ccall((:MPITensor_Solve_Timer_Get, "../../../../../../../deps/Plugin/MPITensor/build/libMPITensor.so"), Cdouble, ())/3

calculate_gradients(x)
ccall((:MPITensor_Solve_Timer_SetZero, "../../../../../../../deps/Plugin/MPITensor/build/libMPITensor.so"), Cvoid, ())
stats = @timed begin 
    for i = 1:3
        calculate_gradients(x)
    end
end 
t1 = stats[2]/3
s1 = ccall((:MPITensor_Solve_Timer_Get, "../../../../../../../deps/Plugin/MPITensor/build/libMPITensor.so"), Cdouble, ())/3

using DelimitedFiles

if mpi_rank()==0
    i = parse(Int64, ARGS[2])
    open("result.txt", "a") do io 
        writedlm(io, [mpi_size() i t0 t1 s0 s1])
    end 
end 

if mpi_size()>=1
    mpi_finalize()
end