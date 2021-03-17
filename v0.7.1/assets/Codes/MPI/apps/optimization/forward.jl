include("../../ccode/mpiops.jl")

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

mc = MPIConfig(300)
global_to_local, local_to_global = dofmap(mc)
X, Y = get_xy(mc)
f_local = rhs.(X, Y)
# using Random; Random.seed!(233)
# θ = Variable(fc_init([2,20,20,20,1]))
# θ_shared = mpi_bcast(θ)
# κ_local = (fc([X'[:] Y'[:]], [20,20,20,1], θ_shared) + 5.0)|>squeeze
# κ_local = reshape(κ_local, (mc.n, mc.n))
κ_local = kappa.(X, Y)
u_local = poisson_solver(κ_local, f_local, mc)


sess = Session(); init(sess)
U = run(sess, u_local)


# change_directory("data")
@save "data/$(mpi_size())_$(mpi_rank()).jld2" U 


if mpi_size()>1
    mpi_finalize()
end