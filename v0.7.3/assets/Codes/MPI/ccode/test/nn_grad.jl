include("../mpiops.jl")

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

mc = MPIConfig(50)
global_to_local, local_to_global = dofmap(mc)
X, Y = get_xy(mc)
f_local = rhs.(X, Y)
using Random; Random.seed!(233)
θ = Variable(fc_init([2,20,20,20,1]))
θ_shared = mpi_bcast(θ)
κ_local = (fc([X'[:] Y'[:]], [20,20,20,1], θ_shared) + 5.0)|>squeeze
κ_local = reshape(κ_local, (mc.n, mc.n))
u_local = poisson_solver(κ_local, f_local, mc)
u = mpi_gather(u_local)[global_to_local]

loss = sum(u^2)
g = gradients(loss, θ)

sess = Session(); init(sess)

function calculate_loss(x)
    run(sess, loss, θ=>x)
end

function calculate_gradients(x)
    run(sess, g, θ=>x)
end

f(x) = (calculate_loss(x), calculate_gradients(x))

test_gradients(f, run(sess, θ), mpi=true)

if mpi_size()>1
    mpi_finalize()
end