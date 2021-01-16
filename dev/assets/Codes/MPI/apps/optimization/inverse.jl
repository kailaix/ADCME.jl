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

mc = MPIConfig(300)
global_to_local, local_to_global = dofmap(mc)
X, Y = get_xy(mc)
f_local = rhs.(X, Y)
using Random; Random.seed!(233)
θ = Variable(fc_init([2,20,20,20,1]))
θ_shared = mpi_bcast(θ)
κ_local = abs(fc([X'[:] Y'[:]], [20,20,20,1], θ_shared) + 5.0)|>squeeze
κ_local = reshape(κ_local, (mc.n, mc.n))
u_local = poisson_solver(κ_local, f_local, mc)

@load "data/$(mpi_size())_$(mpi_rank()).jld2" U 

loss = sum(mpi_sum((u_local - U)^2))
g = gradients(loss, θ)


sess = Session(); init(sess)


L = run(sess, loss)

if mpi_rank()==0
    @info "Initial loss = $L"
end 

function calculate_loss(x)
    L = run(sess, loss, θ=>x)
    L 
end

function calculate_gradients(G, x)
    G[:] = run(sess, g, θ=>x)
end


losses  = Float64[]
function step_callback(x)
    @info "Loss = $x"
    push!(losses, x)
end


initial_x = run(sess, θ)
options = Options()
result = ADOPT.mpi_optimize(calculate_loss, calculate_gradients, initial_x, LBFGS(), options; step_callback = step_callback)

if mpi_rank()==0
    minimizer = result.minimizer
    @save "result.jld2" result losses
end

if mpi_size()>1
    mpi_finalize()
end