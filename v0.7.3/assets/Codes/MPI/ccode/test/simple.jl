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
κ_local = kappa.(X, Y)
u_local = poisson_solver(κ_local, f_local, mc)
u = mpi_gather(u_local)[global_to_local]

Uexact = reshape(constant(ufunc.(X, Y)), (-1,))
Uexact = mpi_gather(Uexact)[global_to_local]

sess = Session(); init(sess)


U = run(sess, u)
Uexact = run(sess, Uexact)


if mpi_rank()==0
    close("all")
    figure(figsize = (15,4))
    subplot(131)
    pcolormesh(reshape(U, mc.n * mc.N, mc.n * mc.N))
    colorbar()
    subplot(132)
    pcolormesh(reshape(Uexact, mc.n * mc.N, mc.n * mc.N))
    colorbar()
    subplot(133)
    pcolormesh(reshape(abs.(U - Uexact), mc.n * mc.N, mc.n * mc.N))
    colorbar()
    savefig("poisson_test_$(mpi_size())_$(mc.n).png")
end

if mpi_size()>1
    mpi_finalize()
end