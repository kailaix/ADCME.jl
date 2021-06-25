using Revise
import Optim
using ADCME 
mpi_init()

r = mpi_rank()
θ0 = Variable(rand(2))
θ = mpi_bcast(θ0)
a = 1.
b = 100.
x, y = θ[1], θ[2]
fxy = (a-x)^2 + b*(y-x^2)^2
local_loss = fxy * (r+1)
loss = mpi_sum(local_loss)
g = gradients(loss, θ0)

sess = Session(); init(sess)

_f_eval = 0
_g_eval = 0
function f(x0)
    global _f_eval += 1
    L = run(sess, loss, θ0=>x0)
    L
end

function g!(G, x0)
    global _g_eval += 1
    gv = run(sess, g, θ0=>x0)
    G[:] = gv
end

x0 = rand(2)
mpi_optimize(f, g!, x0)

mpi_finalize()