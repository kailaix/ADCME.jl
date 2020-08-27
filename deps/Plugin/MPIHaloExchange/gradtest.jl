using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function halo_exchange_two_d(u,fill_value,m,n)
    halo_exchange_two_d_ = load_op_and_grad("./build/libHaloExchangeTwoD","halo_exchange_two_d")
    u,fill_value,m,n = convert_to_tensor(Any[u,fill_value,m,n], [Float64,Float64,Int64,Int64])
    halo_exchange_two_d_(u,fill_value,m,n)
end

mpi_init()
U = reshape(1:24, 4, 6)'|>Array

m = 3
n = 2
fill_value = 1.0

M = mpi_rank()÷n+1
N = mpi_rank()%n+1
ulocal = U[(M-1)*2 + 1: M * 2, (N-1)*2+1:N*2]


sess = Session(); init(sess)

# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    x = mpi_bcast(x)
    u = x .* ulocal
    return mpi_sum(sum(halo_exchange_two_d(u,fill_value,m,n)^2))
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(2,2))
v_ = rand(2,2)
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

sess = Session(); init(sess)
sval_ = run(sess, s_)
wval_ = run(sess, w_)

if mpi_rank()==0 
    close("all")
    loglog(gs_, abs.(sval_), "*-", label="finite difference")
    loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
    loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
    loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

    plt.gca().invert_xaxis()
    legend()
    xlabel("\$\\gamma\$")
    ylabel("Error")
    savefig("gradtest.png")
end

mpi_finalize()
