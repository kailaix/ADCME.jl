using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using DelimitedFiles
using Random
Random.seed!(233)

function halo_exchange_neighbor_two(u,fill_value,m,n,tag,w)
    halo_exchange_neighbor_two_ = load_op_and_grad("./build/libHaloExchangeNeighborTwo","halo_exchange_neighbor_two")
    u,fill_value,m,n,tag,w = convert_to_tensor(Any[u,fill_value,m,n,tag,w], [Float64,Float64,Int64,Int64,Int64,Float64])
    halo_exchange_neighbor_two_(u,fill_value,m,n,tag,w)
end

mpi_init()
u = rand(5, 5)
fill_value = 10.0
m = 1
n = 1
tag = 1
w = 1.0
# TODO: specify your input parameters
uext = halo_exchange_neighbor_two(u,fill_value,m,n,tag,w)
sess = Session(); init(sess)
uval =  run(sess, uext)
writedlm(stdout, round.(uval, digits=3))
# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    return sum(halo_exchange_neighbor_two(x,fill_value,m,n,tag,w)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(5, 5))
v_ = rand(5, 5)
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
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
savefig("test.png")
