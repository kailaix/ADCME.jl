using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using SparseArrays
Random.seed!(233)

sparse_solver = load_op_and_grad("$(@__DIR__)/build/libSparseSolver", "sparse_solver")

d0 = 30
nv = 100
nf = 100
aug = Array(1:d0)
ii = [rand(1:d0,nv);aug...]
jj = [rand(1:d0,nv);aug...]
vv = [rand(nv);rand(d0)]
A = sparse(ii,jj,vv,d0,d0)
kk = rand(1:d0,nf)
ff = rand(nf)
rhs = (sparse(kk,ones(Int64,nf),ff,d0,1)|>Array)[:,1]
u_ = A\rhs
# @show Array(A), rhs
ii = constant(ii,dtype=Int64)
jj = constant(jj,dtype=Int64)
vv = constant(vv)
kk = constant(kk,dtype=Int64)
ff = constant(ff)
d = constant(d0,dtype=Int64)


# ii = constant([1;2;3;2],dtype=Int64)
# jj = constant([1;2;3;3],dtype=Int64)
# vv = constant([1.0;2.0;2.0;1.0])
# kk = constant([1;2;3],dtype=Int64)
# ff = constant([1.0;2.0;3.0])
# d = constant(3,dtype=Int64)
u = sparse_solver(ii,jj,vv,kk,ff,d)
sess = Session()
init(sess)
@show norm(run(sess, u)-u_)
# error("")
# TODO: 


# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(m)
    return sum(tanh(sparse_solver(ii,jj,vv,kk,m,d)))
end

# m_ = vv
# v_ = rand(nv+d0)
m_ = ff
v_ = rand(length(m_))
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

sess = Session()
init(sess)
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
sess = Session()
@show run(sess, dy_)
