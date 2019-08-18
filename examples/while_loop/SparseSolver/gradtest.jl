using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using SparseArrays
Random.seed!(233)

################## Load Operator ##################
if Sys.islinux()
py"""
import tensorflow as tf
libSparseSolver = tf.load_op_library('build/libSparseSolver.so')
@tf.custom_gradient
def sparse_solver(ii,jj,vv,kk,ff,d):
    u = libSparseSolver.sparse_solver(ii,jj,vv,kk,ff,d)
    def grad(dy):
        return libSparseSolver.sparse_solver_grad(dy, u, ii,jj,vv,kk,ff,d)
    return u, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libSparseSolver = tf.load_op_library('build/libSparseSolver.dylib')
@tf.custom_gradient
def sparse_solver(ii,jj,vv,kk,ff,d):
    u = libSparseSolver.sparse_solver(ii,jj,vv,kk,ff,d)
    def grad(dy):
        return libSparseSolver.sparse_solver_grad(dy, u, ii,jj,vv,kk,ff,d)
    return u, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libSparseSolver = tf.load_op_library('build/libSparseSolver.dll')
@tf.custom_gradient
def sparse_solver(ii,jj,vv,kk,ff,d):
    u = libSparseSolver.sparse_solver(ii,jj,vv,kk,ff,d)
    def grad(dy):
        return libSparseSolver.sparse_solver_grad(dy, u, ii,jj,vv,kk,ff,d)
    return u, grad
"""
end

sparse_solver = py"sparse_solver"
################## End Load Operator ##################

# TODO:
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
ii = constant(ii,dtype=Int32)
jj = constant(jj,dtype=Int32)
vv = constant(vv)
kk = constant(kk,dtype=Int32)
ff = constant(ff)
d = constant(d0,dtype=Int32)


# ii = constant([1;2;3;2],dtype=Int32)
# jj = constant([1;2;3;3],dtype=Int32)
# vv = constant([1.0;2.0;2.0;1.0])
# kk = constant([1;2;3],dtype=Int32)
# ff = constant([1.0;2.0;3.0])
# d = constant(3,dtype=Int32)
u = sparse_solver(ii,jj,vv,kk,ff,d)
sess = Session()
init(sess)
@show norm(run(sess, u)-u_)
# error("")
# TODO: 


# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(m)
    return sum(tanh(sparse_solver(ii,jj,m,kk,ff,d)))
end

m_ = vv
v_ = rand(nv+d0)
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

sess = Session()
@show run(sess, dy_)
