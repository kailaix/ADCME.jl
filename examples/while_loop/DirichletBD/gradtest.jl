using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libDirichletBD = tf.load_op_library('build/libDirichletBD.so')
@tf.custom_gradient
def dirichlet_bd(ii,jj,dof,vv):
    uu = libDirichletBD.dirichlet_bd(ii,jj,dof,vv)
    def grad(dy):
        return libDirichletBD.dirichlet_bd_grad(dy, uu, ii,jj,dof,vv)
    return uu, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libDirichletBD = tf.load_op_library('build/libDirichletBD.dylib')
@tf.custom_gradient
def dirichlet_bd(ii,jj,dof,vv):
    uu = libDirichletBD.dirichlet_bd(ii,jj,dof,vv)
    def grad(dy):
        return libDirichletBD.dirichlet_bd_grad(dy, uu, ii,jj,dof,vv)
    return uu, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libDirichletBD = tf.load_op_library('build/libDirichletBD.dll')
@tf.custom_gradient
def dirichlet_bd(ii,jj,dof,vv):
    uu = libDirichletBD.dirichlet_bd(ii,jj,dof,vv)
    def grad(dy):
        return libDirichletBD.dirichlet_bd_grad(dy, uu, ii,jj,dof,vv)
    return uu, grad
"""
end

dirichlet_bd = py"dirichlet_bd"

# TODO: 
ii = constant([1;2;3;2], dtype=Int32)
jj = constant([1;2;3;3], dtype=Int32)
vv = constant([2.0;3.0;4.0;4.0])
dof = constant([3], dtype=Int32)
u = dirichlet_bd(ii,jj,dof,vv)
sess = Session()
init(sess)
run(sess, u)
# error("")
# TODO: 


# gradient check -- v
function scalar_function(m)
    return sum(tanh(dirichlet_bd(ii,jj,dof,m)))
end

m_ = vv
v_ = rand(4)
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
