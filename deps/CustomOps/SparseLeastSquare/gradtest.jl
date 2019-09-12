using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

################## Load Operator ##################
if Sys.islinux()
py"""
import tensorflow as tf
libSparseLeastSquare = tf.load_op_library('build/libSparseLeastSquare.so')
@tf.custom_gradient
def sparse_least_square(ii,jj,vv,ff,n):
    u = libSparseLeastSquare.sparse_least_square(ii,jj,vv,ff,n)
    def grad(dy):
        return libSparseLeastSquare.sparse_least_square_grad(dy, u, ii,jj,vv,ff,n)
    return u, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libSparseLeastSquare = tf.load_op_library('build/libSparseLeastSquare.dylib')
@tf.custom_gradient
def sparse_least_square(ii,jj,vv,ff,n):
    u = libSparseLeastSquare.sparse_least_square(ii,jj,vv,ff,n)
    def grad(dy):
        return libSparseLeastSquare.sparse_least_square_grad(dy, u, ii,jj,vv,ff,n)
    return u, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libSparseLeastSquare = tf.load_op_library('build/libSparseLeastSquare.dll')
@tf.custom_gradient
def sparse_least_square(ii,jj,vv,ff,n):
    u = libSparseLeastSquare.sparse_least_square(ii,jj,vv,ff,n)
    def grad(dy):
        return libSparseLeastSquare.sparse_least_square_grad(dy, u, ii,jj,vv,ff,n)
    return u, grad
"""
end

sparse_least_square = py"sparse_least_square"
################## End Load Operator ##################

# TODO: specify your input parameters
ii = Int32[1;1;2;2;3;3]
jj = Int32[1;2;1;2;1;2]
vv = Float64[1;2;3;4;5;6]
ff = Float64[1;1;1]
A = Float64[1 2;3 4;5 6]; f = Float64[1;1;1]
sol = A\f
n = 2
# error()
u = sparse_least_square(ii,jj,vv,ff,constant(n, dtype=Int32))
sess = Session()
init(sess)
run(sess, u)-sol
# error()


# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(m)
    return sum(tanh(sparse_least_square(ii,jj,m,ff,n)))
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(vv)
v_ = rand(6)
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
