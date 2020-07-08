using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using SparseArrays
Random.seed!(233)

function sparse_to_dense(A)
    A = constant(A)
    ij = A.o.indices
    vv = values(A)
    m, n = size(A)
    sparse_to_dense_ = load_op_and_grad("./build/libSparseToDense","sparse_to_dense_ad")
    m_, n_ = convert_to_tensor(Any[m,n], [Int64,Int64])
    out = sparse_to_dense_(ij, vv, m_,n_)
    set_shape(out, (m, n))
end

A = sprand(10,10,0.3)
# TODO: specify your input parameters
u = sparse_to_dense(A)
sess = Session(); init(sess)
@show run(sess, u) - Array(A)

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    B = SparseTensor(ij[:,1]+1, ij[:,2]+1, x, m, n)
    return sum(sparse_to_dense(B)^2)
end

# TODO: change `m_` and `v_` to appropriate values
A = constant(A)
m, n = size(A)
K = length(values(A))
ij = A.o.indices
m_ = constant(rand(K))
v_ = rand(K)
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
