using Revise
using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using Test
using SparseArrays
Random.seed!(233)

sparse_indexing = load_op_and_grad("build/libSparseIndexing", "sparse_indexing", multiple=true)
# function Base.:getindex(s::SparseTensor, i1::Union{PyObject,Array{S,1}},
#          i2::Union{PyObject,Array{T,1}}) where {S<:Real,T<:Real}
#     m_, n_ = length(i1), length(i2)
#     i1 = convert_to_tensor(i1, dtype=Int64)
#     i2 = convert_to_tensor(i2, dtype=Int64)
#     ii1, jj1, vv1 = find(s)
#     m = tf.convert_to_tensor(s.o.shape[1],dtype=tf.int64)
#     n = tf.convert_to_tensor(s.o.shape[2],dtype=tf.int64)
#     ii2, jj2, vv2 = sparse_indexing(ii1,jj1,vv1,m,n,i1,i2)
#     SparseTensor(ii2, jj2, vv2, m_, n_)
# end
################## End Load Operator ##################

i1 = unique(rand(1:20,3))
j1 = unique(rand(1:30,3))
A = sprand(20,30,0.3)
@show i1, j1
Ad = Array(A[i1, j1])
B = SparseTensor(A)
Bd = Array(B[i1, j1])
sess = Session()
init(sess)
Bd_ = run(sess, Bd)
@test Ad≈Bd_

# error()
# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(m_)
    A = SparseTensor(ii1, jj1, m_, 20, 30)
    B = A[i1, j1]
    return sum(B)^2
end

# TODO: change `m_` and `v_` to appropriate values
m = 20
n = 30
ii1,jj1,vv1 = find(B)
m_ = constant(rand(length(B.o.values)))
v_ = rand(length(B.o.values))
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
