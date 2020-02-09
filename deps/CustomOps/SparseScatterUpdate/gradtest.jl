using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using SparseArrays
using Random
Random.seed!(233)

function scatter_update(A::Union{SparseTensor, SparseMatrixCSC{Float64,Int64}},
            ii,jj,B::Union{SparseTensor, SparseMatrixCSC{Float64,Int64}})
    !isa(A, SparseTensor) && (A=SparseTensor(A))
    !isa(B, SparseTensor) && (B=SparseTensor(B))
    ii1, jj1, vv1 = find(A)
    m1_, n1_ = size(A)
    ii2, jj2, vv2 = find(B)

    sparse_scatter_update_ = load_op_and_grad("./build/libSparseScatterUpdate","sparse_scatter_update", multiple=true)
    ii1,jj1,vv1,m1,n1,ii2,jj2,vv2,ii,jj = convert_to_tensor([ii1,jj1,vv1,m1_,n1_,ii2,jj2,vv2,ii,jj], [Int64,Int64,Float64,Int64,Int64,Int64,Int64,Float64,Int64,Int64])
    ii, jj, vv = sparse_scatter_update_(ii1,jj1,vv1,m1,n1,ii2,jj2,vv2,ii,jj)
    SparseTensor(ii, jj, vv, m1_, n1_)
end

# TODO: specify your input parameters
A = sprand(10,10,0.3)
B = sprand(3,3,0.6)
ii = [1;4;5]
jj = [2;4;6]
u = scatter_update(A, ii, jj, B)
C = copy(A)
C[ii,jj] = B
sess = Session(); init(sess)
@show run(sess, u)-C

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(A, B, m)
    ii1, jj1, vv1 = find(A)
    ii2, jj2, vv2 = find(B)
    if length(m)==length(vv1)
        A = SparseTensor(ii1, jj1, m, size(A)...)
    else
        B = SparseTensor(ii2, jj2, m, size(B)...)
    end
    C = scatter_update(A, ii, jj, B)
    return sum(C)^2
end

# TODO: change `m_` and `v_` to appropriate values

A = sprand(10,10,0.3)|>SparseTensor
B = sprand(3,3,0.6)|>SparseTensor
ii = [1;4;5]
jj = [2;4;6]


m_ = constant(rand(length(values(A))))
v_ = rand(length(values(A)))
y_ = scalar_function(A, B, m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(A, B, ms_[i])
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
