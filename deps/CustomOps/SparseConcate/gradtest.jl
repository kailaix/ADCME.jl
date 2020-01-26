using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using SparseArrays
using Random
Random.seed!(233)

function sparse_concate(A1, A2,hcat_::Bool)
    m1,n1 = size(A1)
    m2,n2 = size(A2)
    if !isa(A1, SparseTensor); A1 = SparseTensor(A1); end 
    if !isa(A2, SparseTensor); A2 = SparseTensor(A2); end 
    ii1,jj1,vv1 = find(A1)
    ii2,jj2,vv2 = find(A2)
    sparse_concate_ = load_op_and_grad("./build/libSparseConcate","sparse_concate", multiple=true)
    ii1,jj1,vv1,m1_,n1_,ii2,jj2,vv2,m2_,n2_ = convert_to_tensor([ii1,jj1,vv1,m1,n1,ii2,jj2,vv2,m2,n2], [Int64,Int64,Float64,Int32,Int32,Int64,Int64,Float64,Int32,Int32])
    ii,jj,vv = sparse_concate_(ii1,jj1,vv1,m1_,n1_,ii2,jj2,vv2,m2_,n2_,constant(hcat_))
    if hcat_
        SparseTensor(ii,jj,vv, m1, n1+n2)
    else
        SparseTensor(ii,jj,vv,m1+m2,n1)
    end
end

# TODO: specify your input parameters
A1 = sprand(10,10,0.4)
A2 = sprand(10,10,0.3)
u = sparse_concate(A1, A2, true)
sess = Session(); init(sess)
@show run(sess, u)-[A1 A2]

u = sparse_concate(A1, A2, false)
sess = Session(); init(sess)
@show run(sess, u)-[A1;A2]

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    A1 = SparseTensor(ii1,jj1,m,10,10)
    A2 = SparseTensor(ii2,jj2,m,10,10)
    return sum(sparse_concate(A1, A2, false))^2
end

# TODO: change `m_` and `v_` to appropriate values
ii1 = rand(1:10,10)
jj1 = rand(1:10,10)
ii2 = rand(1:10,10)
jj2 = rand(1:10,10)
vv1 = rand(10)
vv2 = rand(10)
m_ = constant(rand(10))
v_ = rand(10)
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
