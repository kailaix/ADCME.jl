using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using SparseArrays
Random.seed!(233)

function sparse_solver(ii,jj,vv,f,method="SparseLU")
    sparse_solver_ = load_op_and_grad("./build/libSparseSolver","sparse_solver")
    ii,jj,vv,f = convert_to_tensor([ii,jj,vv,f], [Int64,Int64,Float64,Float64])
    sparse_solver_(ii,jj,vv,f,method)
end

# TODO: specify your input parameters
sess = Session(); init(sess)
for method in ["SparseLU", "SparseQR", "SimplicialLDLT", "SimplicialLLT"]
A = sprand(10,10,0.6)
global A = A'*A
global f = rand(10)
A_ = SparseTensor(A)
global ii, jj, vv = find(A_)
u = sparse_solver(ii,jj,vv,f,method)

@show norm(run(sess, u)-A\f)
end

# uncomment it for testing gradients
# error() 
# py"""
# import traceback
# try:
#     $run($sess, $dy_)
# except Exception:
#     print(traceback.format_exc())
# """
# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    return sum(sparse_solver(ii,jj,vv,m)^2)
end

# TODO: change `m_` and `v_` to appropriate values
# m_ = constant(rand(length(vv)))
# v_ = rand(length(vv))

m_ = constant(f)
v_ = rand(length(f))
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
