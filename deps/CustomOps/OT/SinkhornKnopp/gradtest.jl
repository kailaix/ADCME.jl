using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

sk = load_op_and_grad("./build/libSinkhornKnopp","sinkhorn_knopp", multiple=true)

# TODO: specify your input parameters
a = constant(ones(10)/10)
b = constant(ones(20)/20)
m = constant(rand(10,20))
reg = constant(1.0)
iter = constant(10000, dtype=Int64)
tol = constant(1e-10)
method = constant(1);
u = sk(a,b,m,reg,iter,tol,method)
sess = tf.Session()
init(sess)
M, _ = run(sess, u)
@show sum(M, dims=1), sum(M, dims=2)
# error()

# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(m)
    return sk(a,b,m,reg,iter,tol,method)[2]
end

# TODO: change `m_` and `v_` to appropriate values
G = rand(10,20)
m_ = constant(G)
v_ = rand(10,20)
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
