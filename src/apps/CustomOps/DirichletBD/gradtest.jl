using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

dirichlet_bd = load_op_and_grad("./build/libDirichletBd","dirichlet_bd")

ii = constant([1;2;3;4])
jj = constant([1;1;1;1])
dof = constant([2;3])
vv = constant(rand(4))
# TODO: specify your input parameters
u = dirichlet_bd(ii,jj,dof,vv)
sess = tf.Session()
init(sess)
run(sess, u)

# error()
# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(m)
    return sum(dirichlet_bd(ii,jj,dof,m)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(4))
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
