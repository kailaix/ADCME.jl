using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

assemble_op = load_op_and_grad("./build/libAssembleOp","assemble_op", multiple=true)

index = constant([
    1 2 3
    3 4 5
])
ks = constant(ones(2, 3, 3))
sdof = constant(5)
# TODO: specify your input parameters
ii,jj,vv = assemble_op(index,ks,sdof)
A = SparseTensor(ii,jj,vv,5,5)
sess = tf.Session()
init(sess)
B = run(sess, A)
# error()

# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(m)
    return sum(assemble_op(index,m,sdof)[3]^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(ones(2,3,3))
v_ = rand(2,3,3)
y_ = scalar_function(m_)
dy_ = tf.reshape(tf.gradients(y_, m_)[1],(-1,))
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)
# error()
for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(permutedims(v_,[3,2,1])[:]*dy_)
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
