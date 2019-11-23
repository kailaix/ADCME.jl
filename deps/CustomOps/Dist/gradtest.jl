using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

dist = load_op_and_grad("./build/libDist","dist")

# TODO: specify your input parameters
A = rand(10,3)
B = rand(20,3)
M = zeros(10,20)
for i = 1:10
    for j = 1:20
        M[i,j] = sum(abs.(A[i,:] - B[j,:]).^2)
    end
end
M = sqrt.(M)
x = constant(A)
y = constant(B)
order = constant(2)
u = dist(x,y,order)
sess = tf.Session()
init(sess)
N = run(sess, u)
# error()


# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(m)
    return sum(dist(x,m,order)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(20,3))
v_ = rand(20,3)
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
