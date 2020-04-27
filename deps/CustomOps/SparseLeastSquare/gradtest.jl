using Revise

using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using SparseArrays
using Random
# Random.seed!(233)


# TODO: specify your input parameters
A = sprand(10,5,0.3)
f = rand(10)
sol = A\f
u = constant(A)\f

sess = Session()
init(sess)
@show run(sess, u)-sol
# error()


# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(m)
    return sum((constant(A)\m)^2)
    B = SparseTensor(ii, jj, m, size(A)...)
    return sum((B\Array([f f]'))^2)
end

ii, jj, vv = find(constant(A))

# TODO: change `m_` and `v_` to appropriate values
# m_ = constant(rand(length(vv)))
# v_ = rand(length(vv))
m_ = constant(rand(5,10))
v_ = rand(5,10)
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
