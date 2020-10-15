using ADCME
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)
kd = 3
xc = rand(10)
yc = rand(10)
e = rand(10)
c = rand(10)
d = rand(3)
r = RBF2D(xc, yc; c=c, eps=e, d=d, kind = kd)

x = rand(5)
y = rand(5)
o = r(x, y)


sess = Session(); init(sess)
@show run(sess, o)

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    r = RBF2D(xc, m; c=c, eps=e, d=d, kind = kd)
    return sum(r(x,y)^2)
end

# TODO: change `m_` and `v_` to appropriate values
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
