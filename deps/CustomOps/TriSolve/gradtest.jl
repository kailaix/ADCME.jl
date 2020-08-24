using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using Test
Random.seed!(233)

function tri_solve(a,b,c,d)
    tri_solve_ = load_op_and_grad("./build/libTriSolve","tri_solve")
    a,b,c,d = convert_to_tensor(Any[a,b,c,d], [Float64,Float64,Float64,Float64])
    tri_solve_(a,b,c,d)
end

n = 10
a = rand(n-1)
b = rand(n).+10
c = rand(n-1)
d = rand(n)

A = diagm(0=>b, -1=>a, 1=>c)
x = A\d

# TODO: specify your input parameters
u = tri_solve(a,b,c,d)
sess = Session(); init(sess)
@test run(sess, u)≈x

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    return sum(tri_solve(a,b,x,d)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(n-1))
v_ = rand(n-1)
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
