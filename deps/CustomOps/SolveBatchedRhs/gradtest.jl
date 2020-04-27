using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function solve_batched_rhs(a,rhs)
    solve_batched_rhs_ = load_op_and_grad("./build/libSolveBatchedRhs","solve_batched_rhs")
    a,rhs = convert_to_tensor([a,rhs], [Float64,Float64])
    solve_batched_rhs_(a,rhs)
end

a = rand(10,5)
rhs = rand(100, 10)
sol = (a\rhs')'
# TODO: specify your input parameters
u = solve_batched_rhs(a,rhs)
sess = Session(); init(sess)
@show run(sess, u) - sol

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    return sum(solve_batched_rhs(m,rhs)^2)
    # return sum(solve_batched_rhs(a,m)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(10,10))
v_ = rand(10,10)
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
