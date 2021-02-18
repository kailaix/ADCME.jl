using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using Test
using Statistics
Random.seed!(233)

sess = Session(); init(sess)

n = 10
window = 9
# TODO: specify your input parameters
inu = rand(n)
for ops in [(rollsum, sum), (rollmean, mean), (rollstd, std), (rollvar, var)]
    @info ops
    u = ops[1](inu,window)
    out = zeros(n-window+1)
    for i = window:n
        out[i-window+1] = ops[2](inu[i-window+1:i])
    end
    @test maximum(abs.(run(sess, u)-out )) < 1e-8
end

# uncomment it for testing gradients
# error() 

n = 10
window = 9

# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    return sum(rollstd(x,window)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(n))
v_ = rand(n)
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
savefig("test.png")