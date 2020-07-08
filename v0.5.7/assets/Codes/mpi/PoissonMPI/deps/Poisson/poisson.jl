using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using MPI
Random.seed!(233)

function poisson_op(u,f,h)
    poisson_op_ = load_op_and_grad("./build/libPoissonOp","poisson_op")
    u,f,h = convert_to_tensor(Any[u,f,h], [Float64,Float64,Float64])
    poisson_op_(u,f,h)
end


MPI.Init()
m = 10
n = 20
u = ones(m, n)
f = zeros(m, n)
h = 1.0

# U = zeros(m+2, n+2)
# U[2:end-1, 2:end-1] = u 
# Ut = zeros(m,n)
# for i = 2:m+1
#     for j = 2:n+1
#         Ut[i-1, j-1] = (U[i+1,j] + U[i-1,j] + U[i,j+1] + U[i,j-1] - h*h*f[i-1,j-1])/4
#     end
# end
# U = Ut


# TODO: specify your input parameters
u = poisson_op(u,f,h)
sess = Session(); init(sess)
@show run(sess, u)

# # uncomment it for testing gradients
# error() 


# # TODO: change your test parameter to `m`
# #       in the case of `multiple=true`, you also need to specify which component you are testings
# # gradient check -- v
# function scalar_function(m)
#     return sum(poisson_op(u,f,h)^2)
# end

# # TODO: change `m_` and `v_` to appropriate values
# m_ = constant(rand(10,20))
# v_ = rand(10,20)
# y_ = scalar_function(m_)
# dy_ = gradients(y_, m_)
# ms_ = Array{Any}(undef, 5)
# ys_ = Array{Any}(undef, 5)
# s_ = Array{Any}(undef, 5)
# w_ = Array{Any}(undef, 5)
# gs_ =  @. 1 / 10^(1:5)

# for i = 1:5
#     g_ = gs_[i]
#     ms_[i] = m_ + g_*v_
#     ys_[i] = scalar_function(ms_[i])
#     s_[i] = ys_[i] - y_
#     w_[i] = s_[i] - g_*sum(v_.*dy_)
# end

# sess = Session(); init(sess)
# sval_ = run(sess, s_)
# wval_ = run(sess, w_)
# close("all")
# loglog(gs_, abs.(sval_), "*-", label="finite difference")
# loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
# loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
# loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

# plt.gca().invert_xaxis()
# legend()
# xlabel("\$\\gamma\$")
# ylabel("Error")
