using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

relu(x) = max(x, 0)
function extended_nn(x,config,theta,activation)
    extended_nn_ = load_op_and_grad("./build/libExtendedNn","extended_nn", multiple=true)
    x,config_,theta = convert_to_tensor([x,config,theta], [Float64,Int64,Float64])
    u, du = extended_nn_(x,config_,theta,activation)
    n = length(x)÷config[1]
    reshape(u, (n, config[end])), du 
end

# TODO: specify your input parameters
x = rand(20)
config = [2,20,50,8]
activation = "tanh"

W1 = rand(2,20); b1 = rand(20)
W2  = rand(20,50); b2 = rand(50);
W3 = rand(50,8); b3 = rand(8)
X = reshape(x, 2, 10)'|>Array 

y1 = tanh.(X*W1 .+ b1')
y2 = tanh.(y1*W2 .+ b2')
y3 = y2*W3 .+ b3'

# x = rand(10)
# config = [1,2]
# activation = "tanh"

# W1 = ones(1,2); b1 = ones(2)
# X = reshape(x, 1, 10)'|>Array 

# y1 = X*W1 .+ b1'

θ = [W1'[:];b1[:];W2'[:];b2[:];W3'[:];b3[:]]
# θ = [W1'[:];b1[:]]
tfx = constant(x)
u = extended_nn(x,config,θ,activation)
sess = Session(); init(sess)
@show run(sess, u)[1]-y3

# S = gradients(u[1][:,1], tfx)
# run(sess, S)

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    return sum(extended_nn(m,config,θ,activation)[1]^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(length(θ)))
v_ = rand(length(θ))
m_ = constant(rand(length(x)))
v_ = rand(length(x))
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
