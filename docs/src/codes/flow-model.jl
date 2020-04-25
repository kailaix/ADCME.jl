# Adapted from https://github.com/karpathy/pytorch-normalizing-flows
using Revise
using ADCME
using PyCall
using PyPlot
using Random

# `nmoons` is adapted from https://github.com/wildart/nmoons
function nmoons(::Type{T}, n::Int=100, c::Int=2;
    shuffle::Bool=false, ε::Real=0.1, d::Int = 2,
    translation::Vector{T}=zeros(T, d),
    rotations::Dict{Pair{Int,Int},T} = Dict{Pair{Int,Int},T}(),
    seed::Union{Int,Nothing}=nothing) where {T <: Real}
    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(Int(seed))
    ssize = floor(Int, n/c)
    ssizes = fill(ssize, c)
    ssizes[end] += n - ssize*c
    @assert sum(ssizes) == n "Incorrect partitioning"
    pi = convert(T, π)
    R(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    X = zeros(d,0)
    for (i, s) in enumerate(ssizes)
    circ_x = cos.(range(zero(T), pi, length=s)).-1.0
    circ_y = sin.(range(zero(T), pi, length=s))
    C = R(-(i-1)*(2*pi/c)) * hcat(circ_x, circ_y)'
    C = vcat(C, zeros(d-2, s))
    dir = zeros(d)-C[:,end] # translation direction
    X = hcat(X, C .+ dir.*translation)
    end
    y = vcat([fill(i,s) for (i,s) in enumerate(ssizes)]...)
    if shuffle
        idx = randperm(rng, n)
        X, y = X[:, idx], y[idx]
    end
    # Add noise to the dataset
    if ε > 0.0
        X += randn(rng, size(X)).*convert(T,ε/d)
    end
    # Rotate dataset
    for ((i,j),θ) in rotations
        X[[i,j],:] .= R(θ)*view(X,[i,j],:)
    end
    return X, y
end

function sample_moons(n)
    X, _ = nmoons(Float64, n, 2, ε=0.05, d=2, translation=[0.25, -0.25])
    return Array(X')
end


#------------------------------------------------------------------------------------------
# RealNVP
function mlp(x, k, id)
    x = constant(x)
    variable_scope("layer$k$id") do
        x = dense(x, 24, activation="leaky_relu")
        x = dense(x, 24, activation="leaky_relu")
        x = dense(x, 24, activation="leaky_relu")
        x = dense(x, 1)
    end
    return x
end
flows = [AffineHalfFlow(2, mod(i,2)==1, x->mlp(x, i, 0), x->mlp(x, i, 1)) for i = 0:8]


#------------------------------------------------------------------------------------------
# NICE
# function mlp(x, k, id)
#     x = constant(x)
#     variable_scope("layer$k$id") do
#         x = dense(x, 24, activation="leaky_relu")
#         x = dense(x, 24, activation="leaky_relu")
#         x = dense(x, 24, activation="leaky_relu")
#         x = dense(x, 1)
#     end
#     return x
# end
# flow1 = [AffineHalfFlow(2, mod(i,2)==1, missing, x->mlp(x, i, 1)) for i = 0:4]
# flow2 = [AffineConstantFlow(2, shift=false)]
# flows = [flow1;flow2]


# SlowMAF
#------------------------------------------------------------------------------------------
# function mlp(x, k, id)
#     x = constant(x)
#     variable_scope("layer$k$id") do
#         x = dense(x, 24, activation="leaky_relu")
#         x = dense(x, 24, activation="leaky_relu")
#         x = dense(x, 24, activation="leaky_relu")
#         x = dense(x, 2)
#     end
#     return x
# end
# flows = [SlowMAF(2, mod(i,2)==1, [x->mlp(x, i, 0)]) for i = 0:3]

# MAF
#------------------------------------------------------------------------------------------ 
# flows = [MAF(2, mod(i,2)==1, [24, 24, 24], name="layer$i") for i = 0:3]



# IAF 
#------------------------------------------------------------------------------------------ 
# flows = [IAF(2, mod(i,2)==1, [24, 24, 24], name="layer$i") for i = 0:3]
# prior = ADCME.MultivariateNormalDiag(loc=zeros(2))
# model = NormalizingFlowModel(prior, flows)

# Insert ActNorm to any of the flows 
#------------------------------------------------------------------------------------------ 
# flow2 = [ActNorm(2, "ActNorm$i") for i = 1:length(flows)]
# flows = permutedims(hcat(flow2, flows))[:]
# # error()
# # msample = rand(model,1)
# # zs, prior_logprob, log_det = model([0.0040 0.4426])
# # sess = Session(); init(sess)
# # run(sess, msample)
# # run(sess,zs)


# GLOW
#------------------------------------------------------------------------------------------ 
# function mlp(x, k, id)
#     x = constant(x)
#     variable_scope("layer$k$id") do
#         x = dense(x, 24, activation="leaky_relu")
#         x = dense(x, 24, activation="leaky_relu")
#         x = dense(x, 24, activation="leaky_relu")
#         x = dense(x, 1)
#     end
#     return x
# end
# flows = [Invertible1x1Conv(2, "conv$i") for i = 0:2]
# norms = [ActNorm(2, "ActNorm$i") for i = 0:2]
# couplings = [AffineHalfFlow(2, mod(i, 2)==1, x->mlp(x, i, 0), x->mlp(x, i, 1)) for i = 0:length(flows)-1]
# flows = permutedims(hcat(norms, flows, couplings))[:]

#------------------------------------------------------------------------------------------ 
# Neural Splines Coupling
# function mlp(x, k, id)
#     x = constant(x)
#     variable_scope("fc$k$id") do
#         x = dense(x, 16, activation="leaky_relu")
#         x = dense(x, 16, activation="leaky_relu")
#         x = dense(x, 16, activation="leaky_relu")
#         x = dense(x, 3K-1)
#     end
#     return x
# end
# K = 8
# flows = [NeuralCouplingFlow(2, x->mlp(x, i, 0), x->mlp(x, i, 1), K) for i = 0:2]
# convs = [Invertible1x1Conv(2, "conv$i") for i = 0:2]
# norms = [ActNorm(2, "ActNorm$i") for i = 0:2]
# flows = permutedims(hcat(norms, convs, flows))[:]

#------------------------------------------------------------------------------------------ 

prior = ADCME.MultivariateNormalDiag(loc=zeros(2))
model = NormalizingFlowModel(prior, flows)


x = placeholder(rand(128,2))
zs, prior_logpdf, logdet = model(x)
log_pdf = prior_logpdf + logdet
loss = -sum(log_pdf)

model_samples = rand(model, 128*8)
sess = Session(); init(sess)
opt = AdamOptimizer(1e-4).minimize(loss)
sess = Session(); init(sess)
for i = 1:10000
    _, l = run(sess, [opt, loss], x=>sample_moons(128))
    if mod(i,100)==0
        @info i, l 
    end
end

z = run(sess, model_samples[end]) 
x = sample_moons(128*8)
scatter(x[:,1], x[:,2], c="b", s=5, label="data")
scatter(z[:,1], z[:,2], c="r", s=5, label="prior --> posterior")
axis("scaled"); xlabel("x"); ylabel("y")#