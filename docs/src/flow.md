# Flow-based Generative Model

In this article we introduce the ADCME module for flow-based generative models. The flow-based generative models can be used to model the joint distribution of high-dimensional random variables. It constructs a sequence of invertible transformation of distributions
$$
x = f(u) \quad u \sim \pi(u)
$$
based on the change of variable equation
$$
p(x) = \pi(f^{-1}(x)) \left|\det\left(\frac{\partial f^{-1}}{\partial x}\right)\right|
$$
Compared to other generative models such as variational autoencoder (VAE) and generative neural networks (GAN), the flow-based generative models give us explicit formuas of density functions. For model training, we can directly minimizes the posterier log likelihood in the flow-based generative models, while use approximate likelihood functions in VAE and adversarial training in GAN. In general, the flow-based generative model is easier to train than VAE and GAN. In the following, we give some examples of using flow-based generatives models in ADCME. 

## Type Hierarchy

The flow-based generative model is organized as follows, from botton level to top level:

* `FlowOp`. This consists of unit invertible transformations, such as [`AffineConstantFlow`](@ref) and [`Invertible1x1Conv`](@ref).
* `NormalizingFlow`. This is basically a sequence of `FlowOp`. It is not exposed to users. 
* `NormalizingFlowModel`. This is a container of the sequence of `FlowOp`s and a prior distribution. `NormalizingFlowModel` is callable and can "normalize" the data distribution. We can also sample from `NormalizingFlowModel`, where the prior distribution is transformed to data distribution. 

## A Simple Example

Let's consider a simple example for transforming the two moons dataset to a univariate Gaussian distribution. First, we adapt a function from [here](https://github.com/wildart/nmoons) and use it to generate the dataset

```julia
using Revise
using ADCME
using PyCall
using PyPlot

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

x = sample_moons(128)
scatter(x[:,1],x[:,2])
axis("equal")
```

![](./assets/moons.png)

Next we construct a flow-based generative model, as follows:

```julia
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
prior = ADCME.MultivariateNormalDiag(loc=zeros(2))
model = NormalizingFlowModel(prior, flows)

x = placeholder(Float64, shape=[128,2])
zs, prior_logpdf, logdet = model(x)
log_pdf = prior_logpdf + logdet
loss = -sum(log_pdf)
```

Finally, we maximize the log llikelihood function using [`AdamOptimizer`](@ref)

```julia
sess = Session(); init(sess)
opt = AdamOptimizer(1e-4).minimize(loss)
sess = Session(); init(sess)
for i = 1:10000
    _, l = run(sess, [opt, loss], x=>sample_moons(128))
    if mod(i,100)==0
        @info i, l 
    end
end

x = sample_moons(128*8)
zs = rand(model, 128*8)
z = zs[end]
z = run(sess, z)
scatter(x[:,1], x[:,2], c="b", s=5, label="data")
scatter(z[:,1], z[:,2], c="r", s=5, label="prior --> posterior")
axis("scaled")
```

![](./assets/moonresult.png)