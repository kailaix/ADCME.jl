export GAN, 
jsgan,
klgan,
wgan,
rklgan,
lsgan,
sample,
predict,
dcgan_generator,
dcgan_discriminator,
wgan_stable

mutable struct GAN
    latent_dim::Int64
    batch_size::Int64
    dim::Int64
    dat::PyObject
    generator::Union{Missing, Function}
    discriminator::Union{Missing, Function}
    loss::Function 
    d_vars::Union{Missing,Array{PyObject}}
    g_vars::Union{Missing,Array{PyObject}}
    d_loss::Union{Missing,PyObject}
    g_loss::Union{Missing,PyObject}
    is_training::Union{Missing,PyObject, Bool}
    update::Union{Missing,Array{PyObject}}
    fake_data::Union{Missing,PyObject}
    true_data::Union{Missing,PyObject}
    ganid::String
    noise::Union{Missing,PyObject}
    ids::Union{Missing,PyObject}
    STORAGE::Dict{String, Any}
end

"""
    build!(gan::GAN)

Builds the GAN instances. This function returns `gan` for convenience.
"""
function build!(gan::GAN)
    gan.noise = placeholder(get_dtype(gan.dat), shape=(gan.batch_size, gan.latent_dim))
    gan.ids = placeholder(Int32, shape=(gan.batch_size,))
    variable_scope("generator_$(gan.ganid)") do
        gan.fake_data = gan.generator(gan.noise, gan)
    end
    gan.true_data = tf.gather(gan.dat,gan.ids-1)
    variable_scope("discriminator_$(gan.ganid)") do
        gan.d_loss, gan.g_loss = gan.loss(gan)
    end
    gan.d_vars = Array{PyObject}(get_collection("discriminator_$(gan.ganid)"))
    gan.d_vars = length(gan.d_vars)>0 ? gan.d_vars : missing
    gan.g_vars = Array{PyObject}(get_collection("generator_$(gan.ganid)"))
    gan.g_vars = length(gan.g_vars)>0 ? gan.g_vars : missing
    gan.update = Array{PyCall.PyObject}(get_collection(UPDATE_OPS))
    gan.update = length(gan.update)>0 ? gan.update : missing
    # gan.STORAGE["d_grad_magnitude"] = gradient_magnitude(gan.d_loss, gan.d_vars)
    # gan.STORAGE["g_grad_magnitude"] = gradient_magnitude(gan.g_loss, gan.g_vars)
    gan
end 

@doc raw"""
    GAN(dat::Union{Array,PyObject}, generator::Function, discriminator::Function,
    loss::Union{Missing, Function}=missing; latent_dim::Union{Missing, Int64}=missing,
        batch_size::Int64=32)

Creates a GAN instance. 

- `dat` ``\in \mathbb{R}^{n\times d}`` is the training data for the GAN, where ``n`` is the number of training data, and ``d`` is the dimension per training data.
- `generator```:\mathbb{R}^{d'} \rightarrow \mathbb{R}^d`` is the generator function, ``d'`` is the hidden dimension.
- `discriminator```:\mathbb{R}^{d} \rightarrow \mathbb{R}`` is the discriminator function. 
- `loss` is the loss function. See [`klgan`](@ref), [`rklgan`](@ref), [`wgan`](@ref), [`lsgan`](@ref) for examples.
- `latent_dim` (default=``d``, the same as output dimension) is the latent dimension.
- `batch_size` (default=32) is the batch size in training.

# Example: Constructing a GAN
```julia
dat = rand(10000,10)
generator = (z, gan)->10*z
discriminator = (x, gan)->sum(x)
gan = GAN(dat, generator, discriminator, "wgan_stable")
```

# Example: Learning a Gaussian random variable 
```julia
using ADCME 
using PyPlot
using Distributions
dat = randn(10000, 1) * 0.5 .+ 3.0
function gen(z, gan)
    ae(z, [20,20,20,1], "generator_$(gan.ganid)", activation = "relu")
end
function disc(x, gan)
    squeeze(ae(x, [20,20,20,1], "discriminator_$(gan.ganid)", activation = "relu"))
end
gan = GAN(dat, gen, disc, g->wgan_stable(g, 0.001); latent_dim = 10)

dopt = AdamOptimizer(0.0002, beta1=0.5, beta2=0.9).minimize(gan.d_loss, var_list=gan.d_vars)
gopt = AdamOptimizer(0.0002, beta1=0.5, beta2=0.9).minimize(gan.g_loss, var_list=gan.g_vars)
sess = Session(); init(sess)
for i = 1:5000
    batch_x = rand(1:10000, 32)
    batch_z = randn(32, 10)
    for n_critic = 1:1
        global _, dl = run(sess, [dopt, gan.d_loss], 
                feed_dict=Dict(gan.ids=>batch_x, gan.noise=>batch_z))
    end
    _, gl, gm, dm, gp = run(sess, [gopt, gan.g_loss, 
        gan.STORAGE["g_grad_magnitude"], gan.STORAGE["d_grad_magnitude"], 
        gan.STORAGE["gradient_penalty"]],
        feed_dict=Dict(gan.ids=>batch_x, gan.noise=>batch_z))
    mod(i, 100)==0 && (@info i, dl, gl, gm, dm, gp)
end

hist(run(sess, squeeze(rand(gan,10000))), bins=50, density = true)
nm = Normal(3.0,0.5)
x0 = 1.0:0.01:5.0
y0 = pdf.(nm, x0)
plot(x0, y0, "g")
```
"""
function GAN(dat::Union{Array,PyObject}, generator::Function, discriminator::Function,
    loss::Union{Missing, Function, String}=missing; latent_dim::Union{Missing, Int64}=missing,
        batch_size::Int64=32)
    dim = size(dat, 2)
    dat = convert_to_tensor(dat)
    if ismissing(latent_dim)
        latent_dim=dim
    end
    if ismissing(loss)
        loss = "jsgan"
    end

    if isa(loss, String)
        if loss=="jsgan"
            loss = jsgan
        elseif loss=="klgan"
            loss = klgan
        elseif loss=="wgan"
            loss = wgan 
        elseif loss=="wgan_stable"
            loss = wgan_stable
        elseif loss=="rklgan"
            loss = rklgan
        elseif loss=="lsgan"
            loss = lsgan
        else
            error("loss function $loss not found!")
        end
    end
    gan = GAN(latent_dim, batch_size, dim, dat, generator, discriminator, loss, missing, missing, missing, 
        missing, ADCME.options.training.training, missing, missing, missing, randstring(), missing, missing, Dict())
    build!(gan)
    gan
end
GAN(dat::Array{T}) where T<:Real = GAN(constant(dat))


function Base.:show(io::IO, gan::GAN)
    yes_or_no = x->ismissing(x) ? "✘" : "✔️"
    print(
"""
( Generative Adversarial Network -- $(gan.ganid) )
==================================================
    $(gan.latent_dim) D          $(gan.dim) D
(Latent Space)---->(Data Space)
     batch_size = $(gan.batch_size)

loss function: $(gan.loss)
generator:     $(gan.generator)
discriminator: $(gan.discriminator)
d_vars     ... $(yes_or_no(gan.d_vars))
g_vars     ... $(yes_or_no(gan.g_vars))
d_loss     ... $(yes_or_no(gan.d_loss))
g_loss     ... $(yes_or_no(gan.g_loss))
update     ... $(yes_or_no(gan.update))
true_data  ... $(size(gan.true_data))
fake_data  ... $(size(gan.fake_data))
is_training... Placeholder (Bool), $(gan.is_training)
noise      ... Placeholder (Float64) of size $(size(gan.noise))
ids        ... Placeholder (Int32) of size $(size(gan.ids))
"""
    )
end
##################### GAN Library #####################
"""
    klgan(gan::GAN)

Computes the KL-divergence GAN loss function.
"""
function klgan(gan::GAN)
    P, Q = gan.true_data, gan.fake_data
    D_real = gan.discriminator(P, gan)
    D_fake = gan.discriminator(Q, gan)
    D_loss = -mean(log(D_real) + log(1-D_fake))
    G_loss = mean(log((1-D_fake)/D_fake))
    D_loss, G_loss
end

"""
    jsgan(gan::GAN)

Computes the vanilla GAN loss function.
"""
function jsgan(gan::GAN)
    P, Q = gan.true_data, gan.fake_data
    D_real = gan.discriminator(P, gan)
    D_fake = gan.discriminator(Q, gan)
    D_loss = -mean(log(D_real) + log(1-D_fake))
    G_loss = -mean(log(D_fake))
    D_loss, G_loss
end

"""
    wgan(gan::GAN)

Computes the Wasserstein GAN loss function.
"""
function wgan(gan::GAN)
    P, Q = gan.true_data, gan.fake_data
    D_real = gan.discriminator(P, gan)
    D_fake = gan.discriminator(Q, gan)
    D_loss = mean(D_fake)-mean(D_real)
    G_loss = -mean(D_fake)
    D_loss, G_loss
end

@doc raw"""
    wgan_stable(gan::GAN, λ::Float64)

Returns the discriminator and generator loss for the Wasserstein GAN loss with penalty parameter $\lambda$

The objective function is 
```math 
L = E_{\tilde x\sim P_g} [D(\tilde x)] - E_{x\sim P_r} [D(x)] + \lambda E_{\hat x\sim P_{\hat x}}[(||\nabla_{\hat x}D(\hat x)||^2-1)^2]
```
"""
function wgan_stable(gan::GAN, λ::Float64=1.0)
    real_data, fake_data = gan.true_data, gan.fake_data
    @assert length(size(real_data))==2
    @assert length(size(fake_data))==2
    D_logits = gan.discriminator(real_data, gan)
    D_logits_ = gan.discriminator(fake_data, gan)

    g_loss = -mean(D_logits_)
    d_loss_real = -mean(D_logits)
    d_loss_fake = tf.reduce_mean(D_logits_)
    α = tf.random_uniform(
        shape=(gan.batch_size, 1), 
        minval=0.,
        maxval=1.,
        dtype=tf.float64
    )
    differences = fake_data - real_data
    interpolates = real_data + α .* differences # nb x dim
    d_inter = gan.discriminator(interpolates, gan) 
    @assert length(size(d_inter))==1
    gradients = tf.gradients(d_inter, interpolates)[1] # ∇D(xt), nb x dim
    slopes = sqrt(sum(gradients^2, dims=2)) # ||∇D(xt)||, (nb,)
    gradient_penalty = mean((slopes-1.)^2) 
    d_loss = d_loss_fake + d_loss_real + λ * gradient_penalty
    gan.STORAGE["gradient_penalty"] = mean(slopes)
    return d_loss, g_loss
end

"""
    rklgan(gan::GAN)

Computes the reverse KL-divergence GAN loss function.
"""
function rklgan(gan::GAN)
    P, Q = gan.true_data, gan.fake_data
    D_real = gan.discriminator(P, gan)
    D_fake = gan.discriminator(Q, gan)
    G_loss = mean(log((1-D_fake)/D_fake))
    D_loss = -mean(log(D_fake)+log(1-D_real))
    D_loss, G_loss
end

"""
    lsgan(gan::GAN)

Computes the least square GAN loss function.
"""
function lsgan(gan::GAN)
    P, Q = gan.true_data, gan.fake_data
    D_real = gan.discriminator(P, gan)
    D_fake = gan.discriminator(Q, gan)
    D_loss = mean((D_real-1)^2+D_fake^2)
    G_loss = mean((D_fake-1)^2)
    D_loss, G_loss
end
#######################################################

"""
    sample(gan::GAN, n::Int64)
    rand(gan::GAN, n::Int64)

Samples `n` instances from `gan`.
"""
function sample(gan::GAN, n::Int64)
    local out
    @info "Using a normal latent vector"
    noise = constant(randn(n, gan.latent_dim))
    variable_scope("generator_$(gan.ganid)") do
        out = gan.generator(noise, gan)
    end
    out
end

Base.:rand(gan::GAN) = squeeze(sample(gan, 1), dims=1)
Base.:rand(gan::GAN, n::Int64) = sample(gan, n)

"""
    predict(gan::GAN, input::Union{PyObject, Array}) 

Predicts the GAN `gan` output given input `input`. 
"""
function predict(gan::GAN, input::Union{PyObject, Array})
    local out
    flag = false
    if length(size(input))==1
        flag = true
        input = reshape(input, 1, length(input))
    end
    input = convert_to_tensor(input)
    variable_scope("generator_$(gan.ganid)", initializer=random_uniform_initializer(0.0,1e-3)) do
        out = gan.generator(input, gan)
    end
    if flag; out = squeeze(out); end
    out
end

# adapted from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dcgan.py
function dcgan_generator(x::PyObject, n::Int64=1)
    if length(size(x))!=2
        error("ADCME: input must have rank 2, rank $(length(size(x))) received")
    end
    variable_scope("generator", reuse=AUTO_REUSE) do 
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        x = tf.layers.dense(x, units=6n * 6n * 128)
        x = tf.nn.tanh(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 6, 6, 128)
        x = tf.reshape(x, shape=[-1, 6n, 6n, 128])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
        # Deconvolution, image shape: (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
        # Apply sigmoid to clip values between 0 and 1
        x = tf.nn.sigmoid(x)
    end
    return squeeze(x)
end

function dcgan_discriminator(x)
    variable_scope("Discriminator", reuse=AUTO_REUSE) do
        # Typical convolutional neural network to classify images.
        x = tf.layers.conv2d(x, 64, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(x, 2)
    end
    return x
end