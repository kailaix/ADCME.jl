export GAN, 
jsgan,
klgan,
wgan,
rklgan,
lsgan,
sample,
predict,
dcgan_generator,
dcgan_discriminator

mutable struct GAN
    latent_dim::Int64
    batch_size::Int64
    dim::Int64
    dat::PyObject
    generator::Union{Missing, Function}
    discriminator::Union{Missing, Function}
    loss::Union{Missing, String, Function}
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
end


@doc raw"""
    GAN(dat::PyObject, 
        generator::Function, 
        gan::GAN,
        loss::Union{String, Function, Missing}=missing; 
        latent_dim::Union{Missing, Int64}=missing, 
        batch_size::Union{Missing, Int64}=missing)

Creates a GAN instance. 

- `dat` ``\in \mathbb{R}^{n\times d}`` is the training data for the GAN, where ``n`` is the number of training data, and ``d`` is the dimension per training data.
- `generator```:\mathbb{R}^{d'} \rightarrow \mathbb{R}^d`` is the generator function, ``d'`` is the hidden dimension.
- `discriminator```:\mathbb{R}^{d} \rightarrow \mathbb{R}`` is the discriminator function. 
- `loss` is the loss function. See [`klgan`](@ref), [`rklgan`](@ref), [`wgan`](@ref), [`lsgan`](@ref) for examples.
- `latent_dim` (default=``d``) is the latent dimension.
- `batch_size` (default=32) is the batch size in training.
"""
function GAN(dat::Union{Array,PyObject}, generator::Function, discriminator::Function,
    loss::Union{String, Function, Missing}=missing; latent_dim::Union{Missing, Int64}=missing,
        batch_size::Union{Missing, Int64}=missing)
    dim = size(dat, 2)
    dat = convert_to_tensor(dat)
    if ismissing(latent_dim); latent_dim=dim; end
    if ismissing(batch_size); batch_size=32; end
    gan = GAN(latent_dim, batch_size, dim, dat, generator, discriminator, loss, missing, missing, missing, 
        missing, placeholder(true, shape=[]), missing, missing, missing, randstring(), missing, missing)
    build!(gan)
    gan
end
GAN(dat::Array{T}) where T<:Real = GAN(constant(dat))


function Base.:show(io::IO, gan::GAN)
    print("GAN(id=$(gan.ganid), dat(shape=$(size(gan.dat)), type=$(get_dtype(gan.dat)), latent_dim=$(gan.latent_dim), batch_size=$(gan.batch_size), $(length(gan.d_vars)) d_vars, $(length(gan.g_vars)) g_vars, $(length(gan.update)) update ops, fake_data(shape=$(size(gan.fake_data))), true_data(shape=$(size(gan.true_data))))")
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
    build!(gan::GAN)

Builds the GAN instances. This function returns `gan` for convenience.
"""
function build!(gan::GAN)
    gan.noise = placeholder(get_dtype(gan.dat), shape=(gan.batch_size, gan.latent_dim))
    gan.ids = placeholder(Int32, shape=(gan.batch_size,))
    variable_scope("generator_$(gan.ganid)", initializer=random_uniform_initializer(0.0,1e-3)) do
        gan.fake_data = gan.generator(gan.noise, gan)
    end
    gan.true_data = tf.gather(gan.dat,gan.ids-1)
    if ismissing(gan.loss); gan.loss="jsgan"; end
    if isa(gan.loss, String)
        gan.loss = eval(Meta.parse(gan.loss))
    end
    variable_scope("discriminator_$(gan.ganid)", initializer=random_uniform_initializer(0.0,1e-3)) do
        gan.d_loss, gan.g_loss = gan.loss(gan)
    end
    gan.d_vars = Array{PyObject}(get_collection("discriminator_$(gan.ganid)"))
    gan.g_vars = Array{PyObject}(get_collection("generator_$(gan.ganid)"))
    gan.update = Array{PyCall.PyObject}(get_collection(UPDATE_OPS))
    gan
end 

"""
    sample(gan::GAN, n::Int64)

Samples `n` instances from `gan`.
"""
function sample(gan::GAN, n::Int64)
    local out
    noise = normal(n, gan.latent_dim)
    gan.is_training = false
    variable_scope("generator_$(gan.ganid)", initializer=random_uniform_initializer(0.0,1e-3)) do
        out = gan.generator(noise, gan)
    end
    gan.is_training = placeholder(true, shape=[])
    out
end

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