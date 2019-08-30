export GAN, 
jsgan,
klgan,
wgan,
rklgan,
lsgan,
sample
"""
`GAN(dat::PyObject, generator::Function, gan::GAN,
loss::Union{String, Function, Missing}=missing; latent_dim::Union{Missing, Int64}=missing, batch_size::Union{Missing, Int64}=missing)`

Users must provide: `dat`, `generator`, `discriminator`, `loss` (string or function) 
Alternative argument: `latent_dim`, `batch_size`
Training process is the most creative part and therefore no built-in algorithm is provided. 
Users have access to `d_loss`, `g_loss`, `d_vars`, `g_vars`, which are sufficient for designing training algorithms.
"""
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
end


function GAN(dat::Union{Array,PyObject}, generator::Function, discriminator::Function,
    loss::Union{String, Function, Missing}=missing; latent_dim::Union{Missing, Int64}=missing,
        batch_size::Union{Missing, Int64}=missing)
    dim = size(dat, 2)
    dat = convert_to_tensor(dat)
    if ismissing(latent_dim); latent_dim=dim; end
    if ismissing(batch_size); batch_size=32; end
    gan = GAN(latent_dim, batch_size, dim, dat, generator, discriminator, loss, missing, missing, missing, 
        missing, placeholder(true, shape=[]), missing, missing, missing, randstring())
    build!(gan)
    gan
end
GAN(dat::Array{T}) where T<:Real = GAN(constant(dat))


function Base.:show(io::IO, gan::GAN)
    print("GAN(id=$(gan.ganid), dat(shape=$(size(gan.dat)), type=$(get_dtype(gan.dat)), latent_dim=$(gan.latent_dim), batch_size=$(gan.batch_size), $(length(gan.d_vars)) d_vars, $(length(gan.g_vars)) g_vars, $(length(gan.update)) update ops, fake_data(shape=$(size(gan.fake_data))), true_data(shape=$(size(gan.true_data))))")
end
##################### GAN Library #####################

"""
D_loss, G_loss = klgan(P::PyObject, Q::PyObject, gan::GAN)

return discriminator loss and generator loss for KL divergence
`P` is the real distribution, `Q` is the generated distribution, 
`discriminator` is a critic function that outputs values in (0,1) (e.g. the last activation function is sigmoid)
"""
function klgan(P::PyObject, Q::PyObject, gan::GAN)
    D_real = gan.discriminator(P, gan)
    D_fake = gan.discriminator(Q, gan)
    D_loss = -mean(log(D_real) + log(1-D_fake))
    G_loss = mean(log((1-D_fake)/D_fake))
    D_loss, G_loss
end

"""
D_loss, G_loss = gan(P::PyObject, Q::PyObject, gan::GAN)

return discriminator loss and generator loss for JS divergence
`P` is the real distribution, `Q` is the generated distribution, 
`discriminator` is a critic function that outputs values in (0,1) (e.g. the last activation function is sigmoid)
"""
function jsgan(P::PyObject, Q::PyObject, gan::GAN)
    D_real = gan.discriminator(P, gan)
    D_fake = gan.discriminator(Q, gan)
    D_loss = -mean(log(D_real) + log(1-D_fake))
    G_loss = -mean(log(D_fake))
    D_loss, G_loss
end

"""
D_loss, G_loss = wgan(P::PyObject, Q::PyObject, gan::GAN)

return discriminator loss and generator loss for 1 Wasserstein
`P` is the real distribution, `Q` is the generated distribution, 
No constraint is imposed on discriminator
`clamp` is required for the discriminator weights
"""
function wgan(P::PyObject, Q::PyObject, gan::GAN)
    D_real = gan.discriminator(P, gan)
    D_fake = gan.discriminator(Q, gan)
    D_loss = mean(D_fake)-mean(D_real)
    G_loss = -mean(D_fake)
    D_loss, G_loss
end

"""
D_loss, G_loss = rklgan(P::PyObject, Q::PyObject, gan::GAN)

return discriminator loss and generator loss for reverse KL divergence
`P` is the real distribution, `Q` is the generated distribution, 
`discriminator` is a critic function that outputs values in (0,1) (e.g. the last activation function is sigmoid)
"""
function rklgan(P::PyObject, Q::PyObject, gan::GAN)
    D_real = gan.discriminator(P, gan)
    D_fake = gan.discriminator(Q, gan)
    G_loss = mean(log((1-D_fake)/D_fake))
    D_loss = -mean(log(D_fake)+log(1-D_real))
    D_loss, G_loss
end


"""
D_loss, G_loss = lsgan(P::PyObject, Q::PyObject, gan::GAN)

return discriminator loss and generator loss for least square
`P` is the real distribution, `Q` is the generated distribution, 
`discriminator` is a critic function that outputs values in (0,1) (e.g. the last activation function is sigmoid)
1 for real, 0 for fake
"""
function lsgan(P::PyObject, Q::PyObject, gan::GAN)
    D_real = gan.discriminator(P, gan)
    D_fake = gan.discriminator(Q, gan)
    D_loss = mean((D_real-1)^2+D_fake^2)
    G_loss = mean((D_fake-1)^2)
    D_loss, G_loss
end
#######################################################

function build!(gan::GAN)
    z = normal(gan.batch_size, gan.latent_dim)
    variable_scope("generator_$(gan.ganid)") do
        gan.fake_data = gan.generator(z, gan)
    end
    gan.true_data = choice(gan.dat, gan.batch_size)
    if ismissing(gan.loss); gan.loss="jsgan"; end
    if isa(gan.loss, String)
        gan.loss = eval(Meta.parse(gan.loss))
    end
    variable_scope("discriminator_$(gan.ganid)") do
        gan.d_loss, gan.g_loss = gan.loss(gan.true_data, gan.fake_data, gan)
    end
    gan.d_vars = get_collection("discriminator_$(gan.ganid)")
    gan.g_vars = get_collection("generator_$(gan.ganid)")
    gan.update = Array{PyCall.PyObject}(get_collection(UPDATE_OPS))
end 

function sample(gan::GAN, n::Int64)
    local out
    z = normal(n, gan.latent_dim)
    gan.is_training = false
    variable_scope("generator_$(gan.ganid)") do
        out = gan.generator(z, gan)
    end
    gan.is_training = placeholder(true, shape=[])
    out
end

