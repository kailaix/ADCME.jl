# Convolutional Neural Network 

The convolutional neural network (CNN) is one of the key building blocks for deep learning. Mathematically, it is a linear operator whose actions are "local", in the sense that each output only depends on a small number of inputs. These actions share the same kernel functions, and the sharing reduces the number of parameters significantly. 

One remarkable feature of CNNs is that they are massively parallelizable. The parallesim makes CNNs very efficient on GPUs, which are good at doing a large number of simple tasks at the same time. 

In the practical use of CNNs, we can stick to images, which have four dimensions: batch number, height, width, and channel. A CNN transforms the images to another images with the same four dimensions, but possibly with different heights, widths, and channels. In the following script, we use CNNs instead of fully connected neural networks to train a variational autoencoder. Readers can compare the results with [this article](./vae.md). 

You are also encouraged to run the same script on CPUs and GPUs. You might get surprised at the huge performance gap for training CNNs on these two different computing environment. We also observe some CNN artifacts (the dots in the images).

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/cnn.png?raw=true)


```julia
using ADCME
using PyPlot
using MLDatasets
using ProgressMeter
using Images

mutable struct Generator
    dim_z::Int64
    layers::Array
end

function Generator( dim_z::Int64 = 100, ngf::Int64 = 8)
    layers = [
        Conv2DTranspose(ngf*32, 4, 1, use_bias=false)
        BatchNormalization()
        relu
        Conv2DTranspose(ngf*16, 4, 1, padding="same", use_bias=false)
        BatchNormalization()
        relu
        Conv2DTranspose(ngf*8, 4, 1, use_bias=false)
        x -> pad(x, [
            0 0 
            0 1
            0 1
            0 0
        ])
        BatchNormalization()
        relu
        Conv2DTranspose(1, 4, 4, use_bias = false)
        BatchNormalization()
        sigmoid
    ]
    Generator(dim_z, layers)
end

function (g::Generator)(z)
    z = constant(z)
    z = reshape(z, (-1, 1, 1, g.dim_z))
    @info size(z)
    for l in g.layers
        z = l(z)
        @info size(z)
    end
    return z 
end
function encoder(x, n_hidden, n_output, rate)
    local μ, σ
    variable_scope("encoder") do 
        y = dense(x, n_hidden, activation = "elu")
        y = dropout(y, rate, ADCME.options.training.training)
        y = dense(y, n_hidden, activation = "tanh")
        y = dropout(y, rate, ADCME.options.training.training)
        y = dense(y, 2n_output)
        μ = y[:, 1:n_output]
        σ = 1e-6 + softplus(y[:,n_output+1:end])
    end
    return μ, σ
end

function decoder(z, n_hidden, n_output, rate)
    Generator(dim_z)(z)
end

function autoencoder(xh, x, dim_img, dim_z, n_hidden, rate)
    μ, σ = encoder(xh, n_hidden, dim_z, rate)
    z = μ + σ .* tf.random_normal(size(μ), 0, 1, dtype=tf.float64)
    y = decoder(z, n_hidden, dim_img, rate)
    y = clip(y, 1e-8, 1-1e-8)
    y = tf.reshape(y, (-1,32^2))

    marginal_likelihood = sum(x .* log(y) + (1-x).*log(1-y), dims=2)
    KL_divergence = 0.5 * sum(μ^2 + σ^2 - log(1e-8 + σ^2) - 1, dims=2)

    marginal_likelihood = mean(marginal_likelihood)
    KL_divergence = mean(KL_divergence)

    ELBO = marginal_likelihood - KL_divergence
    loss = -ELBO 
    return y, loss, -marginal_likelihood, KL_divergence
end

function step(epoch)
    tx = train_x[1:batch_size,:]
    @showprogress for i = 1:div(60000, batch_size)
        idx = Array((i-1)*batch_size+1:i*batch_size)
        run(sess, opt, x=>train_x[idx,:])
    end
    y_, loss_, ml_, kl_ = run(sess, [y, loss, ml, KL_divergence],
            feed_dict = Dict(
                ADCME.options.training.training=>false, 
                x => tx
            ))
    println("epoch $epoch: L_tot = $(loss_), L_likelihood = $(ml_), L_KL = $(kl_)")

    close("all")
    for i = 1:3
        for j = 1:3
            k = (i-1)*3 + j 
            img = reshape(y_[k,:], 32, 32)'|>Array
            img = imresize(img, 28, 28)
            subplot(3,3,k)
            imshow(img)
        end
    end
    savefig("result$epoch.png")
end



n_hidden = 500
rate = 0.1
dim_z = 100
dim_img = 32^2
batch_size = 32
ADCME.options.training.training = placeholder(true)
x = placeholder(Float64, shape = [32, 32^2])
xh = x
y, loss, ml, KL_divergence = autoencoder(xh, x, dim_img, dim_z, n_hidden, rate)
opt = AdamOptimizer(1e-3).minimize(loss)

train_x_ = MNIST.traintensor(Float64);
train_x = zeros(60000, 32^2)
for i = 1:60000
    train_x[i,:] = imresize(train_x_[:, :, i], 32, 32)[:]
end

sess = Session(); init(sess)
for i = 1:100
    @info i 
    step(i)
end
```