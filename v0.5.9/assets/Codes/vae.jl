using ADCME
using PyPlot
using MLDatasets
using ProgressMeter


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
    local y 
    variable_scope("decoder") do 
        y = dense(z, n_hidden, activation="tanh")
        y = dropout(y, rate, ADCME.options.training.training)
        y = dense(y, n_hidden, activation="elu")
        y = dropout(y, rate, ADCME.options.training.training)
        y = dense(y, n_output, activation="sigmoid")
    end
    return y 
end

function autoencoder(xh, x, dim_img, dim_z, n_hidden, rate)
    μ, σ = encoder(xh, n_hidden, dim_z, rate)
    z = μ + σ .* tf.random_normal(size(μ), 0, 1, dtype=tf.float64)
    y = decoder(z, n_hidden, dim_img, rate)
    y = clip(y, 1e-8, 1-1e-8)

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
            img = reshape(y_[k,:], 28, 28)'|>Array
            subplot(3,3,k)
            imshow(img)
        end
    end
    savefig("result$epoch.png")
end



n_hidden = 500
rate = 0.1
dim_z = 20
dim_img = 28^2
batch_size = 128
ADCME.options.training.training = placeholder(true)
x = placeholder(Float64, shape = [128, 28^2])
xh = x
y, loss, ml, KL_divergence = autoencoder(xh, x, dim_img, dim_z, n_hidden, rate)
opt = AdamOptimizer(1e-3).minimize(loss)

# prepare data 
train_x = MNIST.traintensor(Float64);
train_x = Array(reshape(train_x, :, 60000)');

sess = Session(); init(sess)
for i = 1:100
    step(i)
end