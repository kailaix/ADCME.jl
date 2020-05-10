# Variational Autoencoder

Let's see how to implement an autoencoder for generating MNIST images in ADCME. The mathematics underlying autoencoder is the Bayes formula

$$p(z|x) = \frac{p(x|z)p(z)}{p(x)}$$

where $x$ a sample from the data distribution and $z$ is latent variables. To model the data distribution given the latent variable, $p(x|z)$, we use a deep generative neural network $g_\phi$ that takees $z$ as the input and outputs $x$. This gives us the approximate $p_\phi(x|z) \approx p(x|z)$. 

However, computing $p(z|x)$ directly can be intractable. To this end, we approximate the posterior using $z\sim \mathcal{N}(\mu_x, \sigma_x^2I)$, where $\mu_x$ and $\sigma_x$ are both encoded using neural networks, where $x$ is the input to the neural network. In this way, we obtain an approximate posterior 

$$p_w(z|x) = \frac{1}{(\sqrt{2\pi \sigma_x^2})^d}\exp\left( -\frac{\|z-\mu_x)\|^2}{2\sigma_x^2} \right) \tag{1}$$

How can we choose the correct weights and biases $\phi$ and $w$? The idea is to minimize the discrepancy between the true posterior and the approximate posterior Equation (1). We can use the KL divergence, which is a metric for measuring the discrepancy between two distributions

$$\mathrm{KL}(p_w(z|x)|| p(z|x)) = \mathbb{E}_{p_w}(\log p_w(z|x) - \log p(z|x)) \tag{2}$$

However, computing Equation 2 is still intractable since we do not know $\log p(z|x)$. Instead, we seek to minimize a maximize bound of the KL divergence 

$$\begin{aligned}
\mathrm{ELBO} &=  \log p(x) - \mathrm{KL}(p_w(z|x)|| p(z|x))\\
& = \mathbb{E}_{p_w}( \log p(z,x) - \log p_w(z|x)) \\ 
& = \mathbb{E}_{p_w(z|x)}[\log p_\phi(x|z)] - \mathrm{KL}(p_w(z|x) || p(z))
\end{aligned}$$

Note that we assumed that the generative neural network $g_\phi$ is sufficiently expressive so $p_\phi(y|z)\approx p(y|z)$. Additionally, because KL divergence is always positive

$$\mathrm{ELBO} \leq \log p(x)\tag{3}$$

Equation (3) justifies the name "evidence lower bound". 

Let's consider how to compute ELBO for our autoencoder. For the marginal likelihood term $\mathbb{E}_{p_w(z|x)}[\log p_\phi(x|z)]$, for each given sample $y$, we can calculate the mean and covariance of $z$, namely $\mu_x$ and $\sigma_x^2I$. We sample $z_i\sim \mathcal{N}(\mu_x, \sigma_x^2I)$ and plug them into $g_\phi$ and obtain the outputs $x_i = g_\phi(z_i)$. If we assume that the decoder model is subject to Bernoulli distribution $x \sim Ber(g_\phi(z))$ (in this case we have $g_\phi(z)\in [0,1]$), we have the approximation 

$$\mathbb{E}_{p_w(z|x)}[\log p_\phi(x|z)] \approx \frac{1}{n}\sum_{i=1}^n \left[x_i\log (g_\phi(z_i)) + (1-x_i) \log(1-g_\phi(z_i))\right]\tag{4}$$

Now let us consider the second term $\mathrm{KL}(p_w(z|x) || p(z))$. If we assign a unit Gaussian prior on $z$, we have

$$\begin{aligned}
\mathrm{KL}(p_w(z|x) || p(z)) &= \mathbb{E}_{p_w}[\log(p_w(z|x)) - \log(p(z)) ]\\ 
& =  \mathbb{E}_{p_w}\left[-\frac{\|z-\mu_x\|^2}{2\sigma_x^2} - d\log(\sigma_x) + \frac{\|z\|^2}{2} \right]\\
& = -d - d\log(\sigma_x) +\frac{1}{2} \|\mu_x\|^2 + \frac{d}{2}\sigma_x^2 
\end{aligned} \tag{5}$$

Using Equation 4 and 5 we can formulate a loss function, which we can use a stochastic gradient descent method to minimize. 

The following code is an example of applying the autoencoder to learn a data distribution from MNIST dataset. Here is the result using this script:

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/vae.png?raw=true)

```julia
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

train_x = MNIST.traintensor(Float64);
train_x = Array(reshape(train_x, :, 60000)');

sess = Session(); init(sess)
for i = 1:100
    step(i)
end
```
