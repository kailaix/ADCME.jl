using ADCME
using PyPlot 
using ProgressMeter
using Statistics

function f(x, σ)
    ε = randn(size(x)...) * σ
    return 10 * sin.(2π*x) + ε
end

batch_size = 32
noise = 1.0

X = reshape(LinRange(-0.5, 0.5, batch_size)|>Array, :, 1)
y = f(X, noise)
y_true = f(X, 0.0)

close("all")
scatter(X, y, marker="+", label="Training Data")
plot(X, y_true, label="Truth")
legend()
savefig("bnn_training_data.png")

mutable struct VariationalLayer
    units
    activation
    prior_σ1
    prior_σ2
    prior_π1
    prior_π2
    Wμ
    bμ
    Wρ
    bρ
    init_σ
end

function VariationalLayer(units; activation=relu, prior_σ1=1.5, prior_σ2=0.1,
        prior_π1=0.5)
    init_σ = sqrt(
        prior_π1 * prior_σ1^2 + (1-prior_π1)*prior_σ2^2
    )
    VariationalLayer(units, activation, prior_σ1, prior_σ2, prior_π1, 1-prior_π1,
                        missing, missing, missing, missing, init_σ)
end

function kl_loss(vl, w, μ, σ)
    dist = ADCME.Normal(μ,σ)
    return sum(logpdf(dist, w)-logprior(vl, w))
end

function logprior(vl, w)
    dist1 = ADCME.Normal(constant(0.0), vl.prior_σ1)
    dist2 = ADCME.Normal(constant(0.0), vl.prior_σ2)
    log(vl.prior_π1*exp(logpdf(dist1, w)) + vl.prior_π2*exp(logpdf(dist2, w)))
end

function (vl::VariationalLayer)(x)
    x = constant(x)
    if ismissing(vl.bμ)
        vl.Wμ = get_variable(vl.init_σ*randn(size(x,2), vl.units))
        vl.Wρ = get_variable(zeros(size(x,2), vl.units))
        vl.bμ = get_variable(vl.init_σ*randn(1, vl.units))
        vl.bρ = get_variable(zeros(1, vl.units))
    end
    Wσ = softplus(vl.Wρ)
    W = vl.Wμ + Wσ.*normal(size(vl.Wμ)...) 
    bσ = softplus(vl.bρ)
    b = vl.bμ + bσ.*normal(size(vl.bμ)...)
    loss = kl_loss(vl, W, vl.Wμ, Wσ) + kl_loss(vl, b, vl.bμ, bσ)
    out = vl.activation(x * W + b)
    return out, loss 
end

function neg_log_likelihood(y_obs, y_pred, σ)
    y_obs = constant(y_obs)
    dist = ADCME.Normal(y_pred, σ)
    sum(-logpdf(dist, y_obs))
end

ipt = placeholder(X)
x, loss1 = VariationalLayer(20, activation=relu)(ipt)
x, loss2 = VariationalLayer(20, activation=relu)(x)
x, loss3 = VariationalLayer(1, activation=x->x)(x)

loss_lf = neg_log_likelihood(y, x, noise)
loss = loss1 + loss2 + loss3 + loss_lf

opt = AdamOptimizer(0.08).minimize(loss)
sess = Session(); init(sess)
@showprogress for i = 1:5000
    run(sess, opt)
end


X_test = reshape(LinRange(-1.5,1.5,32)|>Array, :, 1)
y_pred_list = []
@showprogress for i = 1:10000
    y_pred = run(sess, x, ipt=>X_test)
    push!(y_pred_list, y_pred)
end

y_preds = hcat(y_pred_list...)

y_mean = mean(y_preds, dims=2)[:]
y_std = std(y_preds, dims=2)[:]

close("all")
plot(X_test, y_mean)
scatter(X[:], y[:], marker="+")
fill_between(X_test[:], y_mean-2y_std, y_mean+2y_std, alpha=0.5)
savefig("bnn_prediction.png")
