# rewrite of [TensorFlow Probability Example](https://medium.com/tensorflow/an-introduction-to-probabilistic-programming-now-available-in-tensorflow-probability-6dcc003ca29e) in `ADCME`
using Statistics
using ADCME
using PyCall
using PyPlot
using DelimitedFiles

if !("challenger_data.csv" in readdir("."))
    download("https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter2_MorePyMC/data/challenger_data.csv", "challenger_data.csv")
end

F = readdlm("challenger_data.csv",',', header=true)[1][:,2:3]
I = ones(Bool, size(F,1))
for i = 1:size(F,1)
    if F[i,2]=="NA" || F[i,2] == "Challenger Accident"
        I[i] = false
    end
end
F = Array{Float32}(F[I,:]);

sess = Session()

temperature = constant(F[:,1], dtype=Float32)
D = constant(F[:,2], dtype=Float32)

function challenger_joint_log_prob(D, temperature, α, β)
    rv_alpha = Normal(loc=0., scale=1000.)
    rv_beta = Normal(loc=0., scale=1000.)
    logistic_p = 1.0/(1+exp(β*temperature+α ))
    rv_observed = Bernoulli(probs = logistic_p)
    
    return log_prob(rv_alpha, α) + log_prob(rv_beta, β) + 
        sum(log_prob(rv_observed, D))
end

number_of_steps = 40000 
burnin = 38000
initial_chain_state = [
    constant(0.0, dtype=Float32),
    constant(0.0, dtype=Float32)
]

function unnormalized_posterior_log_prob(args...)
    challenger_joint_log_prob(D, temperature,args...)
end

unconstraining_bijectors = [Identity(), Identity()]


variable_scope("mcmc", reuse=AUTO_REUSE) do
global step_size = get_variable(
        "step_size",
        initializer=constant(0.5, dtype=Float32),
        trainable=false
    )
end

ham=TransformedTransitionKernel(HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=2,
        step_size_update_fn=make_simple_step_size_update_policy(),
        state_gradients_are_stopped=true,
        step_size=step_size),
        bijector=unconstraining_bijectors)

states,is_accepted_, kernel_results = sample_chain(
    num_results = number_of_steps,
    num_burnin_steps = burnin,
    current_state=initial_chain_state,
    kernel=ham
    )

init(sess)
posterior_α, posterior_β, is_accepted =  run(sess, [states[1], states[2], is_accepted_])

println("Accepted rate=", sum(is_accepted)/length(is_accepted))

function logistic(x, beta, alpha)
    return 1.0 ./ (1.0 .+ exp.(beta*x .+ alpha) )
end

figure()
xval = LinRange(minimum(F[:,1])-1.0, maximum(F[:,1])+1.0, 150)
y1 = logistic(xval, posterior_β[end-2000], posterior_α[end-2000])
y2 = logistic(xval, posterior_β[end-8], posterior_α[end-8])
y3 = logistic(xval, mean(posterior_β), mean(posterior_α))
scatter(F[:,1], F[:,2])
plot(xval, y1, label="posterior1")
plot(xval, y2, label="posterior2")
plot(xval, y3, label="mean")
ylim(-0.1,1.1)
legend()

# challenger
figure()
plt.hist(logistic(31, posterior_β[end-2000:end], posterior_α[end-2000:end]), 50)
