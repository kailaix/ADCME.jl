using LinearAlgebra
using PyPlot 
using Statistics
using Distributions
using ProgressMeter

σ = 1
τ = sqrt(10)
μ = 5
n = 5



X = [9.37;10.18;9.16;11.60;10.33]
μn = mean(X)*n/σ^2/(n/σ^2 + 1/τ^2) + μ*(1/τ^2)/(n/σ^2 + 1/τ^2)
τn2 = 1/(n/σ^2+1/τ^2)

# Hyperparameters
# proposal step size
δ = 3/100
burnin = 2000
N = 10000
θ0 = 0.0

function logf(θ)
    -sum((X.-θ).^2)/2σ^2 - (θ-μ)^2/2τ^2
end

function proposal(x)
    x + (rand()-0.5)*2 * δ
end


sim = zeros(N)
sim[1] = θ0
@showprogress for i = 1:N-1
    x = sim[i]
    x_star = proposal(x)
    Δ =  logf(x_star) - logf(x)
    if log(rand())<Δ
        sim[i+1] = x_star 
    else 
        sim[i+1] = x
    end
end
L = logf.(sim)


# sim = sim[burnin+1:end]
figure(figsize=(12,4))
subplot(131)
plot(sim)
plot(1:length(sim), ones(length(sim))*μn, "--")
title("\$\\theta\$ Value")
xlabel("Iteration")
ylabel("\$\\theta\$")
subplot(132)
plot(L)
title("Log likelihood")
xlabel("Iteration")
ylabel("Log likelihood")
subplot(133)
hist(sim[burnin+1:end], density=true, bins=50)
y = pdf.(Normal(μn, sqrt(τn2)), LinRange(8.5,11.5,100))
plot(LinRange(8.5,11.5,100), y, label="Exact")
title("Distribution")
xlabel("\$\\theta\$")
ylabel("Density")
legend()
tight_layout()

