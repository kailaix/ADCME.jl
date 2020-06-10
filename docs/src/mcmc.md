# Uncertainty Quantification of Neural Networks in Physics Informed Learning using MCMC 

In this section, we consider uncertainty quantification of a neural network prediction using Markov Chain Monte Carlo.
The idea is that we use MCMC to sample from the posterior distribution of the **neural network weights and biases**.
We consider an inverse problem, where the governing equation is a heat equation in an 1D interval $[0,1]$.

The simulation is conducted over a time horizon $[0,1]$. We record the temperature $u(0,t)$ on the left of the interval.
The diffusivity coefficient $\kappa(x)$ is assumed unknown and will be estimated from the temperature record. $\kappa(x)$
is approximated by a neural network

$$\kappa(x) = \mathcal{NN}_{w}(x)$$

Here $w$ is the neural network weights and biases.

First of all, we define a function `simulate` that takes in the diffusivity coefficient, and returns the solution of the PDE.

File `heateq.jl`:

```julia
using ADCME
using PyPlot
using ADCME
using PyCall
using ProgressMeter
using Statistics
using MAT
using DelimitedFiles
mpl = pyimport("tikzplotlib")

function simulate(κ)
    κ = constant(κ)
    m = 50
    n = 50
    dt = 1 / m
    dx = 1 / n
    F = zeros(m + 1, n)
    xi = LinRange(0, 1, n + 1)[1:end - 1]
    f = (x, t)->exp(-50(x - 0.5)^2)
    for k = 1:m + 1
        t = (k - 1) * dt
        F[k,:] = dt * f.(xi, t)
    end

    λ = κ*dt/dx^2
    mask = ones(n-1)
    mask[1] =  2.0
    A = spdiag(n, -1=>-λ[2:end], 0=>1+2λ, 1=>-λ[1:end-1].*mask)


    function condition(i, u_arr)
        i <= m + 1
    end

    function body(i, u_arr)
        u = read(u_arr, i - 1)
        rhs = u + F[i]
        u_next = A \ rhs
        u_arr = write(u_arr, i, u_next)
        i + 1, u_arr
    end

    F = constant(F)
    u_arr = TensorArray(m + 1)
    u_arr = write(u_arr, 1, zeros(n))
    i = constant(2, dtype = Int32)
    _, u = while_loop(condition, body, [i, u_arr])
    u = set_shape(stack(u), (m + 1, n))
end
```

We set up the geometry as follows

```julia
n = 50
xi = LinRange(0, 1, n + 1)[1:end - 1]
x = Array(LinRange(0, 1, n+1)[1:end-1])
```


# Forward Computation

The forward computation is run with an analytical $\kappa(x)$, given by
$$\kappa(x) = 5x^2 + \exp(x) + 1.0$$
We can generate the code using the following code:

```julia
include("heateq.jl")

κ = @. 5x^2 + exp(x) + 1.0
out = simulate(κ)
obs = out[:,1]

sess = Session(); init(sess)
obs_ = run(sess, obs)

writedlm("obs.txt", run(sess, out))
o = run(sess, out)
pcolormesh( (0:49)*1/50, (0:50)*1/50, o, rasterized=true)
xlabel("\$x\$")
ylabel("\$t\$")
savefig("solution.png")

figure()
plot((0:50)*1/50, obs_)
xlabel("\$t\$")
ylabel("\$u(0,t)\$")
savefig("obs.png")

figure()
plot(x, κ)
xlabel("\$x\$")
ylabel("\$\\kappa\$")
savefig("kappa.png")
```



| Solution            | Observation    | $\kappa(x)$      |
| ------------------- | -------------- | ---------------- |
| ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mcmc/./solution.png?raw=true)  | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mcmc/./obs.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mcmc/./kappa.png?raw=true) |



# Inverse Modeling

Although it is possible to use MCMC to solve the inverse problem, the convergence can be very slow if our initial guess is
far away from the solution.

Therefore, we first solve the inverse problem by solving a PDE-constrained optimization problem.
We use the [`BFGS!`](@ref) optimizer. Note we do not need to solve the inverse problem very accurately because in Bayesian approaches,
the solution is interpreted as a probability, instead of a point estimation.

```julia
include("heateq.jl")
using PyCall
using Distributions
using Optim
using LineSearches
reset_default_graph()
using Random; Random.seed!(2333)
w = Variable(ae_init([1,20,20,20,1]), name="nn")
κ = fc(x, [20,20,20,1], w, activation="tanh") + 1.0
u = simulate(κ)
obs = readdlm("obs.txt")
loss = sum((u[:,1]-obs[:,1])^2)
loss = loss*1e10

sess = Session(); init(sess)


BFGS!(sess, loss)

κ1 = @. 5x^2 + exp(x) + 1.0
plot(x, run(sess, κ), "+--", label="Estimation")
plot(x, κ1, label="Reference")
legend()
savefig("inversekappa.png")
```



![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mcmc/./inversekappa.png?raw=true)

We also save the solution for MCMC

```julia
matwrite("nn.mat", Dict("w"=>run(sess, w)))
```


# Uncertainty Quantification

Finally, we are ready to conduct uncertainty quantification using MCMC.
We will use the `Mamba` package, which provides MCMC utilities. We will use the random walk MCMC because of its simplicity.

```julia
include("heateq.jl")
using PyCall
using Mamba
using ProgressMeter
using PyPlot
```

The neural network weights and biases are conveniently expressed as a `placeholder`.
This allows us to `sample` from a distribution of weights and biases easily.

```julia
w = placeholder(ae_init([1,20,20,20,1]))
κ = fc(x, [20,20,20,1], w, activation="tanh") + 1.0
u = simulate(κ)
obs = readdlm("obs.txt")

sess = Session(); init(sess)
w0 = matread("nn.mat")["w"]
```

The log likelihood function (up to an additive constant) is given by
$$-{{{{\left\| {{u_{{\rm{est}}}}(w) - {u_{{\rm{obs}}}}} \right\|}^2}} \over {2{\sigma ^2}}} - {{{{\left\| w \right\|}^2}} \over {2\sigma _w^2}}$$

The absolute value of $\sigma$ and $\sigma_w$ does not really matter. Only their ratios matter. Let's fix $\sigma = 1$. What is the interpretation of $\sigma_w$?

A large $\sigma_w$ means very wide prior, and a small $\sigma_w$ means a very narrow prior.
The relative value $\sigma/\sigma_w$ implies **the strength of prior influence**.
Typically, we can choose a very large $\sigma_w$ so that the prior does not influence the posterior too much. 

```julia
σ = 1.0
σx = 1000000.0
function logf(x)
    y = run(sess, u, w=>x)
    -sum((y[:,1] - obs[:,1]).^2)/2σ^2 - sum(x.^2)/2σx^2
end

n = 5000
burnin = 1000
sim = Chains(n, length(w0))
```

A second important parameter is the scale (0.002 in the following code). It controls **the uncertainty bound width** via the way we generate the random numbers.

```julia
θ = RWMVariate(copy(w0), 0.001ones(length(w0)), logf, proposal = SymUniform)
```

An immediate consequence is that the smaller the scale factor we use, the narrower the uncertainty band will be.
In sum, we have two important parameters--relative standard deviation and the scaling factor--to control our uncertainty bound.

```julia

@showprogress for i = 1:n 
    sample!(θ)
    sim[i,:,1] = θ
end


v = sim.value
K = zeros(length(κ), n-burnin)
@showprogress for i = 1:n-burnin
    ws = v[i+burnin,:,1]
    K[:,i] = run(sess, κ, w=>ws)
end 

kappa = mean(K, dims=2)[:]
k_std = std(K, dims=2)[:]
figure()
κ1 = @. 5x^2 + exp(x) + 1.0
PyPlot.plot(x, kappa, "--", label="Posterior Mean")
PyPlot.plot(x, κ1, "r", label="True")
PyPlot.plot(x, run(sess, κ, w=>w0), label="Point Estimation")
fill_between(x, kappa-3k_std, kappa+3k_std, alpha=0.5)
legend()
xlabel("x")
ylabel("\$\\kappa(x)\$")
savefig("kappa_mcmc.png")
```

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mcmc/./kappa_mcmc.png?raw=true)