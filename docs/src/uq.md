# Uncertainty Quantification


## Theory

### Basic Model 
We consider a physical model

$$\begin{aligned}
y &= h(s) + \delta \\ 
s &= g(z) + \epsilon
\end{aligned}\tag{1}$$

Here $\delta$ and $\epsilon$ are independent Gaussian noises. $s\in \mathbb{R}^m$ is the physical quantities we are interested, and $y\in \mathbb{R}^n$ is the measurement. $z$ is a hidden factor that $s$ depends on.  $\delta$ can be interpreted as the measurement error

$$\mathbb{E}(\delta\delta^T) = R$$

$\epsilon$ is interpreted as our prior for $s$

$$ \mathbb{E}(\epsilon\epsilon^T) = Q$$


### Linear Gaussian Model 
When the standard deviation of $\epsilon$ is small, we can safely approximate $h(s)$ using its linearized form 

$$h(s)\approx \nabla h(s_0) (s-s_0) + h(s_0) := \mu + H s$$

Here
$$\mu = h(s_0) - \nabla h(s_0) s_0\quad H = \nabla h(x_0)$$

Therefore, we have an approximate governing equation for Equation 1:

$$\begin{aligned}
y &= H s + \mu + \delta\\
s &= g(z) + \epsilon
\end{aligned}\tag{2}$$

Using Equation 2, we have

$$\begin{aligned}
\mathbb{E}(y) & = H g(z) + \mu \\ 
\Psi = \text{cov}(y) & = \mathbb{E}\left[(H (x-g(z)) + \delta )(H (x-g(z)) + \delta )^T \right] = H QH^T + R
\end{aligned}$$

### Bayesian Inversion 

The posterior of $s$ given the observation is also Gaussian, with a mean vector $\hat s$ and a covariance matrix $\Sigma$

$$s \sim \mathcal{N}(\hat s, \Sigma)$$

The quantity $s$ and $\Sigma$ can be computed by first solving a $(m+n)\times (m+n)$ matrix

$$\boxed{
    \begin{bmatrix}
    \Psi & H \\ 
    H^T & 0 
    \end{bmatrix}\begin{bmatrix}
    \Lambda^T \\ 
    M 
    \end{bmatrix} = \begin{bmatrix}
    HQ\\ 
    I 
    \end{bmatrix}
}$$


$\hat s$ and $\Sigma$ are computed using 

$$\begin{aligned}
\hat s &= \Lambda y \\ 
\Sigma &= Q - M - C^T \Lambda^T
\end{aligned}$$


In ADCME, we provide the implementation [`uq`](@ref)

```julia
s, Σ = uq(y, H, R, Q)
```

## Example 1: UQ for Parameter Inverse Problems

We consider a simple example for 2D Poisson problem.

$$\begin{aligned}
\nabla (K(x, y) \nabla u(x, y)) &= 1 & \text{ in } \Omega\\ 
u(x,y) &= 0  & \text{ on } \partial \Omega
\end{aligned}$$

where $K(x,y) = e^{c_1 + c_2 x + c_3 y}$. 

Here $c_1$, $c_2$, $c_3$ are parameter to be estimated. We first generate data using $c_1=1,c_2=2,c_3=3$ and add Gaussian noise $\mathcal{N}(0, 10^{-3})$ to 64 observation in the center of the domain $[0,1]^2$. We run the inverse modeling and obtain an estimation of $c_i$'s. Finally, we use [`uq`](@ref) to conduct the uncertainty quantification. We assume $\text{error}_{\text{model}}=0$. 

The following plot shows the estimated mean together with 2 standard deviations. 


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/uq_poisson.png?raw=true)

```julia
using ADCME
using PyPlot
using PoreFlow


Random.seed!(233)
idx = fem_randidx(100, m, n, h)

function poisson1d(c)
    m = 40
    n = 40
    h = 0.1
    bdnode = bcnode("all", m, n, h)
    c = constant(c)
    xy = gauss_nodes(m, n, h)
    κ = exp(c[1] + c[2] * xy[:,1] + c[3]*xy[:,2])
    κ = compute_space_varying_tangent_elasticity_matrix(κ, m, n, h)
    K = compute_fem_stiffness_matrix1(κ, m, n, h)
    K, _ = fem_impose_Dirichlet_boundary_condition1(K, bdnode, m, n, h)
    rhs = compute_fem_source_term1(ones(4m*n), m, n, h)
    rhs[bdnode] .= 0.0
    sol = K\rhs
    sol[idx]
end

c = Variable(rand(3))
y = poisson1d(c)
Γ = gradients(y, c) 
Γ = reshape(Γ, (100, 3))

# generate data 
sess = Session(); init(sess)
run(sess, assign(c, [1.0;2.0;3.0]))
obs = run(sess, y) + 1e-3 * randn(100)

# Inverse modeling 
loss = sum((y - obs)^2)
init(sess)
BFGS!(sess, loss)

y = obs 
H = run(sess, Γ)
R = (2e-3)^2 * diagm(0=>ones(100))
X = run(sess, c)
Q = diagm(0=>ones(3))
m, V = uqlin(y, H, R, X, Q)
plot([1;2;3], [1.;2.;3.], "o", label="Reference")
errorbar([1;2;3],m + run(sess, c), yerr=2diag(V), label="Estimated")
legend()
```

!!! info "The choice of $R$" 
    The standard deviation $2\times 10^{-3}$ consists of the model error ($10^{-3}$) and the measurement error $10^{-3}$. 

## Example 2: UQ for Function Inverse Problems

In this example, let us consider uncertainty quantification for function inverse problems. We consider the same problem as Example 1, except that $K(x,y)$ is represented by a neural network (the weights and biases are represented by $\theta$)

$$\mathcal{NN}_\theta:\mathbb{R}^2 \rightarrow \mathbb{R}$$

We consider a highly nonlinear $K(x,y)$

$$K(x,y) = 0.1 + \sin x+ x(y-1)^2  + \log (1+y)$$


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/uq_poisson2-0.png?raw=true)

The left panel above shows the exact $K(x,y)$ and the learned $K(x,y)$. We see we have a good approximation but with some error. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/uq_poisson2.png?raw=true)

The left panel above shows the exact solution while the right panel shows the reconstructed solution after learning. 

We apply the UQ method and obtain the standard deviation plot on the left, together with absolute error on the right. We see that our UQ estimation predicts that the right side has larger uncertainty, which is true in consideration of the absolute error. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/uq_poisson2-1.png?raw=true)


```julia
using Revise
using ADCME
using PyPlot
using PoreFlow


m = 40
n = 40
h = 1/n
bdnode = bcnode("all", m, n, h)
xy = gauss_nodes(m, n, h)
xy_fem = fem_nodes(m, n, h)

function poisson1d(κ)
    κ = compute_space_varying_tangent_elasticity_matrix(κ, m, n, h)
    K = compute_fem_stiffness_matrix1(κ, m, n, h)
    K, _ = fem_impose_Dirichlet_boundary_condition1(K, bdnode, m, n, h)
    rhs = compute_fem_source_term1(ones(4m*n), m, n, h)
    rhs[bdnode] .= 0.0
    sol = K\rhs
end

κ = @. 0.1 + sin(xy[:,1]) + (xy[:,2]-1)^2 * xy[:,1] + log(1+xy[:,2])
y = poisson1d(κ)

sess = Session(); init(sess)
SOL = run(sess, y)


# inverse modeling 
κnn = squeeze(abs(ae(xy, [20,20,20,1])))
y = poisson1d(κnn)

using Random; Random.seed!(233)
idx = fem_randidx(100, m, n, h)
obs = y[idx]
OBS = SOL[idx]
loss = sum((obs-OBS)^2)

init(sess)
BFGS!(sess, loss, 200)

figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fem_points(SOL, m, n, h)
subplot(122)
visualize_scalar_on_fem_points(run(sess, y), m, n, h)
plot(xy_fem[idx,1], xy_fem[idx,2], "o", c="red", label="Observation")
legend()


figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_gauss_points(κ, m, n, h)
title("Exact \$K(x, y)\$")
subplot(122)
visualize_scalar_on_gauss_points(run(sess, κnn), m, n, h)
title("Estimated \$K(x, y)\$")


H = gradients(obs, κnn) 
H = run(sess, H)
y = OBS 
hs = run(sess, obs)
R = (1e-1)^2*diagm(0=>ones(length(obs)))
s = run(sess, κnn)
Q = (1e-2)^2*diagm(0=>ones(length(κnn)))
μ, Σ = uqnlin(y, hs, H, R, s, Q)

σ = diag(Σ)
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_gauss_points(σ, m, n, h)
title("Standard Deviation")
subplot(122)
visualize_scalar_on_gauss_points(abs.(run(sess, κnn)-κ), m, n, h)
title("Absolute Error")
```

## Example 3: UQ for Function Inverse Problem 

In this case, we consider a more challenging case, where $K$ is a function of the state variable, i.e., $K(u)$. $K$ is approximated by a neural network, but we need an iterative solver that involves the neural network to solve the problem 

$$\begin{aligned}
\nabla (K(u) \nabla u(x, y)) &= 1 & \text{ in } \Omega\\ 
u(x,y) &= 0  & \text{ on } \partial \Omega
\end{aligned}$$

 