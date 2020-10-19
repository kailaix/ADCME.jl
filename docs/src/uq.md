# Uncertainty Quantification

<!-- qunatifying uncertainty of neural networks in inverse problems using linearized Gaussian modeels -->
## Theory

### Basic Model 
We consider a physical model

$$\begin{aligned}
y &= h(s) + \delta \\ 
s &= g(z) + \epsilon
\end{aligned}\tag{1}$$

Here $\delta$ and $\epsilon$ are independent Gaussian noises. $s\in \mathbb{R}^m$ is the physical quantities we are interested in predicting, and $y\in \mathbb{R}^n$ is the measurement. $g$ is a function approximator, which we learn from observations in our inverse problem. $z$ is considered fixed for quantifying the uncertainty for a specific observation under small perturbation, although $z$ and $s$ may have complex dependency. 
$\delta$ can be interpreted as the measurement error

$$\mathbb{E}(\delta\delta^T) = R$$

$\epsilon$ is interpreted as our prior for $s$

$$\mathbb{E}(\epsilon\epsilon^T) = Q$$


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
\text{cov}(y) & = \mathbb{E}\left[(H (x-g(z)) + \delta )(H (x-g(z)) + \delta )^T \right] = H QH^T + R
\end{aligned}$$

### Bayesian Inversion 


#### Derivation 

From the model Equation 2 we can derive the joint distribution of $s$ and $y$, which is a multivariate Gaussian distribution

$$\begin{bmatrix}
x_1\\ 
x_2
\end{bmatrix}\sim \mathcal{N}\left( 
    \begin{bmatrix}
        g(z)\\ 
        Hg(z) + \mu 
    \end{bmatrix} \Bigg| \begin{bmatrix}
    Q & QH^T \\ 
    HQ & HQH^T + R
    \end{bmatrix}
     \right)$$
Here the covariance matrix $\text{cov}(s, y)$ is obtained via 

$$\text{cov}(s, y) = \mathbb{E}(s, Hs + \mu+\delta) = \mathbb{E}(s-g(z), H(s-g(z))) = \mathbb{E}((s-g(z))(s-g(z))^T) H^T = QH^T$$

Recall the formulas for conditional Gaussian distributions:

Given 

$$\begin{bmatrix}
s\\ 
y
\end{bmatrix}\sim \mathcal{N}\left( 
    \begin{bmatrix}
        \mu_1\\ 
        \mu_2
    \end{bmatrix} \Bigg| \begin{bmatrix}
    \Sigma_{11} & \Sigma_{12} \\ 
    \Sigma_{21} & \Sigma_{22}
    \end{bmatrix}
     \right)$$

We have 

$$x_1 | x_2 \sim \mathcal{N}(\mu_{1|2}, V_{1|2})$$

where 

$$\begin{aligned}
\mu_{1|2} &= \mu_1 + \Sigma_{12}\Sigma_{22}^{-1} (x_2-\mu_2)\\ 
V_{1|2} &= \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1}\Sigma_{21}
\end{aligned}$$

Let $x_1 = s$, $x_2 = y$, we have the following formula for Baysian inversion:

$$\begin{aligned}
\mu_{s|y} &= g(z) + QH^{T}(HQH^T + R)^{-1} (y - Hg(z) - \mu)\\ 
V_{s|y} &= Q - QH^T(HQH^T + R)^{-1} HQ 
\end{aligned}\tag{3}$$



#### Analysis 

Now we consider how to compute Equation 3. In practice, we should avoid direct inverting the matrix $HQH^T + R$ since the cost is cubic in the size of dimensions of the matrix. Instead, the following theorem gives us a convenient way to solve the problem 

!!! info "Theorem"
    Let $\begin{bmatrix}L \\ x^T\end{bmatrix}$ be the solution to 

    $$\begin{bmatrix}
    HQH^T + R & H g(z) \\ 
    g(z)^T H^T & 0 
    \end{bmatrix}\begin{bmatrix}
    L \\ 
    x^T
    \end{bmatrix} = \begin{bmatrix}
    HQ \\ 
    g(z)^T
    \end{bmatrix}\tag{4}$$

    Then we have 

    $$\begin{aligned}
    \mu_{s|y} = g(z) + L^T (y-\mu) \\ 
    V_{s|y} = Q - gx^T - QH^TL
    \end{aligned}\tag{5}$$


The linear system in Equation 5 is symmetric but may not be SPD and therefore we may encounter numerical difficulty when solving the linear system Equation 4. In this case, we can add perturbation $\varepsilon g^T g$ to the zero entry. 

!!! info "Theorem"
    If $\varepsilon> \frac{1}{4\lambda_{\min}}$, where $\lambda_{\min}$ is the minimum eigenvalue of $Q$, then the linear system in Equation 4 is SPD. 

The above theorem has a nice interpretation: typically we can choosee our prior for the physical quantity $s$ to be a scalar matrix $Q = \sigma_{{s}}^2 I$, where $\sigma_{s}$ is the standard deviation, then $\lambda_{\min} = \sigma_s^2$. This indicates that if we use a very concentrated prior, the linear system can be far away from SPD and requires us to use a large perturbation for numerical stability. Therefore, in the numerical example below, we choose a moderate $\sigma_s$. The alternative approach is to add the perturbation. 



In ADCME, we provide the implementation [`uq`](@ref)

```julia
s, Σ = uqlin(y-μ, H, R, gz, Q)
```

## Benchmark 

To show how the proposed method work compared to MCMC, we consider a model problem: estimating Young's modulus and Poisson's ratio from sparse observations. 

$$\begin{aligned}
\mathrm{div}\; \sigma &= f & \text{ in } \Omega \\ 
\sigma n &= 0 & \text{ on }\Gamma_N \\ 
u &= 0 & \text{ on }\Gamma_D \\ 
\sigma & = H\epsilon
\end{aligned}$$

Here the computational domain $\Omega=[0,1]\times [0,1.5]$. We fixed the left side ($\Gamma_D$) and impose an upward pressure on the right side. The other side is considered fixed. We consider the plane stress linear elasticity, where the constitutive relation determined by 

$$H = \frac{E}{(1+\nu)(1-2\nu)}\begin{bmatrix}
1-\nu & \nu & 0 \\ 
\nu & 1-\nu & 0 \\ 
0 & 0 & \frac{1-2\nu}{2}
\end{bmatrix}$$

Here the true parameters 

$$E = 200\;\text{GPa} \quad \nu = 0.35$$

They are the parameters to be calibrated in the inverse modeling. The observation is given by the displacement vectors of 20 random points on the plate. 

We consider a uniform prior for the random walk MCMC simuation, so the log likelihood up to a constant is  given by 

$$l(y') = -\frac{(y-y')^2}{2\sigma_0^2}$$

where $y'$ is the current proposal, $y$ is the measurement, and $\sigma_0$ is the standard deviation. We simulate 100000 times, and the first 20% samples are used as "burn-in" and thus discarded.

For the linearized Gaussian model, we use $Q=I$ and $R=\sigma_0^2I$ to account for a unit Gaussian prior and measurement error, respectively. 

The following plots show the results



| $\sigma_0=0.01$                                              | $\sigma_0=0.05$                                              | $\sigma_0=0.1$                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/sigma0.01.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/sigma0.05.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/sigma0.1.png?raw=true) |
| $\sigma_0=0.2$                                               | $\sigma_0=0.5$                                               |                                                              |
| ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/sigma0.2.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/sigma0.5.png?raw=true) |                                                              |


We see that when $\sigma_0$ is small, the approximation is quite consistent with MCMC results. When $\sigma_0$ is large, due to the assumption that the uncertainty is Gaussian, the linearized Gaussian model does not fit well with the uncertainty shape obtained with MCMC; however, the result is still consistent since the linearized Gaussian model yields a larger standard deviation. 

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
using AdFem


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
using AdFem


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
\nabla\cdot (K(u) \nabla u(x, y)) &= 1 & \text{ in } \Omega\\ 
u(x,y) &= 0  & \text{ on } \partial \Omega
\end{aligned}$$

We tested two cases: in the first case, we use the synthetic observation $u_{\text{obs}}\in\mathbb{R}$ without adding any noise, while in the second case, we add 1% Gaussian noise to the observation data

$$u'_{\text{obs}} = u_{\text{obs}} (1+0.01 z)\quad z\sim \mathcal{N}(0, I_n)$$

The prior for $K(u)$ is $\mathcal{N}(0, 10^{-2})$, where one standard deviation is around 10%~20% of the actual $K(u)$ value.  The measurement prior is given by 

$$\mathcal{N}(0, \sigma_{\text{model}}^2 + \sigma_{\text{noise}}^2)$$

The total error is modeled by $\sigma_{\text{model}}^2 + \sigma_{\text{noise}}^2\approx 10^{-4}$.



| Description                 | Uncertainty Bound (two standard deviation) | Standard Deviation at Grid Points |
| --------------------------- | ---- | ---- |
| $\sigma_{\text{noise}}=0$   | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/nn2-uq0.0-1.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/nn2-uq0.0-2.png?raw=true) |
| $\sigma_{\text{noise}}=0.01$ | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/nn2-uq0.01-1.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/nn2-uq0.01-2.png?raw=true) |

We see that in general when $u$ is larger, the uncertainty bound is larger. For small $u$, we can estimate the map $K(u)$ quite accurately using a neural network. 