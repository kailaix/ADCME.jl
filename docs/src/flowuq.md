# Uncertainty Quantification using Normalizing Flows

The forward problem represents a mapping from model parameters to observations. The mapping might result from a discretized system of partial differentiatial equations. The model is written as 

$$F(u, \theta) = 0$$ 

where $u\in \mathbb{R}^N$ is the discrete state vector of $N$ unknowns, and $\theta\in \mathbb{R}^p$  is the physical parameter from a domain $D\subset \mathbb{R}^p$. 

The observation is typically a linear combination of state vectors

$$y = Cu$$

where $y\in\mathbb{R}^q$ is a vector of $q$ output and $C\in \mathbb{R}^{q\times N}$ is a constant matrix.

The **forward problem** can be stated as: given a model parameter $\theta$, calculate $y$. In a data-driven model, we have measurement of the output, $\tilde y$, and we want to estimate the model parameter $\theta$ from this measurement. This procedure is called the **inverse problem**.

The deterministic inverse problem seeks the optimal model parameter $\theta$, which solves 
$$\begin{aligned}
\min_{\theta, \theta}&\; \|y - \tilde y\|_2^2\\
\text{s.t.}&\; F(u, \theta) = 0\\ 
&\; y = Cu
\end{aligned}\tag{1}$$
However, solving the deterministic inverse problem only gives us an estimate of the model parameter $\theta$ without any information on the uncertainty of our estimation. To quantify the uncertainty, we can recast the inverse problem in the Bayesian framework, where we seek the **posterior probability**, $p(\theta|\tilde y)$, of the model parameter.

We consider the case where we have some measurement errors in the outputs
$$y = Cu + \epsilon$$
where $\epsilon$ is a zero-mean Gaussian random variable, and its errors in each component are uncorrelated and have the same standard deviation $\sigma$. In the Bayesian framework, we also assign a **prior distribution** $p(\theta)$ to the model parameter. 

Before we describe our method, let us see how this stochastic inverse problem is solved using the standard approach. In the MCMC approach, we apply the Bayes formula and obtain 

$$p(\theta|\tilde y) = \frac{p(\tilde y | \theta) p(\theta)}{p(\tilde y)}$$

We assume a non-informative prior, i.e., $p(\theta)\propto 1$, and therefore

$$p(\theta|\tilde y) \propto \exp\left[ -\frac{1}{2\sigma^2} (\tilde y-y(\theta))^T(\tilde y-y(\theta)) \right]\tag{2}$$

where $y(\theta)$ is the solution to the system Equation (1) for a given parameter $\theta$. The Metropolis-Hastings MCMC algorithm is then used to numerically sample from the posterior distribution in Equation (2). However, the MCMC approach suffers from two critical issues that make them less challenging in practice:

1. The forward computation of $y(\theta)$ at every sampling step can be very expensive for large-scale scientific computing problems. 
2. The MCMC approach is only able to synthesize samples from a data distribution [^1], making the approach incapable of statistical inference tasks that requires explicit likelihood functions. 

[^1]: Methods such as kernel density estimation (KDE) can reconstruct density functions out of samples. However, those methods face challenges in high dimensions or require voluminous samples, which can be expensive to generate. 


We address this issue by modeling the posterior distribution $p(\tilde y|\theta)$ directly using normalizing flows. This approach models the likelihood function directly and enables cheap sampling. 

First, instead of modeling $\theta$ directly, we consider transforming latent variables $z$ to $\theta$ using deep latent Gaussian models, where latent variables $z_1$, $z_2$, $\ldots$, $z_L$, $z_{L+1} = z$, and $\theta$ have a joint distribution

$$p(\theta, z_1, z_2, \ldots, z_L| z_{L+1})  = p(\theta|f_0(z_1))\Pi_{l=1}^L p(z_l | f_l(z_{l+1}))$$

Here, each latent variable $z_l$ (including $z_{L+1}=z$) has a unit Gaussian prior $p(z_l) = \mathcal{N}(0, I)$, and functions $f_i$ are all parametrized by a deep neural network. We denote the set of weights and biases of all neural networks by $\psi$, and the statistical model for $\theta$ is written as $p_\psi(\theta|z)$. Therefore, we have

$$\begin{aligned}
p_\psi( y|z) &= p( y|\theta, z)p_\psi(\theta|z)\\ 
&= p( y|\theta)p_\psi(\theta|z)\\ 
&\propto  \exp\left[ -\frac{1}{2\sigma^2} ( y-y(\theta))^T( y-y(\theta)) \right] p_\psi(\theta|z)
\end{aligned}$$

Next, we approximate the posterior distribution by $p_\phi(z|\tilde y) \approx p(z|\tilde y)$, where $p_\phi$ is a normalizing flow-based generative model, whose explaination is delayed to a later text. The negative standard evidence lower bound (ELBO) is derived as 
$$L(y) = KL(p_\phi(z|y)|| p(z)) - \mathbb{E}_{p_\psi(y|z)} \log p(y|z) \geq -\log p_{\phi, \psi}(y)$$
To find a good local minima for $\phi$ and $\psi$, we minimize the ELBO using stochastic gradient descent method. The biggest challengee is how to compute $\nabla_\psi \mathbb{E}_{p_\psi(y|z)} \log p(y|z)$. We apply the stochastic backpropagation method, which relies on a Monte Carlo approximation of the gradient using 
$$XXX$$ 

Once $\phi$ and $\psi$ are estimated, we can reconstruct the posterior distribution of $p(\theta|y)$ using 

$$p(\theta|y) = \int p(\theta|z, y)p(z) dz \approx \int p_{\phi}(\theta|z) p(z) dz$$

Because we can sample from the posterior distribution $p_\psi(\theta|y)$ cheaply, we can generate samples from the data distribution relatively faster than MCMC. 


