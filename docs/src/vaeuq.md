# Uncertainty Quantification for Bayesian Inverse Problems using Variational Autoencoder

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

We address this issue by modeling the posterior $p(\theta|y)$ using a variational autoencoder (VAE). We have already introduced VAE in [this article](./vae.md). In the following, we use the same notation as that article. In that example, we construct a latent space which is essentially Gaussian. A direct generalization of VAE to this physical problem is that we treat $\theta$ as latent variables. However, this imposes the Gaussian-like distribution constraint on $\theta$ conditioned on $y$. This is not desirable because the uncertainty of physical parameters in many problems is non-Gaussian. 

To model the non-Gaussian uncertainty, we use a generative deep neural network $g_\eta$ that maps latent variables $z$ to $\theta$, where $\eta$ is the weights and biases of the neural network
$$\theta = g_\eta(z)$$

The parameter $\theta$ is mapped to the observation $y$ via solving a PDE system and projecting the state variables onto observable degrees of freedom. Therefore, we have 

$$p_{\eta}(y|z) = \frac{1}{(\sqrt{2\pi \sigma^2})^q}\exp\left[ -\frac{1}{2\sigma^2} \| y-y(g_\eta(z))\|^2 \right]$$

Therefore, the discrete marginal likelihood function is given by 

$$\mathbb{E}_{p_w(z|x)}[\log p_\eta(y|z)] \approx \frac{1}{n}\sum_{i=1}^n \left[-\frac{1}{2\sigma^2}\|y-y(g_\eta(z))\|^2 - \frac{q}{2}\log(2\pi \sigma^2) \right]$$

The KL divergence term in VAE has the same form as the VAE article

$$\mathrm{KL}(p_w(z|y) || p(z)) = -d - d\log(\sigma_y) +\frac{1}{2} \|\mu_y\|^2 + \frac{d}{2}$$

Compared to the typical use case of VAE, for uncertainty quantification of scientific problems, there are some unique characteristics: 

1. Small data problem. Typically we have only a few or even a single measurement $y$. Therefore, in each iteration of an optimization procedure, all the measurement constitutes the whole batch. 
2. Automatic differentiation through numerical solvers. Instead of differentiating through neural networks for calculating gradients, we need to differentiate through numerical solvers. This may be nontrivial in certain scenarios. For example, we may need to differentiate through implicit solvers, where we can apply the physics constrained learning technique. 