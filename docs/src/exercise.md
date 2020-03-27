# Exercise: Estimating Thermal Thermal Diffusivity Distribution from Sparse Sensor Measurements

We saw in the lectures how to implement numerical PDE schemes in ADCME and use the forward computation codes to do inverse modeling with automatic differentiation. Here we apply the concepts of inverse modeling to a learning task---estimate the thermal diffusivity distribution in a material from sparse sensor measurements. The **thermal diffusivity** is the measure of the ease at which the heat can pass through a material. It relies on the material property. Let $T$ be the temperature, and $\kappa$ be the thermal diffusivity, the Fourier's law of heat transfer says

$$\frac{\partial u(\mathbf{x}, t)}{\partial t} = \kappa\Delta u(\mathbf{x}, t) + f(\mathbf{x}, t), \quad t\in [0,T], x\in \Omega \tag{1}$$

Here $f$ is the heat source and $\Omega$ is the domain.

To make use of the heat equation, we need addition information. 

- **Initial Condition**: the initial temperature distribution is given $u(\mathbf{x}, 0) = u_0(\mathbf{x})$. 

- **Boundary Conditions**: the temperature of the material is affected by what happens on the boundary. There are several possible boundary conditions. In this exercise we consider two of them:

   (1) Temperature fixed at a boundary,

   $$u(\mathbf{x}, t) = 0, \mathbf{x}\in \Gamma_u \tag{2}$$

  (2) Insulated boundary. The heat flow can be prescribed (known as the _no flow_ boundary condition)

  $$-\kappa\frac{\partial u(\mathbf{x},t)}{\partial n} = 0, \mathbf{x}\in \Gamma_N \tag{3}$$

  Here $n$ is the outward normal vector. 

  The boundaries $\Gamma_u$ and $\Gamma_N$ satisfies $\partial \Omega = \Gamma_u \cup \Gamma_N$

Assume that we want to experiment with a piece of new material. The material has heterogenous properties in that the thermal diffusivity is a function of the space, i.e., $\kappa(\mathbf{x})$. This is the quantity we want to estimate. To this end, we place some sensors in the domain or on the boundary. The measurements are sparse in the sense that only the temperature from those sensors can be collected. Namely, let the sensors be located at $\{\mathbf{x}_i\}_{i=1}^M$, then we can observe $\{\hat u(\mathbf{x}_i, t)\}_{i=1}^M$, i.e., the measurements of $\{ u(\mathbf{x}_i, t)\}_{i=1}^M$. We also assume that the boundary conditions, initial conditions and the source terms are known. 

## Problem 1: 1D Case

We first consider the simpler 1D case. In this problem the material is a rod $\Omega=[0,1]$. We consider a homogeneous (zero) fixed boundary condition on the left side, and an insulated boundary on the right side. 

(a) Write down the mathematical optimization problem for the inverse modeling.

Now we consider the discretization of the forward problem. We divide the domain $[0,1]$ into $n$ equi-spaced intervals. We consider the time horizon $T = 10$, and divide the time horizon $[0,T]$ into $m$ equi-spaced intervals. We use a finite difference scheme to solve the 1D heat equation Equations 1-3. Specifically, we use an implicit scheme for stability

$$\frac{u^{k+1}_i-u^k_i}{\Delta t} = \kappa_i \frac{u^{k+1}_{i+1}+u^{k+1}_{i-1}-2u^{k+1}_i}{\Delta x^2} + f_i^{k+1}, \quad k=1,2,\ldots, i=1,2,\ldots, n$$

where $\Delta t$ is the time interval, $\Delta x$ is the space interval, $u_i^k$ is the numerical approximation to $u((i-1)\Delta x, (k-1)\Delta t)$, $\kappa_i$ is the numerical approximation to $\kappa((i-1)\Delta x)$, and $f_i^{k} = f((i-1)\Delta x, (k-1)\Delta t)$.

For the insulated boundary, we introduce the ghost node $u_{0}^k$, which satisfies

$$-\kappa_1 \frac{u_2^{k}-u_0^k}{2\Delta x} = 0\tag{4}$$

(b) Let $U^k = \begin{bmatrix}u_1^k\\u_2^k\\\vdots \\u_n^k\end{bmatrix}$ (note the index starts from 1 and ends with $n$), using the finite difference scheme, together with proper elimination of boundary values $u_0^k$, $u_{n+1}^k$, we have the following  formula

$$AU^{k+1} = U^k + F^k$$

Express the matrix $A\in \mathbb{R}^{n\times n}$ in terms of $\Delta t$, $\Delta x$ and $\{\kappa_i\}_{i=1}^{n}$; namely, what is $A_{ij}$? In addition, what is $F^k\in \mathbb{R}^n$?

(c) Now precompute in Julia the force vector $F^k$ and pack it into a matrix $F\in \mathbb{R}^{(m+1)\times n}$. Using [spdiag]([https://kailaix.github.io/ADCME.jl/dev/api/#ADCME.spdiag-Tuple{Integer,Vararg{Pair,N}%20where%20N}](https://kailaix.github.io/ADCME.jl/dev/api/#ADCME.spdiag-Tuple{Integer,Vararg{Pair,N} where N})) to construct `A` as a `SparseTensor`. Use $m=20$ and $n=10$. Use $f(x, t) = \exp(-10(x-0.5)^2)$ for all the following questions of Problem 1. $\kappa_i$ is given as a vector 

```julia
κ = constant(1 .+ exp.(Array(LinRange(0,1,n))))
```

(d) The computational graph of the dynamical system can be efficiently constructed using `while_loop`. Conduct the forward computation using `while_loop`. For debugging, you can plot the temperature at the left side. You should have something similar to XXX. 

Hint: You might want to read the documentation for [while_loop]([https://kailaix.github.io/ADCME.jl/dev/api/#ADCME.while_loop-Tuple{Union{Function,%20PyObject},Function,Union{PyObject,%20Array{Any,N}%20where%20N,%20Array{PyObject,N}%20where%20N}}](https://kailaix.github.io/ADCME.jl/dev/api/#ADCME.while_loop-Tuple{Union{Function, PyObject},Function,Union{PyObject, Array{Any,N} where N, Array{PyObject,N} where N}})) for its usage.

(e) Now we are ready to perform inverse modeling. Now using the initial guess

```julia
κ = Variable(ones(n))
```

Perform the mathematical optimization using `BFGS!`. Plot the $\kappa$ values after the optimization converges. 

Hint: To debug for your inverse modeling code, refer to [Inverse Modeling Recipe](https://kailaix.github.io/ADCME.jl/dev/tu_recipe/)



## Problem 2: 2D Case

Now we consider the 2D case. We assume that $\Omega=[0,1]^2$, and $\Gamma_u=\partial\Omega$. In this case, we assume that the sensors are located at the following points

XXX

We still use the finite difference method to discretize the PDE. Additionally, a similar implicit scheme is used for stability

XXX

Let $U^k$ be the vector of vectorized $\{u_{ij}\}$, and the order is $u_{22}, u_{23}, \ldots, u_{2n}, u_{31}, \ldots, u_{nn}$. Convince yourself that the evolution formula also has the form as Equation 4 (no need to provide justification). We provide a custom operator here XXX, which implements a differentiable SparseTensor $A$ for you (refer to the instructions on how to compile and use it). 

(a) Similar to Problem 1, conduct forward computation using `while_loop`. Plot  the curve of the temperature at $(0.5,0.5)$. For debugging, you should obtain something as follows



The parameters used in this problem: $m=10$, $n=10$, $T=1$, $K=50$, $f(\mathbf{x},t) = e^{-t}\exp(-10\|\mathbf{x}-[0.5;0.5]\|^2_2)$, $\kappa(\mathbf{x}) = 1 + \|\mathbf{x}\|^2_2$

(b) Now assume $\kappa(\mathbf{x})$ is unknown but we have observations from the sensors (the observations can be computed using the forward simulation code you just developed). Conduct mathematical optimization using `BFGS!`. The initial guess for $\kappa$ is $\kappa(\mathbf{x})=1$. 

(c) (Bonus) Can you implement a custom operator kernel for the 1D case? 

Warning: This problem may be time consuming, but this technique is frequently used for developing high performance codes for inverse modeling. 