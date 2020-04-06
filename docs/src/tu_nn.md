#  Combining Neural Networks with Numerical Schemes 

Modeling unknown componenets in a physical models using a neural networks is an important method for function inverse problem. This approach includes a wide variety of applications, including

* Koopman operator in dynamical systems
* Constitutive relations in solid mechanics.
* Turbulent closure relations in fluid mechanics.
* ...... 

To use the known physics to the largest extent, we couple the neural networks and numerical schemes (e.g., finite difference, finite element, or finite volumn method) for partial differential equations. Based on the nature of the observation data, we present three methods to train the neural networks using gradient-based optimization method: the residual minimization method, the penalty method, and the physics constrained learning. We discuss the pros and cons for each method and show how the gradients can be computed using automatic differentiation in ADCME. 

## Introduction

Many science and engineering problems use models to describe the physical processes. These physical processes are usually derived from first principles or based on empirical relations. Neural networks have shown to be effective to model complex and high dimensional functions, and can be used to model the unknown mapping within a physical model. The known part of the model, such as conservation laws and boundary conditions, are preserved to obey the physics. Basically, we substutite unknown part of the model using neural networks in a physical system. As an example, consider a simple 1D heat equation

$$\begin{aligned}
\frac{\partial}{\partial x}\left(\kappa(u)\frac{\partial u}{\partial x}\right) &= f(x)& x\in \Omega\\
u(0) = u(1) &= 0
\end{aligned}$$

The diffusivity coefficient $\kappa(u)$ is a function of the temperature $u$, and is an unknown function to be calibrated. We consider a data-driven approach to discover $\kappa(u)$  using only the temperature data set $\mathcal{T}$. In the case where we do not know the constitutive relation, it can be modeled using a neural network  $\kappa_\theta (u)$, where $\theta$ is the weights and biases of the neural network. 

In the following, we will discuss three methods based on the nature of the observation data set $\mathcal{T}$ to train the neural network. 

## Residual Minimization for Full Field Data

In the case $\mathcal{T}$ consist of full-field data, i.e., the values of $u(x)$ on a very fine grid, we can use the **residual minimization** to learn the neural network. Specifically, we can discretize the PDE using a numerical scheme, such as finite element method (FEM), and obtain the residual term 

$$R_j(\theta) =\sum_i u(x_i)\int_0^1  \kappa_\theta(\sum_i c_i \phi_i) \phi_i'(x) \phi_j'(x) dx - \int_0^1 f(x)\phi_j(x)dx$$

where $\phi_i$ are the basis functions in FEM. To find the optimal values for $\theta$, we can perform a residual minimization 

$$\min_\theta \sum_j R_j(\theta)^2$$

The residual minimization method avoids solving the PDE system, which can be expensive. The implication is straightforward using ADCME: all we need is to evaluate $R_j(\theta)$ and $\frac{\partial R_j(\theta)}{\partial \theta}$ (using automatic differention). However, the limitation is that this method is only applicable when full-field data are available. 

In the following two references, we explore the applications of the residual minimization method to constitutive modeling

DZ Huang, K Xu, C Farhat, E Darve. [Learning Constitutive Relations from Indirect Observations Using Deep Neural Networks](https://arxiv.org/pdf/1905.12530.pdf)

K Xu, DZ Huang, E Darve. [Learning Constitutive Relations using Symmetric Positive Definite Neural Networks](https://arxiv.org/pdf/2004.00265.pdf)

## Penalty Method for Sparse Observations

In the case where $\mathcal{T}$ only consists of sparse observations, i.e., $u_o(x_i)$ (observation of $u(x_i)$) at only a few locations $\{x_i\}$, we can formulate the problem as a PDE-constrained optimization problem

$$\begin{aligned}
\min_\theta& \;\sum_i (u(x_i) - u_o(x_i))^2\\
\text{s.t.} &\; \frac{\partial}{\partial x}\left(\kappa_\theta(u)\frac{\partial u}{\partial x}\right) = f(x)& x\in \Omega\\
& u(0) = u(1) = 0
\end{aligned}\tag{1}$$

As pointed in [this article](https://kailaix.github.io/ADCME.jl/dev/tu_optimization/), we can apply the **penalty method** to solve the contrained optimization problem, 

$$\min_{\theta,u} \;\sum_i (u(x_i) - u_o(x_i))^2 + \rho \|F(u, \theta)\|^2_2$$

Here $\rho$ is the penalty parameter and $F(u, \theta)$ is the discretized form of the PDE. For example, $F(u,\theta)_i$ can be $R_i(\theta)$ in the residual minimization problem. 

The penalty method is conceptually simple and easy to implement. Like the residual minimization method, the penalty method requires limited insights into the numerical simulator to evaluate the gradients with respect to $u$ and $\theta$. The method does not require solving the PDE.

However, the penalty method treats $u$ as optimization variable and therefore typically has much more degrees of freedom than the original constrained optimization problem. Mathematically, the penalty method suffers from worse conditioning than the constrained one, making it unfavorable in many scenarios. 

## Physics Constrained Learning

An alternative approach to the penalty method in the context of sparse observations is the **physics constrained learning** (PCL). The physics constrained learning reduces Equation 1 to an unconstrained optimization problem by two steps:

1. Solve for $u$ from the PDE constraint, given a candidate neural network;
2. Plug the solution $u(\theta)$ into the objective function and obtain a reduced loss function

The reduced loss function only depends on $\theta$ and therefore we can perform the unconstrained optimization, in which case a wide variety of off-the-shelf optimizers are available. The advantage of this approach is that PCL enforces the physics constraints, which can be crucial for some applications and are essential for numerical solvers. Additionally, PCL typically exhibits high efficiency and fast convergence compared to the other two methods. However, PCL requires deep insights into the numerical simulator since an analytical form of $u(\theta)$ is not always tractable. It requires automatic differentiation through implict numerical solvers as well as iterative algorithms, which are usually not supported in AD frameworks. In the references below, we provide the key techniques for solving these problems. A final limitation of PCL is that it consumes much more memory and runtime per iteration than the other two methods. 

For more details on PCL, and comparison between the penalty method and PCL, see the following reference

K Xu, E Darve. [Physics Constrained Learning for Data-driven Inverse Modeling from Sparse Observations](https://arxiv.org/pdf/2002.10521.pdf)

## Summary

Using physics based machine learning  to solve inverse problems requires techniques from different areas, deep learning, automatic differentiation, numerical PDEs, and physical modeling. By a combination of the best from all worlds, we can leverage the power of modern high performance computing environments to solve long standing inverse problems in physical modeling. Specially, in this article we review three methods for training a neural network in a physical model using automatic differentiation. These methods can be readily implemented in ADCME. We conclude this article by a direct comparison of the three methods

| Method                         | Residual Minimization | Penalty Method | Physics Constrained Learning |
| ------------------------------ | --------------------- | -------------- | ---------------------------- |
| Sparse Observations            | ❌                     | ✔️              | ✔️                            |
| Easy-to-implement              | ✔️                     | ✔️              | ❌                            |
| Enforcing Physical Constraints | ❌                     | ❌              | ✔️                            |
| Fast Convergence               | ❌                     | ❌              | ✔️                            |
| Minimal Optimization Variables | ✔️                     | ❌              | ✔️                            |



