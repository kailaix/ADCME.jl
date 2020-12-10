# Training Deep Neural Networks with Trust-Region Methods

Trust-region methods are a class of global optimization methods. The basic idea is to successively solve an approximated optimization problem in a small neighborhood of the current state. For example, we can approximate the local landscape of the objective function using a quadratic function, and thus can solve efficiently and accurately. The biggest advantage of trust-region methods in the context of deep neural network is that they can escape saddle points, which are demonstrated to be the dominant causes for slow convergence, with proper algorithm design. 

However, the most challenging problem with trust-region methods is that we need to calculate the Hessian (curvature information) to leverage the local curvature information. Computing the Hessian can be quite expensive and challenging, especially if the forward computation involves complex procedures and has a large number of optimizable variables. Fortunately, for many deep neural network based inverse problems, the DNNs do not need to be huge for good accuracy. Therefore, calculating the Hessian is plausible. This does not mean that efficient computation is easy, and we introduce the technique is another post. In this post, we compare the trust-region method with other competing methods (L-BFGS-B, BFGS, and ADAM optimizer) for training deep neural networks that are coupled with a numerical solver. We also shed lights on why the other optimizers slow down. 


## Trust-region Methods

We consider an unconstrained optimization problem 

$$\min_x f(x) \tag{1}$$

The trust-region method solves the optimization problem Eq. 1 by iteratively solving many simpler subproblems, which are good approximation to $f(x_k)$ at the neighborhood of $x_k$. We model $f(x_k+s)$ using a quadratic model 

$$m(s) = f_k + s^T g_k + \frac{1}{2}s^T H_k s \tag{2}$$

Here $f_k = f(x_k)$, $g_k = \nabla f(x_k)$, $H_k = \nabla^2 f(x_k)$. 

Eq. 2 is essentially the Taylor expansion of $f(x)$ at $x_k$. This approximation is only accurate within the neighborhood of $x_k$. Therefore, we constrain our subproblem to a **trust region** 

$$||s||\leq \Delta_k$$

The subproblem has the following form 

$$\begin{aligned}\min_{s} & \; m(s) \\ \text{s.t.} & \; \|s\|\leq \Delta_k\end{aligned} \tag{3}$$

In this work, we use the method proposed in [^trust-region] to solve Eq. 3 nearly exactly. 

[^trust-region]: Cline, A. K., Moler, C. B., Stewart, G. W., Wilkinson, J. H. An estimate for the condition number of a matrix. 1979. SIAM Journal on Numerical Analysis, 16(2), 368-375.

## Model Problem: Static Poisson's Equation

In this example, we consider the Poisson's equation 

$$\nabla \cdot (\kappa_\theta(x) \nabla u)) = f(x), \;x\in \Omega, \; x\in \partial\Omega$$

Here $\kappa_\theta(x)$ is a deep neural network and $\theta$ is the weights and biases. We discretize $\Omega$ using a uniform grid. Assume we can observe the full field data $u_{obs}$ on the grid points. We can then train the deep neural network using the residual minimization method [^residual-minimization]

$$\min_\theta \sum_{i,j} (F_{i,j}(\theta) - f_{i,j})^2 \tag{4}$$

Here $F_{i,j}(\theta)$ is the finite difference discretization of $\nabla \cdot (\kappa_\theta(x) \nabla u_{obs}))$ at the grid points. In our benchmark, we add 10% uniform random noise to $f$ and $u_{obs}$ to make the problem more challenging.  

[^residual-minimization]: Huang, Daniel Z., et al. "Learning constitutive relations from indirect observations using deep neural networks." Journal of Computational Physics (2020): 109491.


We apply 4 optimizers to solve Eq. 4. Because the optimization results depend on the initialization of the deep neural network, we use 5 different initial guess for DNNs. The result is shown below

| Case        | Convergence Plots | 
|-------------|---|
|1 | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/losses2_static.png?raw=true)|
|2 | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/losses23_static.png?raw=true)|
|3 | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/losses233_static.png?raw=true)|
|4 | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/losses2333_static.png?raw=true)|
|5 | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/losses23333_static.png?raw=true)|

We can see for all cases, the trust-region method provides a much more accurate result, and in general converges faster. The ADAM optimizer is the least competent, partially because it's a first-order optimizer and is not able to fully leverage the curvature information. The BFGS optimizers constructs an approximate Hessian that is SPD. The L-BFGS-B optimizer is an approximation to BFGS, where it uses only a limited number of previous iterates to construct the Hessian matrix. As mentioned, in the optimization problem involving deep neural networks, the slow down is mainly due to the saddle point, where the descent direction corresponds to the negative eigenvalues of the Hessian matrix. Because BFGS and L-BFGS-B ensure that the Hessian matrix is SPD, they cannot provide approximate guidance to escape the saddle point. This hypothesis is demonstrated in the following plot, where we show the distribution of Hessian eigenvalues at the last step for Case 2

| Optimizer        | L-BFGS-B | BFGS | Trust Region |
|-------------|---|---|---|
| Eigenvalue Distribution | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_lbfgs_eig.png?raw=true)|![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_bfgs_eig.png?raw=true)|![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_tr_eig.png?raw=true)|

We also show the number of **negative** eigenvalues for the BFGS optimizer

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_positive_eigvals.png?raw=true)

We can see that the number of negative eigenvalues stays at around 150 after 1300 iterations. That's also where the BFGS optimizer starts to stagnate. This means that the Hessian matrix of the optimization problem is **never SPD** during the iteration of the BFGS optimizer. The non-SPD property is actually inconsistent with the local convexity assumption at stationary points  in the convergence theory of the BFGS optimizer!

We also analyze the direction of the search direction $p_k$ in the BFGS optimizer. We consider two values

$$\begin{aligned}\cos(\theta_1) &= \frac{-p_k^T g_k}{\|p_k\|\|g_k\|} \\ \cos(\theta_2) &= \frac{p_k^T q_k}{\|p_k\|\|q_k\|}\end{aligned}$$

Here $q_k$ is the direction for the **Cauchy point**, which is the descent direction in the trust-region method

$$q_k = -H_k^{-1}g_k$$

The two quantities are shown in the following plots (since the trust-region method converges in around 270 iterations, $\cos(\theta2)$ only has limited data points)

| $\cos(\theta_1)$        | $\cos(\theta_2)$ |
|-------------|---|
| ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_angles.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_angles2.png?raw=true)|

There are two conclusions to draw from the plots

1. The search direction of the BFGS optimizer deviates from the gradient descent method. 
2. The search direction of the BFGS optimizer is not very correlated with the Cauchy point direction; this indicates the search direction poorly recognizes the negative curvature directions. 

## Another Example: Heat Equation

We consider a time-dependent PDE problem: the heat equation

$$\frac{\partial u}{\partial t} = \nabla \cdot (\kappa_\theta(x) \nabla u)) + f(x), \;x\in \Omega, \; x\in \partial\Omega$$

We assume that we can observe the full field data of $u$ as snapshots. We again apply the residual minimization method to train the deep neural network. The following shows the convergence plots for different initial guesses of the DNNs. 


| Case        | Convergence Plots | 
|-------------|---|
|1 | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/losses2_dynamic.png?raw=true)|
|2 | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/losses23_dynamic.png?raw=true)|
|3 | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/losses233_dynamic.png?raw=true)|
|4 | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/losses2333_dynamic.png?raw=true)|
|5 | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/losses23333_dynamic.png?raw=true)|
We see that the trust-region is more competitive than the other methods. 


## Limitations 

Despite many promising features of the trust region method, it is not without limitations, which we want to discuss here. The current trust-region method requires calculating the Hessian matrix. Firstly, computing the Hessian matrix can be technically difficult, especially when DNNs are coupled with a sophisticated numerical PDE solver. There are many existing techniques for computing the Hessian. The TensorFlow backend supports Hessian computation concurrently, but it requires users to implement rules for calculating "gradients of gradients". Additionally, TensorFlow uses forward propagation to evaluate the Hessian. This means that TensorFlow loops over each gradient component and calculating a row of Hessian at a time. This does not leverage the symmetry of Hessians and can be quite inefficient if the number of unknowns is large. Another approach, edge pushing algorithms, uses one backward pass to evaluate the Hessian. This approach takes advantage of the symmetry of Hessians. However, the implementation can be quite convolved and computations can be expensive in some scenarios. We will discuss0 in more details in another post. 


## Conclusion

Trust-region methods are a class of global optimization techniques. They are less popular in the deep learning approach because the DNNs tend to be huge and the computation of Hessians is expensive. However, they are very suitable for many computational engineering problems, where DNNs are typically small, and convergence as well as accuracy is a critical concern. Our point of view is that although the Hessian computations are expensive, they are quite rewarding. Future researches will focus on efficient computation and automation of Hessian computations. 