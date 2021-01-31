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

[^trust-region]: A.R. Conn, N.I. Gould, and P.L. Toint, "Trust region methods", Siam, pp. 169-200, 2000.

## Example: Static Poisson's Equation

In this example, we consider the Poisson's equation 

$$\nabla \cdot (\kappa_\theta(u) \nabla u)) = f(x), \;x\in \Omega, \; x\in \partial\Omega$$

Here $\kappa_\theta(u)$ is a deep neural network and $\theta$ is the weights and biases. We discretize $\Omega$ using a uniform grid. Assume we can observe the full field data $u_{obs}$ on the grid points. We can then train the deep neural network using the residual minimization method [^residual-minimization]

$$\min_\theta \sum_{i,j} (F_{i,j}(\theta) - f_{i,j})^2 \tag{4}$$

Here $F_{i,j}(\theta)$ is the finite difference discretization of $\nabla \cdot (\kappa_\theta(u_{obs}) \nabla u_{obs}))$ at the grid points. In our benchmark, we add 10% uniform random noise to $f$ and $u_{obs}$ to make the problem more challenging.  

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


In the following, we show the eigenvalue distribution of the Hessian matrix for 

$$l(\theta) = \sum_{i=1}^n (\kappa_\theta(u_i) - \kappa_i)^2$$

We can see that the Hessian possesses some negative eigenvalues. This implies that the DNN and DNN-FEM loss functions indeed have different curvature structures at the local minimum. The structure is altered by the PDE constraint. 


| Optimizer        | L-BFGS-B | BFGS | Trust Region |
|-------------|---|---|---|
| Eigenvalue Distribution | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/nn_static_lbfgs_eig.png?raw=true)|![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/nn_static_bfgs_eig.png?raw=true)|![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/nn_static_tr_eig.png?raw=true)|


We also show the number of **negative** eigenvalues for the BFGS and trust region optimizer. Here we use a threshold $\epsilon=10^{-6}$: for a given eigenvalue $\lambda$, it is treated as "positive" if $\lambda>\epsilon \lambda_{\max}$, and "negative" if $\lambda < - \epsilon \lambda_{\max}$, otherwise zero. Here $\lambda_{\max}$ is the maximum eigenvalue. 

| BFGS | Trust Region |
|-------------|---|
|![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_positive_eigvals.png?raw=true)|![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_positive_eigvals_tr.png?raw=true)|



We can see that the number of positive eigenvalues stays at around 18 and 30 for BFGS and trust region methods after a sufficient number of iterations. The number of negative eigenvalues is nearly zero. This means that both optimizers converge to points with positive semidefinite Hessian matrices. Stationary points are true local minima, instead of saddle points. 

We also analyze the direction of the search direction $p_k$ in the BFGS optimizer. We consider two values

$$\begin{aligned}\cos(\theta_1) &= \frac{-p_k^T g_k}{\|p_k\|\|g_k\|} \\ \cos(\theta_2) &= \frac{p_k^T q_k}{\|p_k\|\|q_k\|}\end{aligned}$$

Here $q_k$ is the direction for the **Newton's point**

$$q_k = -H_k^{-1}g_k$$

The two quantities are shown in the following plots (since the trust-region method converges in around 270 iterations, $\cos(\theta2)$ only has limited data points)

| $\cos(\theta_1)$        | $\cos(\theta_2)$ |
|-------------|---|
| ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_angles.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_angles2.png?raw=true)|

There are two conclusions to draw from the plots

1. The search direction of the BFGS optimizer deviates from the gradient descent method. 
2. The search direction of the BFGS optimizer is not very correlated with the Newton's point direction; this indicates the search direction poorly recognizes the negative curvature directions. 

## Example: Heat Equation

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

We also show the eigenvalue distribution of the Hessian matrices for Case 3. 

| ADAM        | BFGS | LBFGS| Trust Region |
|---|---|---|---| 
|![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/dynamic_ADAM_eig.png?raw=true)| ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/dynamic_BFGS_eig.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/dynamic_LBFGS_eig.png?raw=true)| ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/dynamic_TrustRegion_eig.png?raw=true)|
|50|31|22|35|

The eigenvalues of Hessian matrices are nonnegative except for the ADAM case, where the optimizer does not converge to a satisfactory local minimum after 5000 iterations. Hence, in what follows, we omit the discussion of ADAM optimizers. 

The third row show the number of positive eigenvalues using the criterion mentioned before. We again see that among all three methods---BFGS, LBFGS, and trust region---the smaller loss function at the last step is associated with a larger number of positive eigenvalues. 

We can interpret the eigenvalues associated with zero eigenvalues as "inactive directions", in the sense that given the gradient norm is small, perturbation in the direction of zero eigenvalues almost does not change the loss function values. In other words, the local minimum found by trust region methods has more active directions than BFGS and LBFGS. The active directions can also be viewed as "effective degrees of freedoms (DOFs)", and thus we conclude trust region methods find a local minimum with smaller loss function due to more effective DOFs. 

The readers may wonder why different local minimums have different effective DOFs. To answer this question, we show the cumulative distribution of the maginitude of weights and biases in the following plot

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/weight_dist.png?raw=true)

The plot shows that BFGS and LBFGS outweight trust region methods in terms of large weights and biases (in terms of maginitudes). Because we use $\tanh$ as activation values, for fixed intermediate activation values, large weights and biases are more likely to cause saturation of activation values, i.e., the inputs to $\tanh$ is large or small and thus the outputs are close to 1. To illustrate the idea, consider a simple function 

$$y = w_1 \tanh(w_2 x + b_2) + b_1$$

Given a reasonable $x$ (e.g., $x\approx 0.1$), if $|w_2|$ or $|b_2|$ is large, $y \approx b_1 \pm w_1$, and thus the effective DOF is 2; if $w_2$ and $b_2$ is close to 0, $y\approx w_1 w_2 x + w_1 b_2 + b_1$, perturbation of all four parameters $w_1$, $w_2$, $b_1$, $b_2$ may contribute to the change of $y$, and thus the effective DOF is 4. In sum, trust region methods yield weights and biases with smaller magnitudes compared to BFGS/LBFGS in general, and thus achieve more effective DOFs. 

This conjecture is confirmed by the following plot, which shows the histogram of the intermediate activation values. We fixed the input $x = (0.5,0.5)$ (the midpoint of the computational domain), and collected all the outputs of the $\tanh$ function within the DNN. The figure shows that compared to the trust region method, the activation values of ADAM, BFGS and LBFGS are more concentrated near the extreme values $-1$ and $1$. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/activation_dist.png?raw=true)


How can trust region methods manage the magnitudes of the weights and biases? The benefit is intrinsic to how the trust region method works: it only searches for "optimal solution" with a small neighborhood of the current state. However, BFGS and LBFGS searches for "optimal solution" along a direction aggressively. Given so many local minima, it is very likely that BFGS and LBFGS get trapped in a local minimum with smaller effective DOFs. In this perspective, trust region methods are useful methods for avoiding (instead of "escaping") bad local minima. 


| ADAM        | BFGS | LBFGS| Trust Region |
|---|---|---|---| 
|![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/nn_dynamic_ADAM_eig.png?raw=true)| ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/nn_dynamic_BFGS_eig.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/nn_dynamic_LBFGS_eig.png?raw=true)| ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/nn_dynamic_TrustRegion_eig.png?raw=true)|
|132|34|41|38|

In the above plot, we show the eigenvalue distribution of the Hessian matrix for 

$$l(\theta) = \sum_{i=1}^n (\kappa_\theta(x_i) - \kappa_i)^2$$

Here $\kappa_i$ is the true $\kappa$ value at location $x_i$ ($x_i$ is the Gauss quadrature point), and $\kappa_\theta(x_i)$ is the DNN estimate. We get rid of the PDE out of the loss function. The pattern of the eigenvalue distribution---a few positive eigenvalues accompanied by zero eigenvalues---still persists. The difference is that the number of positive eigenvalues are slightly larger than the loss function that couples DNNs and PDEs.This implies that PDEs restricts effective DOFs. We attribute the diminished effective DOFs to the physical constraints imposed by PDEs. 



## Example: FEM for Static Poisson's Equation

Consider the Poisson's equation again. This time, the loss function is formulated as 

$$L(\theta) = \sum_i (u_{obs}(x_i) - u_\theta(x_i))^2$$

Here $u_\theta$ is the numerical solution to the Poisson's equation. The evaluation of $\nabla^2_\theta L(\theta)$ requires back-propagating the Hessian matrix through various operators including the sparse solver (see [this post](https://kailaix.github.io/ADCME.jl/dev/second_order_pcl/#Example:-Developing-Second-Order-PCL-for-a-Sparse-Linear-Solver) for how the back-propagation rule is implemented). 

The following shows an example of the implementation, which is annotated for convenience. 

```julia
using AdFem 
using PyPlot 
using LinearAlgebra
using JLD2

# disable auth reordering so that the Hessian back-propagation works properly 
ADCME.options.sparse.auto_reorder = false

using Random; Random.seed!(233)
sess = Session()

function simulate(θ)

    # The first part is a standard piece of codes for doing numerical simulation in ADCME 
    global mmesh = Mesh(10,10,0.1)
    x = gauss_nodes(mmesh)
    Fsrc = eval_f_on_gauss_pts((x,y)->1.0, mmesh)
    
    global kappa = squeeze(fc(x, [20,20,20,1], θ)) + 0.5
    A_orig = compute_fem_laplace_matrix1(kappa, mmesh)
    F = compute_fem_source_term1(Fsrc, mmesh)
    global bdnode = bcnode(mmesh)
    global A, F = impose_Dirichlet_boundary_conditions(A_orig, F, bdnode, zeros(length(bdnode)))

    @load "Data/fwd_data.jld2" sol 
    SOL = sol 
    
    global sol = A\F

    global loss = sum((sol-SOL)^2)

    # We now calculate some extra tensors for use in Hessian back-propagation 

    # We use the TensorFlow tf.hessians (hessian in ADCME) to calculate the Hessian of DNNs. Note this algorithm is different from second order PCL 
    global H_dnn_pl, W_dnn_pl = pcl_hessian(kappa, θ, loss)
    global dsol = gradients(loss, sol)
    global dθ = gradients(loss, θ)

    # We need the indices for sparse matrices in the Hessian back-propagation 
    init(sess)
    global indices_orig = run(sess, A_orig.o.indices) .+ 1
    global indices = run(sess, A.o.indices) .+ 1

end


function calculate_hessian(θ0)
    # Retrieve intermediate values. Note in an optimized implementation, these values should already be available in the "tape". However, because second order PCL is currently in development, we recalculate these values for simplicity 
    A_vals, sol_vals, dsol_vals  = run(sess, [A.o.values, sol, dsol], θ=>θ0)

    # SoPCL for `loss = sum((sol-SOL)^2)`
    W = pcl_square_sum(length(sol))

    # SoPCL for `sol = A\F` 
    W = pcl_sparse_solve(indices, 
        A_vals, 
        sol_vals, 
        W, 
        dsol_vals)
        
    # SoPCL for `A, F = impose_Dirichlet_boundary_conditions(...)`
    J = pcl_impose_Dirichlet_boundary_conditions(indices_orig, bdnode, size(indices,1))
    W = pcl_linear_op(J, W)

    # SoPCL for `A_orig = compute_fem_laplace_matrix1(kappa, mmesh)`
    J = pcl_compute_fem_laplace_matrix1(mmesh)
    W = pcl_linear_op(J, W)

    # SoPCL for DNN
    run(sess, H_dnn_pl, feed_dict=Dict(W_dnn_pl=>W, θ=>θ0))
end

function calculate_gradient(θ0)
    run(sess, dθ, θ=>θ0)
end

function calculate_loss(θ0)
    L = run(sess, loss, θ=>θ0)
    @info "Loss = $L"
    L
end

# The optimization step
θ = placeholder(fc_init([2,20,20,20,1]))
simulate(θ)
res = opt.minimize(
    calculate_loss,
    θ0,
    method = "trust-exact",
    jac = calculate_gradient,
    hess = calculate_hessian,
    tol = 1e-12,
    options = Dict(
        "maxiter"=> 5000,
        "gtol"=>0.0 # force the optimizer not to stop
    )
)
```

It is very important that before we perform the optimization, we carry out the Hessian test using the [`test_hessian`](@ref) function.

```julia
function test_f(θ0)
    calculate_gradient(θ0), calculate_hessian(θ0)
end
θ0 = run(sess, θ)
test_jacobian(test_f, θ0, scale=1e-3)
```

This should give us a plot as follows:

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/hessian_test.png?raw=true)

Now let us consider the inverse problem. First we generate the observation using 

$$\kappa(x) = \frac{1}{1+\|x\|_2^2}+1$$

The observation is shown as below

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/sol.png?raw=true)

We use the full field data for simplicity, although our method also applies to sparse observations. The following plots shows results where the trust region method performs significantly better than the BFGS and LBFGS method. 


**Case 1**

| Description         | Result |
|--------------|---|
| Loss         | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/loss2.png?raw=true) |
| LBFGS        | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/lbfgs_2.png?raw=true) |
| BFGS         | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/bfgs_2.png?raw=true) |
| Trust Region | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/tr_2.png?raw=true)  |


**Case 2**

| Description         | Result |
|--------------|---|
| Loss         | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/loss2333.png?raw=true) |
| LBFGS        | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/lbfgs_2333.png?raw=true) |
| BFGS         | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/bfgs_2333.png?raw=true) |
| Trust Region | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/tr_2333.png?raw=true)  |


**Case 3**

| Description         | Result |
|--------------|---|
| Loss         | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/loss23333.png?raw=true) |
| LBFGS        | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/lbfgs_23333.png?raw=true) |
| BFGS         | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/bfgs_23333.png?raw=true) |
| Trust Region | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/tr_23333.png?raw=true)  |


In the following plots, we show the absolute eigenvalue distributions of Hessians at the terminal point for Case 2. The red dashed line represents the level $10^{-6}\lambda_{\max}$, where $\lambda_{\max}$ is the maximum eigenvalue. 

| LBFGS         | BFGS | Trust Region|
|--------------|---|---|
|![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/lbfgs_eigs.png?raw=true)|![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/bfgs_eigs.png?raw=true)|![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/tr_eigs.png?raw=true)|

Eigenvalues that lie below the red dashed line can be treated as zero. This means that for BFGS and the trust region method, the optimizers find local minima. In fact, we show 

$$F(\alpha) = L(x^* + \alpha v)$$

in the following plots, where $x^*$ is the converged point for LBFGS, $v$ is the eigenvector corresponding to either the minimum or maximum eigenvalues of the Hessian. The profile for the former case is quite flat, indicating that small perturbation along the eigenvector direction makes little change to the loss function. Thus, for LBFGS, we can also assume that a local minimum is found. 


Interestingly, even though all optimizers find local minima. The final loss functions and errors are quite different. Trust methods perform much better in these cases compared to BFGS and LBFGS (in some other cases, BFGS may perform better). 

| $\lambda_{\min}$         | $\lambda_{\max}$  |
|--------------|---|
|![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/lbfgs_vmin.png?raw=true)|![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/static_poisson/lbfgs_vmax.png?raw=true)|

The problem itself is nonconvex and has many local minima---different from the common belief that in deep learning, stationary points are usually saddle points if they are not the global minimum. Trust region methods do not guarantee that we can find a global  minimum, or even a "good" local minimum. However, because trust region methods shows faster convergence and superior accuracy in many cases, it never harms to add trust region methods into the optimization tool box. Additionally, the Hessian calculated using the second order PCL is a powerful weapon for diagnosing the convergence and provides curvature information for more sophisticated optimizers. 

## Limitations 

Despite many promising features of the trust region method, it is not without limitations, which we want to discuss here. The current trust-region method requires calculating the Hessian matrix. Firstly, computing the Hessian matrix can be technically difficult, especially when DNNs are coupled with a sophisticated numerical PDE solver. There are many existing techniques for computing the Hessian. The TensorFlow backend supports Hessian computation concurrently, but it requires users to implement rules for calculating "gradients of gradients". Additionally, TensorFlow uses reverse-mode automatic differentiation to evaluate the Hessian. This means that TensorFlow loops over each gradient component and calculating a row of Hessian at a time. This does not leverage the symmetry of Hessians and can be quite inefficient if the number of unknowns is large. Another approach, the edge pushing algorithm, uses one backward pass to evaluate the Hessian. This approach takes advantage of the symmetry of Hessians. However, the implementation can be quite convolved and computations can be expensive in some scenarios. We will cover this topic in more details in [another post](https://kailaix.github.io/ADCME.jl/dev/second_order_pcl/#Second-Order-Physics-Constrained-Learning). 


## Conclusion

Trust-region methods are a class of global optimization techniques. They are less popular in the deep learning approach because the DNNs tend to be huge and the computation of Hessians is expensive. However, they are very suitable for many computational engineering problems, where DNNs are typically small, and convergence as well as accuracy is a critical concern. Our point of view is that although the Hessian computations are expensive, they are quite rewarding. Future researches will focus on efficient computation and automation of Hessian computations. 
