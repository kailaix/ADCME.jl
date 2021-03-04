# The Mathematical Structure of DNN Hessians

It has been observed empirically that the converged weights and biases of DNNs are close to a proper initial guess. This indicates that DNNs are well approximated with its first order Taylor expansion. 

Let us consider a scalar-valued DNN $f(x, \theta)$, where $x$ is a $d$-dimensional input and $\theta$ is the weights and biases. The initial guess is $\theta_0$. Then we have 

$$f(x, \theta) \approx f(x, \theta_0) + \nabla_\theta f(x, \theta_0) (\theta - \theta_0)\tag{1}$$

Consider a quadratic loss function 

$$L(\theta) = \sum_{i=1}^m (f(x_i, \theta) - y_i)^2$$

Using the Taylor expansion Eq. 1, we have 

$$L(\theta) \approx \sum_{i=1}^m (f(x_i, \theta_0) + \nabla_\theta f(x_i, \theta_0)^T (\theta - \theta_0) - y_i)^2 \tag{2}$$

Eq. 2 is a quadratic function of $\theta$, and the Hessian matrix is given by 

$$H = \frac{1}{2}\sum_{i=1}^m \nabla_\theta f(x_i, \theta_0) \nabla_\theta f(x_i, \theta_0)^T$$

Let us consider $\nabla_\theta f(x_i, \theta_0) \nabla_\theta f(x_i, \theta_0)^T$, which is a rank-one matrix. Therefore, one straight-forward corollary is 

$$\text{rank}(H) \leq m$$

That is, for small data, the rank of $H$ is small, and the rank of $H$ is always no greater than the size of samples. The implication is significant for applications in computational engineering: unlike many machine learning problems, where plenty of training data are available, data are usually scarce in computational engineering (e.g., expensive to collect in experiments). Thus a low rank Hessian is predominate in engineering applications. 

We can also write $H$ as follows:

$$H = X^TX, \quad X = \begin{bmatrix}\nabla_\theta f(x_1, \theta_0)^T \\ \nabla_\theta f(x_2, \theta_0)^T \\ \ldots \\ \nabla_\theta f(x_m, \theta_0)^T\end{bmatrix}$$

We have $\text{rank}(H) = \text{rank}(X^TX) \leq \text{rank}(X)$, that is, the rank of $H$ is upper bounded by the rank of $X$. The rank of $X$ determines on the information provided by $\{x_i\}_{i=1}^m$. For example, if most of $x_i$ are the same or similar, we expect $X$ to have a low rank. 


We demonstrate the rank of Hessians using the following examples: let $x_i = \frac{i-1}{99}$, $i = 1, 2, \ldots, 100$, and $y_i = \sin(\pi x_i)$. We train a scalar-valued deep neural network $f(x, \theta)$ with  3 hidden neurals, 20 neurons per layer, and tanh activation functions. The loss function is 

$$L(\theta) = \sum_{i\in\mathcal{I}} (y_i - f(x_i, \theta))^2$$

Here $\mathcal{I}$ is a subset of $\{1,2,\ldots, 100\}$ and linearly spaced between 1 and 100. We train the deep neural network using L-BFGS-B optimizer, and calculate the Hessian matrix $H = \frac{\partial^2 L}{\partial \theta \partial \theta^T}$. We report the number of positive eigenvalues, which is defined as 
$$\lambda > 10^{-6} \lambda_{\max}$$
Here $\lambda_{\max}$ is the maximum eigenvalue. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/second_order_optimizer/hessian_eigenvalue_rank.png?raw=true)
