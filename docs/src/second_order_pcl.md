# Second Order Physics Constrained Learning

In this note, we describe the second order physics constrained learning (PCL) for efficient calculating Hessians using computational graphs. To begin with, let $x\in\mathbb{R}^d$ be the coordinates and consider a chain of operations 

$$l(x) = F_n\circ F_{n-1}\circ \cdots \circ F_1(x) \tag{1}$$

Here $l$ is a scalar function, which means $F_n$ is a scalar function. It is not hard to see that we can express any computational graph with a scalar output using Eq. 1. For convenience, given a fixed $k\in\{1,2,\ldots, n\}$, we define $\Phi$, $F$ as follows

$$l(x) = \underbrace{F_n\circ F_{n-1}\circ \cdots \circ F_k}_{\Phi} \circ \underbrace{F_{k-1}\cdots \circ F_1}_{F}(x)$$

We omit $k$ in $\Phi$, $F$ for clarity. 

## Calculating the Hessian in TensorFlow

TensorFlow provides [`tf.hessians`](https://www.tensorflow.org/api_docs/python/tf/hessians) to calculate hessian functions. ADCME exposes this function via [`hessians`](@ref). The idea is to first construct a computational graph for the gradients $\nabla_x l(x)$, then for **each component** of $\nabla_x l(x)$, $\nabla_{x_i} l(x)$, we construct a gradient back-propagation computational graph 

$$\nabla_x \nabla_{x_i} l(x)$$

This gives us a row/column in the Hessian matrix. The following shows the main code from TensorFlow:

```python
_, hessian = control_flow_ops.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j + 1,
                           result.write(j, gradients(gradient[j], x)[0])),
        loop_vars
    )
```

We see it's essentially a loop over each component in `gradient`

Albeit straight-forward, this approach suffers from three drawbacks:

1. The algorithm does not leverage the symmetry of the Hessian matrix. The Hessian structure can be exploited for efficient computations. 
2. The algorithm can be quite expensive. Firstly, it requires a gradient back-propagation over each component of $\nabla_x l(x)$. Although TensorFlow can concurrently evaluates these Hessian rows/columns concurrently, the dimension of $x$ can be very large and therefore the computations cannot be fully parallelized. Secondly, each back-propagation of $\nabla_{x_i} l(x)$ requires both forward computation $l(x)$ and gradient back-propagation for $\nabla_{x} l(x)$, we need to carefully arrange the computations and storages so that these intermediate results can be reused. Otherwise, redundant computations lead to extra costs. 
3. The most demanding requirements of the algorithm is that we need to implement---for every operator---the "gradients of gradients". Although simple for some operators (there are already existing implementations for some operators in TensorFlow!), this can be very hard for sophisticated operators, e.g., implicit operators. 


## Second Order Physics Constrained Learning

Here we consider the second order physics constrained learning. The main idea is to apply the implicit function theorem to 

$$l = \Phi(F(x)) \tag{2}$$

twice. First, we introduce some notation:

$$\Phi_k(y) = \frac{\partial \Phi(y)}{\partial y_k}, \quad \Phi_{kl}(y) = \frac{\partial^2 \Phi(y)}{\partial y_k \partial y_l}$$

$$F_{k,l}(x) = \frac{\partial F_k(x)}{\partial x_l}, \quad F_{k,lr}(x) = \frac{\partial^2 F_k(x)}{\partial x_l\partial x_r}$$

We take the derivative with respect to $x_i$ on both sides of Eq. 2, and get

$$\frac{\partial l}{\partial x_i} = \Phi_k F_{k,i} \tag{3}$$

We take the derivative with respect to $x_j$ on both sides of Eq. 3, and get 

$$\frac{\partial^2 l}{\partial x_i\partial x_j} = \Phi_{kr} F_{k,i}F_{r,j} + \Phi_k F_{k, ij} \tag{4}$$

Here we have used the Einstein notation. Let $\bar\Phi = \nabla \Phi$ but we treat $x$ as a independent variable of $\bar\Phi$, then we can rewrite Eq. 4 to 

$$\nabla_x^2 l = (\nabla_x F) \nabla^2_x\Phi (\nabla_x F)^T + \nabla_x^2 (\bar\Phi^T F)\tag{5}$$

Note the values of $\bar\Phi$ is already available in the gradient back-propagation. 

## Algorithm

Based on Eq. 5, we have the following algorithm for calculating the Hessian

1. Initialize $H = 0$
2. for $k = n-1, n-2,\ldots, 1$
   * Calculate $J = \nabla F_k$ and extract $\bar \Phi_{k+1}$ from the gradient back-propagation tape.
   * Calculate $Z = \nabla^2 (\bar\Phi_{k+1}^T F_k)$
   * $H \gets JHJ^T + Z$

This algorithm only requires one backward pass and constructs the Hessian iteratively. Additionally, we can leverage the symmetry of the Hessian when we do the calculations in the second step. This algorithm also doesn't require looping over each components of the gradient. 

However, the challenge here is that we need to calculate $\nabla F_k$ and $Z = \nabla^2 (\bar\Phi_{k+1}^T F_k)$. Developing a complete support of such calculations for all operators  can be a time-consuming task. But due to the benefit brought by the trust region method, we deem it to be a rewarding investment. 


## Example: Developing Second Order PCL for a Sparse Linear Solver

Here we consider an application of second order PCL for a sparse solver. We focus on the operator that takes the sparse entries of a matrix $A\in\mathbb{R}^{n\times n}$ as input, and outputs $u$

$Au = f$

Let $A = [a_{ij}]$, and some of $a_{ij}$ are zero. According to 2nd order PCL, we need to calculate $\frac{\partial u_k}{\partial a_{ij}}$ and $\frac{\partial^2 (y^T u)}{\partial a_{ij} \partial a_{rs}}$. 

We consider a multi-index $l$ and $r$. We take the gradient with respect to $a_l$ on both sides of 

$$a_{i1}u_1 + a_{i2}u_2 + \ldots + a_{in}u_n = f_i$$

which leads to 

$$a_{i1}^lu_1 + a_{i2}^lu_2 + \ldots + a_{in}^lu_n + a_{i1}u^l_1 + a_{i2}u^l_2 + \ldots + a_{in}u^l_n = 0\tag{6}$$

Here the superscript indicates the derivative. 

Eq. 6 leads to 

$$u^l = -A^{-1}A^l u \tag{7}$$

Note at most one entry in $A^l$ is nonzero, and therefore at most one entry in $A^l u$ is nonzero. Thus to calculate Eq. 7, we can calculate the inverse $A^{-1}$ first, and then $u^l$ can be obtained cheaply by taking a column from $A^{-1}$. The complexity will be $\mathcal{O}(n^3)$---the cost of inverting $A^{-1}$.

Now take the derivative with respect to $a_r$ on both sides of Eq. 6, we have

$$\begin{aligned}
a_{i1}^lu_1^r + a_{i2}^lu_2^r + \ldots + a_{in}^lu_n^r + \\ a_{i1}^ru^l_1 + a_{i2}^ru^l_2 + \ldots + a_{in}^ru^l_n +\\ a_{i1}u^{rl}_1 + a_{i2}u^{rl}_2 + \ldots + a_{in}u^{rl}_n = 0
\end{aligned}$$

which leads to 

$$Au^{rl} = -A^l u^r - A^r u^l$$

Therefore, 

$$(y^Tu)^{rl} = - y^TA^{-1}(A^l u^r + A^r u^l)$$

We can calculate $z^T = y^TA^{-1}$ first with a cost $\mathcal{O}(n^2)$. Because $u^r$, $u^l$ has already been calculated and $A^l$, $A^r$ has at most one nonzero entry, $A^l u^r + A^r u^l$ has at most two nonzero entries. The calculation $z^T(A^l u^r + A^r u^l)$ can be done in $\mathcal{O}(1)$ and therefore the total cost is $\mathcal{O}(d^2)$, where $d$ is the number of nonzero entries. 


Upon obtaining $\frac{\partial u_k}{\partial a_{ij}}$ and $\frac{\partial^2 (y^T u)}{\partial a_{ij} \partial a_{rs}}$, we can apply the recursive the formula to "back-propagate" the Hessian matrix. 