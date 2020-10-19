
# Numerical Scheme in ADCME: Finite Element Example

The purpose of this tutorial is to show how to work with the finite element method (FEM) in ADCME. The tutorial is divided into two part:

* In the first part, we implement a finite element code for a time independent Poisson's equation in 1D and 2D. We present two styles of implementing a finite element code: using vectorized expression and low level C++ implementation using custom operators. The first approach is elegant and only uses ADCME syntax. The second approach is more flexible and allows for efficient optimization. However, the limitation is that you are responsible to calculate the sensititity of your finite element sensitivity matrix. 


* The second part is about solving a time dependent problem. Here we use finite element methods for the spatial discretization and a backward Euler for time integration. The custom operator approach is used. In this example, you will understand how [`while_loop`](@ref) can help avoid creating a computational graph for each time step. This is important because for many applications the number of time steps can be enormous.

## Poisson's Problem: Vectorized Implementation 

Let us consider the following Poisson's equation in $(0,1)$:

$$(\kappa(x) u'(x))' = f(x)\qquad u(0) = u(1) = 0\tag{1}$$

To make the problem more interesting, we make $\kappa$ parameterized by `a`, which is constructed using [`constant`](@ref) so we can keep track of the dependencies of intermediate values on `a`. 

$$\kappa(x) = \frac{1}{1+x^2}, \ u(x) = x(1-x)$$

and $f(x)$ can be calculated according to Equation 1. We use the finite element method with a linear basis to solve Equation 1: find $u\in H_0^1((0,1))$, such that 

$$\int_0^1 \kappa(x) u'(x) v'(x) dx = \int_0^1 f(x) v(x) dx \quad \forall v\in H_0^1((0,1))$$

We consider a uniform grid with $n$ intervals of equal lengths. The common approach for assembling the finite element matrix $A$ is to iterate over elements and compute the contribution  $\int_E \kappa(x) \phi'_i(x)\phi'_j(x) dx$ and add it to the entry $A_{ij}$; here $\phi_i(x)$ is the basis function associated with node $i$. 

The integration is usually done with numerical integration. Here we consider the Gauss quadrature. Consider the $i$-th element, the local stiffness matrix is 

$$L_i = \sum_{k=1}^G h\begin{bmatrix} \frac{w_k\kappa(x_k)}{h^2} & -\frac{w_k\kappa(x_k)}{h^2} \\ -\frac{w_k\kappa(x_k)}{h^2} & \frac{w_k\kappa(x_k)}{h^2} \end{bmatrix} = \begin{bmatrix} 1 & -1\\ -1 & 1 \end{bmatrix}\frac{\sum_{k=1}^G w_k\kappa(x_k)}{h} $$

Here $(\xi_k, w_k)$ are Gauss quadrature points and weights on $[0,1]$, and 

$$x_k = (1-\xi_k) (i-1)h + \xi_k ih$$


The corresponding DOF (degrees of freedom) matrix, i.e., the mapping of local index to global index, is 

$$D_i = \begin{bmatrix}(i,i) & (i,i+1) \\ (i+1, i) & (i+1, i+1)\end{bmatrix}$$


```julia
xk = zeros(n, length(ξ))
for i = 1:length(ξ)
    xk[:,i] = x[1:end-1] * (1-ξ[i]) + x[2:end] * ξ[i]
end
xk = constant(xk) # convert xk from Julia array to tensor
s = kappa(xk) * w / h
i0 = Array(1:n)
i1 = Array(2:n+1)
II = [i0;i0;i1;i1]
JJ = [i0;i1;i0;i1]
VV = [s;-s;-s;s]
A = SparseTensor(II, JJ, VV, n+1, n+1)
```

The right hand side $\int_\Omega f(x) v(x) dx$ can be computed in a similar fashion: the local contribution $l_i$ and DOF $d_i$ are 

$$l_i = \sum_{k=1}^G h\begin{bmatrix}
    w_k f(x_k) (1-\xi_k)\\ 
    w_k f(x_k) \xi_k\\ 
\end{bmatrix}\qquad d_i = \begin{bmatrix}
    i\\ 
    i+1
\end{bmatrix}$$

This is done with 
```julia
rhs = zeros(n+1)
s = [f(xk) * (w .* (1 .-ξ)); f(xk) * (w .* ξ)] * h
rhs = -vector([i0;i1], s, n+1)
```

### Full Code Listing 
```julia
using ADCME

function kappa(x)
    return 1/(1+a*x^2)
end

function f(x)
    return -2*a*x .* (1 - 2*x) ./(a*x^2 + 1)^2 - 2/(a*x^2 + 1)
end

function uexact(x)
    return x*(1-x)
end

n = 100
h = 1/n 
x = Array(LinRange(0, 1, n+1))
a = constant(1.0)
ξ = [0.1127016653792583; 0.5;0.8872983346207417]
w = [5/18; 4/9; 5/18]

xk = zeros(n, length(ξ))
for i = 1:length(ξ)
    xk[:,i] = x[1:end-1] * (1-ξ[i]) + x[2:end] * ξ[i]
end

xk = constant(xk) # convert xk from Julia array to tensor

# Assemble left hand side 
s = kappa(xk) * w / h
i0 = Array(1:n)
i1 = Array(2:n+1)
II = [i0;i0;i1;i1]
JJ = [i0;i1;i0;i1]
VV = [s;-s;-s;s]
A = SparseTensor(II, JJ, VV, n+1, n+1)


# Assemble right hand side 
rhs = zeros(n+1)
s = [f(xk) * (w .* (1 .-ξ)); f(xk) * (w .* ξ)] * h
rhs = -vector([i0;i1], s, n+1)

# Impose boundary condition using static condensation 
A = A[2:end-1, 2:end-1]
rhs = rhs[2:end-1]

# Solve 
sol = A\rhs
sess = Session(); init(sess)

solution = run(sess, sol)
```

The result for $a=1$ is shown below

```@raw html
<center><img src="https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/poisson.png?raw=true" width="50%"></center>
```

## Poisson's Problem: Custom Operators

Vectorized implementation is great and elegant when we can easily figure out the mathematical formula. However, for more complicated problems and modular development, it is awkward to reason about vectorization each time. A preferred approach is to loop over elements and focus on calculating local contribution. However, a direct for loop will create a large computational graph and will not be able to take advantage of efficient implementation (e.g., parallel computing). 

In what follows, we consider another approach: custom operator. To motivate our method, we consider a 2D Poisson's equation in a domain $\Omega$

$$\nabla \cdot (\kappa(x) \nabla u) = f(x)\qquad u(x) = 0, x\in \partial\Omega$$

The weak formulation is given by 

$$\int_\Omega \kappa(x) \nabla u \cdot \nabla v dx = - \int_\Omega f(x) v(x) dx\quad \forall v \in H_0^1(\Omega)$$

The strategy is to implement two custom operators, one for computing the stiffness matrix, and the other for computing the right hand side. We suggest readers to use our [AdFem](https://github.com/kailaix/AdFem.jl) library instead of their own. the AdFem library is built on ADCME and contains a rich set of custom operators that allow users to implement FEM fairly easily. 


```julia
using AdFem 
using PyPlot 

mmesh = Mesh(50, 50, 1/50)

function kappa(x, y)
    return 1/(1+a*(x^2+y^2))
end

function ffun(x, y)
    return -2*x*(1 - x)/(x^2 + y^2 + 1) - 2*x*(-x*y*(1 - y) + y*(1 - x)*(1 - y))/(x^2 + y^2 + 1)^2 - 2*y*(1 - y)/   (x^2 + y^2 + 1) - 2*y*(-x*y*(1 - x) + x*(1 - x)*(1 - y))/(x^2 + y^2 + 1)^2
end

a = constant(1.0)
κ = eval_f_on_gauss_pts(kappa, mmesh, tensor_input = true)
fv = eval_f_on_gauss_pts(ffun, mmesh, tensor_input = true)
A = compute_fem_laplace_matrix1(κ, mmesh)
rhs = -compute_fem_source_term1(fv, mmesh)

bd = bcnode(mmesh)
A, rhs = impose_Dirichlet_boundary_conditions(A, rhs, bd, zeros(length(bd)))
sol = A\rhs 

sess = Session(); init(sess)
U = run(sess, sol)

close("all")
figure(figsize = (15, 5))
subplot(131)
visualize_scalar_on_fem_points(U, mmesh)
subplot(132)
xy = fem_nodes(mmesh)
x, y = xy[:,1], xy[:,2]
visualize_scalar_on_fem_points((@. x*(1-x)*y*(1-y)), mmesh)
subplot(133)
visualize_scalar_on_fem_points(abs.(U - (@. x*(1-x)*y*(1-y))), mmesh)
savefig("poisson2d.png")
```


```@raw html
<center><img src="https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/poisson2d.png?raw=true" width="50%"></center>
```




## Summary

Finite element analysis is a powerful tool in numerical PDEs. However, it is more conceptually sophisticated than the finite difference method and requires more implementation efforts. The important lesson we learned from this tutorial is how to separate the computation into pure Julia and ADCME C++ kernels, and how complex numerical schemes can be implemented in ADCME. 


