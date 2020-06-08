# Optimizers

ADCME provides a rich class of optimizers and acceleration techniques for conducting inverse modeling. The programming model also allows for easily extending ADCME with customer optimizers. In this section, we show how to take advantage of the built-in optimization library by showing how to solve an inverse problem--estimating the diffusivity coefficient of a Poisson equation from sparse observations--using different kinds of optimization techniques.    


## Generating Training Data 

Consider the Poisson equation in 2D:

$$\begin{aligned}
\nabla  \cdot (\kappa (x,y)\nabla u(x,y)) &= f(x) & (x,y) \in \Omega\\ 
u(x,y) &=0 & (x,y)\in \partial \Omega
\end{aligned}\tag{1}$$

Here $\Omega$ is a L-shaped domain, which can be loaded using `meshread`. 

$$f(x,y) = \sin\left(2\pi y+\frac{\pi}{8}\right)$$

and the diffusivity coefficient $\kappa(x,y)$ is given by 


$$\kappa(x,y) = 2+e^x - y^2$$

We can solve Equation 1 using a standard finite element method. Here, we use NNFEM.jl solve the PDE.

First, we specify the types of element in the domain

```julia
using Revise
using ADCME
using NNFEM
using LinearAlgebra
using PyPlot
using JLD2 

nodes, elems = meshread("$(splitdir(pathof(NNFEM))[1])/../deps/Data/lshape_dense.msh")
elements = []
prop = Dict("name"=> "Scalar1D", "kappa"=>2.0)

for k = 1:size(elems,1)
    elnodes = elems[k,:]
    ngp = 2
    coord = nodes[elnodes,:]
    push!(elements, SmallStrainContinuum(coord, elnodes, prop, ngp))
end
```

Next, we impose proper boundary conditions on the domain

```julia
EBC = zeros(Int64, size(nodes,1))
FBC = zeros(Int64, size(nodes,1))
g = zeros(size(nodes,1))
f = zeros(size(nodes,1))

bd = find_boundary(nodes, elems)
EBC[bd] .= -1

ndims = 1
domain = Domain(nodes, elements, ndims, EBC, g, FBC, f)
```


In order to use the custom operators such as `compute_body_force_terms1`, we need to initialize the domain so that the domain domain is copied to the C++ dynamic library memory. 


```julia
init_nnfem(domain)
```

We can now assemble the coefficient matrix and assemble the right hand side:

```julia
α = 0.4*π/2
d = [cos(α);sin(α)]
f = (x,y)->sin(2π*y + π/8)
fext = compute_body_force_terms1(domain, f)

sol = zeros(domain.nnodes)
gnodes = getGaussPoints(domain)
x, y = gnodes[:,1], gnodes[:,2]

kref = @. 2.0 + exp(x) - y^2
k = vector(1:4:4getNGauss(domain), kref, 4getNGauss(domain)) + vector(4:4:4getNGauss(domain), kref, 4getNGauss(domain))
k = reshape(k, (getNGauss(domain),2,2))
K = s_compute_stiffness_matrix1(k, domain)
```

Note here we have use `vector` to formulate the diffusivity tensor, which is a $2\times 2$ tensor for each Gauss point. Here, each component of `k` has the form 

$$\begin{bmatrix}
   {\kappa ({x_i},{y_i})} & {}   \\
   {} & {\kappa ({x_i},{y_i})}   
\end{bmatrix}$$

where $(x_i,y_i)$ is the corresponding Gauss points. 


The linear system can now be solved and the solution is stored in `sol` with 

```julia
S = K\fext
sess = Session(); init(sess)
sol[domain.dof_to_eq] = run(sess, S)
```

Finally, let us visualize $\kappa(x,y)$ and solution $u(x,y)$

```julia
figure(figsize=(10, 5))
subplot(121)
title("\$\\kappa(x,y)\$")
visualize_scalar_on_undeformed_body(kref, domain)
subplot(122)
visualize_scalar_on_undeformed_body(sol, domain)
title("\$u(x,y)\$")
using Random; Random.seed!(233)
idx = rand(findall(domain.dof_to_eq), 900)
scatter(domain.nodes[idx,1], domain.nodes[idx,2], color="magenta", s=5)
savefig("Data/reference.png")

@save "poisson.jld2" domain kref sol 
```

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/reference.png?raw=true)

The magenta dots in the right panel shows the observed state variables. 

## Inverse Modeling Implementation

The above forward computation is implemented in an AD-capable way; that is, all the operators in the above forward computation possess automatic differentiation capabilities. Therefore, we can easily cast the above code to an inverse modeling version. The inverse problem we want to solve is

> Suppose we have observed some state variables (the magenta dots in the right panel above), and suppose the source term and boundary conditions are known, we want to estimate a **spatially-varying** diffusivity coefficient $\kappa(x,y)$.

Note that in terms of degrees of freedom, the number of unknowns ($\kappa(x_i,y_i)$ at each Gauss points) is far more than the number of observations. Therefore the inverse problem is underdetermined, making it necessary to adopt regularization. Here the neural network representation of $\kappa(x,y)$ is a form of regularization.


Here we show the content for `compute_loss.jl`, which is used in a later text. We use `ArgParse` to manage the command line parameters 

```julia
using ArgParse
using DelimitedFiles
using PyPlot

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--linesearch"
        "--optimizer"
        "--alpha"
            arg_type = Float64
            default = 1.0
        "--atype"
            arg_type = Int64 
            default = 1
        "arg1"
    end

    return parse_args(s)
end

parameters = parse_commandline()
println(parameters)


reset_default_graph() 
@load "poisson.jld2" domain kref sol 
init_nnfem(domain)
α = 0.4*π/2
d = [cos(α);sin(α)]
f = (x,y)->sin(2π*y + π/8)
fext = compute_body_force_terms1(domain, f)

gnodes = getGaussPoints(domain)
x, y = gnodes[:,1], gnodes[:,2]


k0 = fc([x y], [20,20,20,1])|>squeeze
k0 = softplus(k0 + 3.0) + 1e-6 # to avoid factorization error 

k = vector(1:4:4getNGauss(domain), k0, 4getNGauss(domain)) +vector(4:4:4getNGauss(domain), k0, 4getNGauss(domain))
k = reshape(k, (getNGauss(domain),2,2))
K = s_compute_stiffness_matrix1(k, domain)
S = K\fext
s = vector(findall(domain.dof_to_eq), S, domain.nnodes)
using Random; Random.seed!(233)
idx = rand(findall(domain.dof_to_eq), 900)
loss = mean((s[idx] - sol[idx])^2) * 1e10
```

One special thing about this script is that we formulate the loss function `loss` using the hypothetical observation `s[idx]` and true observation `sol[idx]`. Another noticeable difference is that we assume that we do not know $\kappa(x,y)$ and therefore we approximate $\kappa(x,y)$ using a neural network

```julia
k0 = fc([x y], [20,20,20,1])|>squeeze
k0 = softplus(k0 + 3.0) + 1e-6 # to avoid factorization error 
```

We transform the neural network using `softplus` and add a small value to avoid negative or close to zero $\kappa(x,y)$, which is clearly not physical. 


## Choosing Different Optimizers 

### (Accelerated) Gradient Descent Optimizers

A straightforward way to estimate the weights and biases of the neural network is using gradient descent (GD) methods. Variations of gradient descent methods exist, and a large class of such variations is GD with momentum. In ADCME, the following optimizers are provided as built-in gradient descent optimizers

```
Descent Momentum Nesterov 
RMSProp ADAM RADAM 
AdaMax ADAGrad ADADelta 
AMSGrad NADAM
```

Additionally, line search algorithms are very important to accelerate and safeguard convergence. The line search algorithms can be accessed from  [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl). Examples include

```
HagerZhang
BackTracking
MoreThuente
Static
```

The following script shows how to use those optimizers and line search algorithms:


```julia
using LineSearches
using Revise
using ADCME
using NNFEM
using LinearAlgebra
using PyPlot
using JLD2 
using Statistics


include("compute_loss.jl")
ls = BackTracking()

adam = AMSGrad()
sess = Session(); init(sess)
uo = UnconstrainedOptimizer(sess, loss)
x0 = getInit(uo)

for i = 1:1000
    global x0 
    f, df = getLossAndGrad(uo, x0)
    Δ = getSearchDirection(adam, x0, df)
    setSearchDirection!(uo, x0, -Δ)
    α, fx = linesearch(uo, f, df, ls, 100.0)
    x0 -= α*Δ 
end
update!(uo, x0)
```

The above code also show a typical routine for iterative optimization algorithms

```
getLossAndGrad → getSearchDirection → setSearchDirection → linesearch
```

Don't forget `update!` after the optimization is finished. Also don't forget `reset_default_graph()` (in `compute_loss.jl`) before creating any graph nodes. 

The following figures show the loss function and estimated $\kappa(x,y)$ and $u(x,y)$.

```@raw html
<center><img src="https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/opt_loss.png?raw=true" width="50%"></center>
```

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/kandsol.png?raw=true)


### Quasi-Newton Optimizers

We can also use quasi-Newton optimizers, which in general converge much faster but are more computationally expensive and consume more memory. ADCME provides [`LBFGS`](@ref) optimizer, with preconditioners as optional arguments.  

To use `LBFGS`, we can simply substitute the gradient descent optimizers with `LBFGS`

```julia
lbfgs = Optimizer.LBFGS()
...
for i = 1:100
    ...
    Δ = getSearchDirection(lbfgs, x0, df)
    ...
end
update!(uo, x0)
```


### Anderson Acceleration

Anderson acceleration has been used widely in nature science such as chemistry for solving the fixed point problem 

$$f(x) = x$$

It is computationally cheap, making it attractive for accelerating the gradient descent methods and replacing expensive line searches. The idea is that the gradient descent ($g^k$ is the search direction, which may not coincide with the negative gradient direction) 

$$x^{k+1} = x^k + \alpha^k g^k$$

can be viewed as solving a fixed point system 

$$x = x + \alpha^k g^k(x)$$

In ADCME, Anderson acceleration is implemented in  [`AndersonAcceleration`](@ref) [^cvx]. The following script shows how to use `AndersonAcceleration` with a gradient descent optimizer

[^cvx]: The backend of Anderson Acceleration is from [here](https://github.com/cvxgrp/aa).

```julia
using LineSearches
using Revise
using ADCME
using NNFEM
using LinearAlgebra
using PyPlot
using JLD2 
using Statistics
include("compute_loss.jl")

adam = Descent()
aa = AndersonAcceleration()
sess = Session(); init(sess)
uo = UnconstrainedOptimizer(sess, loss)
x0 = getInit(uo)
for i = 1:1000
    global x0 
    f, df = getLossAndGrad(uo, x0)
    Δ = getSearchDirection(adam, x0, df)
    x0 = apply!(aa, x0, Δ)
end
update!(uo, x0)
```

Here we show the loss profile for different acceleration strategies on `AdaMax` optimizer:



| Anderson Acceleration Type I                                 | Anderson Acceleration Type II                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/aaloss.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/aa2loss.png?raw=true) |
| Static                                                       | BackTracking                                                 |
| ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/staticloss.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/btloss.png?raw=true) |


## Extending Optimization Libraries

To extend optimization library, users only need to define an optimization method structure and a function `apply!` with a signature 

```julia
apply!(opt, x, Δ) -> d
```

where `opt` is the optimizer, `x` is the current state, `Δ` is the gradient direction, and `d` is the search direction. Looking at the example, such as `ADAM`, in the source code is helpful.


## Appendix: Benchmark Results for Different Optimizers


Here we present a benchmark result for different optimizers for solving Equation 1. We consider two choices for $f(x,y)$:

* High frequency: $f(x,y) = \sin\left(4\pi y+\frac{\pi}{8}\right)$

* Low frequency: $f(x,y) = \sin\left(2\pi y+\frac{\pi}{8}\right)$

Additionally, in the inverse modeling, we consider two cases: very sparse data ($n_{\text{obs}}=20$) and moderate amount of data ($n_{\text{obs}}=900$). Here $n_{\text{obs}}$ is the number of observations. In the latter case, the amount of data consists roughly 70%~80% of the degrees of freedom. 

The following shows reference solutions and observation distributions for $u(x,y)$. 



| $n_{\text{obs}}$ | Low Frequency                                                | High Frequency                                               |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 20               | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/reference20-1.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/reference20-2.png?raw=true) |
| 900              | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/reference900-1.png?raw=true) | ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/reference900-2.png?raw=true) |



In the following plot, we show the normalized loss function versus the number of function evaluations/gradient evaluations. The normalized loss function is defined as


$${{{1 \over N}\sum\limits_{i = 1}^N {{{({u_{{\rm{obs}}}}({x_i}) - {u_\theta }({x_i}))}^2}} } \over {{1 \over N}\sum\limits_{i = 1}^N {{u_{{\rm{obs}}}}{{({x_i})}^2}} }}$$

Here ${{u_{{\rm{obs}}}}({x_i})}$ is the observation function value at $x_i$ and ${{u_\theta }({x_i})}$ is the hypothetical solution computed using the neural network ($\theta$ denotes the weights and biases of the neural network).

Some of the optimizers break (encountering `Inf` or `NaN`) during the optimization process. We do not include them in the plot.  

### Low Frequency, $n_{\text{obs}}=20$

**Function Evaluations**

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/fn-20-1.png?raw=true)


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/aa-fn-20-1.png?raw=true)
**Gradient Evaluations**


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/g-20-1.png?raw=true)


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/aa-fn-20-1.png?raw=true)



### Low Frequency, $n_{\text{obs}}=900$

**Function Evaluations**

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/fn-900-1.png?raw=true)


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/aa-fn-900-1.png?raw=true)

**Gradient Evaluations**

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/g-900-1.png?raw=true)


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/aa-g-900-1.png?raw=true)


### High Frequency, $n_{\text{obs}}=20$

**Function Evaluations**

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/fn-20-2.png?raw=true)


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/aa-fn-20-2.png?raw=true)
**Gradient Evaluations**


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/g-20-2.png?raw=true)


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/aa-g-20-2.png?raw=true)


### High Frequency, $n_{\text{obs}}=900$

**Function Evaluations**


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/fn-900-2.png?raw=true)


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/aa-fn-900-2.png?raw=true)


**Gradient Evaluations**

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/g-900-2.png?raw=true)


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Optimizers/aa-g-900-2.png?raw=true)



We can see that LBFGS with a HagerZhang linesearch algorithm has the best performance, although it is very computationally expensive. Gradient descent methods without line search algorithms are not stable. However, if we apply Anderson Acceleration, the stability can be improved without additionaly function/gradient evaluations. 
