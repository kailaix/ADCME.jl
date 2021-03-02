<p align="center">
  <img src="https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/ADCME.gif?raw=true" alt="ADCME"/>
</p>



![](https://travis-ci.org/kailaix/ADCME.jl.svg?branch=master)
![](https://github.com/kailaix/ADCME.jl/workflows/Documentation/badge.svg)
![Coverage Status](https://coveralls.io/repos/github/kailaix/ADCME.jl/badge.svg?branch=master)


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/demo.png?raw=true)

The ADCME library (**A**utomatic **D**ifferentiation Library for **C**omputational and **M**athematical **E**ngineering) aims at general and scalable inverse modeling in scientific computing with gradient-based optimization techniques. It is built on the deep learning framework, **graph-mode [TensorFlow](https://www.tensorflow.org/)**, which provides the automatic differentiation and parallel computing backend. The dataflow model adopted by the framework makes it suitable for high performance computing and inverse modeling in scientific computing. The design principles and methodologies are summarized in the [slides](https://kailaix.github.io/ADCMESlides/ADCME.pdf).

Check out more about [slides and videos on ADCME](https://kailaix.github.io/ADCME.jl/dev/videos_and_slides/)!

| [Install ADCME and Get Started (Windows, Mac, and Linux)](https://www.youtube.com/playlist?list=PLKBz8ohiA3IlrCI0VO4cRYZp2S6SYG1Ww)                                | [Scientific Machine Learning for Inverse Modeling](https://www.youtube.com/playlist?list=PLKBz8ohiA3In-ZlvBKbvj_TIQEaboGC9_)             | [Solving Inverse Modeling Problems with ADCME](https://www.youtube.com/playlist?list=PLKBz8ohiA3ImaNykOv56ONnCofEQMg3B8)                 | ...**more** on [ADCME Youtube Channel](https://www.youtube.com/channel/UCeaZFluNatYpkIYcq2TTklw/playlists)!                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![Alt text](https://img.youtube.com/vi/fH0QrqgzUeo/0.jpg)](https://www.youtube.com/playlist?list=PLKBz8ohiA3IlrCI0VO4cRYZp2S6SYG1Ww) | [![Alt text](https://img.youtube.com/vi/tNj7HooU6ek/0.jpg)](https://www.youtube.com/playlist?list=PLKBz8ohiA3In-ZlvBKbvj_TIQEaboGC9_) | [![Alt text](https://img.youtube.com/vi/wP39oNhIXh8/0.jpg)](https://www.youtube.com/playlist?list=PLKBz8ohiA3ImaNykOv56ONnCofEQMg3B8) | [![](https://yt3.ggpht.com/a-/AOh14GjNubZSOZameuPrlKlaCVNWe26UHO1MzXtbI3rP=s288-c-k-c0xffffffff-no-rj-mo)](https://www.youtube.com/channel/UCeaZFluNatYpkIYcq2TTklw/playlists) |


Several features of the library are

* **MATLAB-style Syntax**. Write `A*B` for matrix production instead of `tf.matmul(A,B)`.
* **Custom Operators**. Implement operators in C/C++ for performance critical parts; incorporate legacy code or specially designed C/C++ code in `ADCME`; automatic differentiation through implicit schemes and iterative solvers. 
* **Numerical Scheme**. Easy to implement numerical schemes for solving PDEs.
* **Physics Constrained Learning**. Embed neural network into PDEs and solve with any numerical schemes, including implicit and iterative schemes. 
* **Static Graphs**. Compilation time computational graph optimization; automatic parallelism for your simulation codes.
* **Parallel Computing**. [Concurrent execution](https://kailaix.github.io/ADCME.jl/dev/multithreading/) and model/data parallel [distributed optimization](https://kailaix.github.io/ADCME.jl/dev/mpi/).
* **Custom Optimizers**. Large scale constrained optimization? Use `CustomOptimizer` to integrate your favorite optimizer. Try out prebuilt [Ipopt and NLopt](https://kailaix.github.io/ADCME.jl/dev/customopt/#Dropin-substitute-of-BFGS!-1) optimizers. 
* **Sparse Linear Algebra**. Sparse linear algebra library tailored for scientific computing. 
* **Inverse Modeling**. Many inverse modeling algorithms have been developed and implemented in ADCME, with wide applications in solid mechanics, fluid dynamics, geophysics, and stochastic processes. 
* [![](https://raw.githubusercontent.com/ADCMEMarket/ADCMEImages/a82f1ae2c686e2c2a9eb2bc940540afb004c6503/ADCME/newrelease.svg)](https://github.com/kailaix/AdFem.jl)**Finite Element Method**. Get [AdFem.jl](https://github.com/kailaix/AdFem.jl) today for finite element simulation and inverse modeling! 

Start building your forward and inverse modeling using ADCME today!

| Documentation                                                | Tutorial                                                     | Applications                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![](https://img.shields.io/badge/-Documentation-blue)](https://kailaix.github.io/ADCME.jl/dev) | [![](https://img.shields.io/badge/-Tutorial-green)](https://kailaix.github.io/ADCME.jl/dev/tutorial/) | [![](https://img.shields.io/badge/-Applications-orange)](https://kailaix.github.io/ADCME.jl/dev/apps) |

## Graph-mode TensorFlow for High Performance Scientific Computing

Static computational graph (graph-mode AD) enables compilation time optimization. Below is a benchmark of common AD software from [here](https://github.com/microsoft/ADBench). In inverse modeling, we usually have a scalar-valued objective function, so the left panel is most relevant for ADCME. 

![](https://raw.githubusercontent.com/microsoft/ADBench/master/Documents/figs/2020_Jan.png)

# Installation

1. Install [Julia](https://julialang.org/). 

🎉 Support Matrix

|         |  Julia≧1.3 | GPU | Custom Operator |
|---------| ----- |-----|-----------------|
| Linux   |✔ |  ✔   | ✔               |
| MacOS   |✔ |  ✕   | ✔               |
| Windows | ✔ | ✔   | ✔               |




1. Install `ADCME`
```
using Pkg
Pkg.add("ADCME")
```

❗ FOR WINDOWS USERS: See [the instruction](https://kailaix.github.io/ADCME.jl/dev/windows_installation.md) or [the video](https://www.youtube.com/playlist?list=PLKBz8ohiA3IlrCI0VO4cRYZp2S6SYG1Ww) for installation details. 


2. (Optional) Test `ADCME.jl`
```julia
using Pkg
Pkg.test("ADCME")
```
See [Troubleshooting](https://kailaix.github.io/ADCME.jl/dev/tu_customop/#Troubleshooting-1) if you encounter any compilation problems.

3. (Optional) To enable GPU support, make sure `nvcc` is available from your environment (e.g., type `nvcc` in your shell and you should get the location of the executable binary file), and then type 
```julia
ENV["GPU"] = 1
Pkg.build("ADCME")
```


4. (Optional) Check the health of your installed ADCME and install missing dependencies or fixing incorrect paths. 
```julia
using ADCME 
doctor()
``` 
For manual installation without access to the internet, see [here](https://kailaix.github.io/ADCME.jl/dev/).


# Tutorial

Here we present three inverse problem examples. The first one is a parameter estimation problem, the second one is a function inverse problem where the unknown function does not depend on the state variables, and the third one is also a function inverse problem, but the unknown function depends on the state variables. 

### Parameter Inverse Problem 

Consider solving the following problem

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/readme-eq1.svg?raw=true)

where 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/readme-eq2.svg?raw=true)

Assume that we have observed `u(0.5)=1`, we want to estimate `b`.  In this case, he true value should be `b=1`.

```julia
using LinearAlgebra
using ADCME

n = 101 # number of grid nodes in [0,1]
h = 1/(n-1)
x = LinRange(0,1,n)[2:end-1]

b = Variable(10.0) # we use Variable keyword to mark the unknowns
A = diagm(0=>2/h^2*ones(n-2), -1=>-1/h^2*ones(n-3), 1=>-1/h^2*ones(n-3)) 
B = b*A + I  # I stands for the identity matrix
f = @. 4*(2 + x - x^2) 
u = B\f # solve the equation using built-in linear solver
ue = u[div(n+1,2)] # extract values at x=0.5

loss = (ue-1.0)^2 

# Optimization
sess = Session(); init(sess) 
BFGS!(sess, loss)

println("Estimated b = ", run(sess, b))
```
Expected output 
```
Estimated b = 0.9995582304494237
```

The gradients can be obtained very easily. For example, if we want the gradients of `loss` with respect to `b`, the following code will create a Tensor for the gradient
```
julia> gradients(loss, b)
PyObject <tf.Tensor 'gradients_1/Mul_grad/Reshape:0' shape=() dtype=float64>
```

### Function Inverse Problem: Full Field Data

Consider a nonlinear PDE, 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/readme-eq3.svg?raw=true)

where 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/readme-eq4.svg?raw=true)

Here `f(x)` can be computed from an analytical solution 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/readme-eq5.svg?raw=true)

In this problem, we are given the full field data of `u(x)` (the discrete value of `u(x)` is given on a very fine grid) and want to estimate the nonparametric function `b(u)`. We approximate `b(u)` using a neural network and use the [residual minimization method](https://kailaix.github.io/ADCME.jl/dev/tu_nn/) to find the optimal weights and biases of the neural network. The minimization problem is given by 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/readme-eq6.svg?raw=true)

```julia
using LinearAlgebra
using ADCME
using PyPlot

n = 101 
h = 1/(n-1)
x = LinRange(0,1,n)|>collect

u = sin.(π*x)
f = @. (1+u^2)/(1+2u^2) * π^2 * u + u 
# `fc` is short for fully connected neural network. 
# Here we create a neural network with 2 hidden layers, and 20 neuron per layer. 
# The default activation function is tanh.
b = squeeze(fc(u[2:end-1], [20,20,1])) 

residual = -b.*(u[3:end]+u[1:end-2]-2u[2:end-1])/h^2 + u[2:end-1] - f[2:end-1]
loss = sum(residual^2)

sess = Session(); init(sess)
BFGS!(sess, loss)

plot(x, (@. (1+x^2)/(1+2*x^2)), label="Reference")
plot(u[2:end-1], run(sess, b), "o", markersize=5., label="Estimated")
legend(); xlabel("\$u\$"); ylabel("\$b(u)\$"); grid("on")
```

Here we show the estimated coefficient function and the reference one:

<p align="center">
  <img src="https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/readmenn.png?raw=true" style="zoom:50%;" />
</p>

### Function Inverse Problem: Sparse Data

Now we consider the same problem as above, but only consider we have access to sparse observations. We assume that on the grid only the values of `u(x)` on every other 5th grid point are observable. We use the [physics constrained learning](https://arxiv.org/pdf/2002.10521.pdf) technique and train a neural network surrogate for `b(u)` by minimizing 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/readme-eq7.svg?raw=true)

Here `uᶿ` is the solution to the PDE with

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/readme-eq8.svg?raw=true)

We add 1 to the neural network to ensure the initial guess does not result in a singular Jacobian matrix in the Newton Raphson solver.

```julia
using LinearAlgebra
using ADCME
using PyPlot

n = 101 
h = 1/(n-1)
x = LinRange(0,1,n)|>collect

u = sin.(π*x)
f = @. (1+u^2)/(1+2u^2) * π^2 * u + u 

# we use a Newton Raphson solver to solve the nonlinear PDE problem 
function residual_and_jac(θ, x)
    nn = squeeze(fc(reshape(x,:,1), [20,20,1], θ)) + 1.0
    u_full = vector(2:n-1, x, n)
    res = -nn.*(u_full[3:end]+u_full[1:end-2]-2u_full[2:end-1])/h^2 + u_full[2:end-1] - f[2:end-1]
    J = gradients(res, x)
    res, J
end
θ = Variable(fc_init([1,20,20,1]))
ADCME.options.newton_raphson.rtol = 1e-4 # relative tolerance
ADCME.options.newton_raphson.tol = 1e-4 # absolute tolerance
ADCME.options.newton_raphson.verbose = true # print details in newton_raphson
u_est = newton_raphson_with_grad(residual_and_jac, constant(zeros(n-2)),θ)
residual = u_est[1:5:end] - u[2:end-1][1:5:end]
loss = sum(residual^2)

b = squeeze(fc(reshape(x,:,1), [20,20,1], θ)) + 1.0
sess = Session(); init(sess)
BFGS!(sess, loss)

figure(figsize=(10,4))
subplot(121)
plot(x, (@. (1+x^2)/(1+2*x^2)), label="Reference")
plot(x, run(sess, b), "o", markersize=5., label="Estimated")
legend(); xlabel("\$u\$"); ylabel("\$b(u)\$"); grid("on")
subplot(122)
plot(x, (@. sin(π*x)), label="Reference")
plot(x[2:end-1], run(sess, u_est), "--", label="Estimated")
plot(x[2:end-1][1:5:end], run(sess, u_est)[1:5:end], "x", markersize=5., label="Data")
legend(); xlabel("\$x\$"); ylabel("\$u\$"); grid("on")
```

We show the reconstructed `b(u)` and the solution `u` computed from `b(u)`. We see that even though the neural network model fits the data very well, `b(u)` is not the same as the true one. This problem is ubiquitous in inverse modeling, where the unknown may not be unique. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/buu.png?raw=true)

See [Applications](https://kailaix.github.io/ADCME.jl/dev/tutorial/) for more inverse modeling techniques and examples.

### Under the Hood: Computational Graph

A static computational graph is automatic constructed for your implementation. The computational graph guides the runtime execution, saves intermediate results, and records data flows dependencies for automatic differentiation. Here we show the computational graph in the parameter inverse problem:

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/code.png?raw=true)

See a detailed [tutorial](https://kailaix.github.io/ADCME.jl/dev/tutorial/), or a full [documentation](https://kailaix.github.io/ADCME.jl/dev). 

# Featured Applications

| [Constitutive Modeling](https://kailaix.github.io/ADCME.jl/dev/apps_constitutive_law/) | [Seismic Inversion](https://kailaix.github.io/ADCME.jl/dev/apps_adseismic) | [Coupled Two-Phase Flow and Time-lapse FWI](https://kailaix.github.io/ADCME.jl/dev/apps_ad/) | [Calibrating Jump Diffusion](https://kailaix.github.io/ADCME.jl/dev/apps_levy/) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![law](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/law.png?raw=true)                              | ![law](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/earthquake.png?raw=true)                       | ![law](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/geo.png?raw=true)                              | ![law](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/algo.png?raw=true)                             |

Here are some research papers using ADCME:


1. Li, Dongzhuo, Kailai Xu, Jerry M. Harris, and Eric Darve. "Coupled Time‐Lapse Full‐Waveform Inversion for Subsurface Flow Problems Using Intrusive Automatic Differentiation." Water Resources Research 56, no. 8 (2020): e2019WR027032.

2. Xu, Kailai, Alexandre M. Tartakovsky, Jeff Burghardt, and Eric Darve. "Inverse Modeling of Viscoelasticity Materials using Physics Constrained Learning." arXiv preprint arXiv:2005.04384 (2020).

3. Zhu, Weiqiang, Kailai Xu, Eric Darve, and Gregory C. Beroza. "A General Approach to Seismic Inversion with Automatic Differentiation." arXiv preprint arXiv:2003.06027 (2020).

4. Xu, K. and Darve, E., 2019. Adversarial Numerical Analysis for Inverse Problems. arXiv preprint arXiv:1910.06936.

5. Xu, Kailai, Weiqiang Zhu, and Eric Darve. "Distributed Machine Learning for Computational Engineering using MPI." arXiv preprint arXiv:2011.01349 (2020).

6. Xu, Kailai, and Eric Darve. "Physics constrained learning for data-driven inverse modeling from sparse observations." arXiv preprint arXiv:2002.10521 (2020).

7. Xu, Kailai, Daniel Z. Huang, and Eric Darve. "Learning constitutive relations using symmetric positive definite neural networks." arXiv preprint arXiv:2004.00265 (2020).

8. Xu, Kailai, and Eric Darve. "The neural network approach to inverse problems in differential equations." arXiv preprint arXiv:1901.07758 (2019).

9. Huang, D.Z., Xu, K., Farhat, C. and Darve, E., 2019. Predictive modeling with learned constitutive laws from indirect observations. arXiv preprint arXiv:1905.12530.



**Domain specific software based on ADCME**

[ADSeismic.jl](https://github.com/kailaix/ADSeismic.jl): Inverse Problems in Earthquake Location/Source-Time Function, FWI, Rupture Process 

[FwiFlow.jl](https://github.com/lidongzh/FwiFlow.jl): Seismic Inversion, Two-phase Flow, Coupled seismic and flow equations 

[AdFem.jl](https://github.com/kailaix/AdFem.jl/): Inverse Modeling with the Finite Element Method



# LICENSE

ADCME.jl is released under MIT License. See [License](https://github.com/kailaix/ADCME.jl/tree/master/LICENSE) for details. 


