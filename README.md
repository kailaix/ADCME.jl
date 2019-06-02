# ADCME.jl

![](examples/md/demo.png)

The ADCME library (**A**utomatic **D**ifferentiation Library for **C**omputational and **M**athematical **E**ngineering) is written to facilitate scalable and sophisticated scientific computing that requires the evaluation of gradients by leveraging the power of [TensorFlow](https://www.tensorflow.org/) and [PyTorch](). It is particularly dedicated to the inverse modeling problem, with the ultimate goal -- _Once the forward simulation is implemented, the researcher should be able to do the inverse modeling without substantial effort._

Several features of the library are

* Simple and neat Julia syntax (MATLAB-style). Most of the arithmetic operators are overloaded into Python TensorFlow, which indicates that users will obtain the static graph performance of TensorFlow and do not need to worry about the overhead of scripting in Julia. In addition, since operators are overloaded, users can write Julia-style codes, e.g., one can use `A*B` for matrix production instead of `tf.matmul(A,B)`.

* Custom operators are supported. Users can implement their own operators for the bottleneck operations such as those that are difficult to vectorize in TensorFlow. When implementing custom operators, `ADCME.jl` provides access to the `ATen` library in `PyTorch`, which is equipped with automatic differentiation. This further reduces users' effort to do inverse modeling.

* Static graphs. Static computation graphs are used instead of dynamic graphs. This is a key difference between machine learning and scientific computing. In scientific computing, computation graph optimization _do_ matter. One such example is `while_loop`, where in scientific computing, large numbers of iterations are common and direct implementation results in a large computation graph. Another example is the parallelism of different operators.

# Installation

1. Install [Julia](https://www.tensorflow.org/)

2. Install [TensorFlow](https://www.tensorflow.org/). Please install `1.12` or `1.13` instead of `2.0` since in `2.0`, dynamic graphs are the default. 

3. Install dependencies in Julia
```
julia> ]
pkg> add PyCall 
pkg> add MAT
pkg> add FFTW
```

4. Check the PyCall Python version is consistent with the `TensorFlow`. To check the PyCall Python version, 
```
julia> using PyCall
julia> PyCall.libpython 
```
The output should be something similar to `/anaconda3/envs/py36/lib/libpython3.6m.dylib`. If the PyCall Python version is different from `TensorFlow` python version, type
```
julia> ENV["PYTHON"] = <the python location above>
julia> ]
pkg> build PyCall
```

5. Install `ADCME.jl`
```
julia> ]
pkg> add https://github.com/kailaix/ADCME.jl
```

6. Test `ADCME.jl`
```
julia> ]
pkg> test ADCME
```

# Tutorial

Consider solving the following problem

$$-bu''(x)+u(x) = f(x)\qquad x\in [0,1]\qquad u(0)=u(1)=0$$

where 

$$f(x) = 8 + 4x - 4x^2$$

Assume that we have observed `u(0.5)=1​`, we want to estimate `b​`. The true value in this case should be `b=1`.

```julia
using LinearAlgebra
using ADCME

n = 101 # number of grid nodes in [0,1]
h = 1/(n-1)
x = LinRange(0,1,n)[2:end-1]

b = Variable(10.0) # create a Variable for `b`
A = diagm(0=>2/h^2*ones(n-2), -1=>-1/h^2*ones(n-3), 1=>-1/h^2*ones(n-3)) # discrete Laplacian matrix
B = -b*A + diagm(0=>ones(n-2))  # coefficient matrix
f = @. 8 + 4x - 4x^2 # right hand side
u = B\f # solve the equation
ue = u[div(n+1,2)] # extract values at x=0.5

loss = (ue-1.0)^2 # form the loss function

# Optimization
opt = ScipyOptimizerInterface(loss)
sess = Session(); init(sess)
ScipyOptimizerMinimize(sess, opt)

println("Estimated b = ", run(sess, b))
```
Expected output 
```
INFO:tensorflow:Optimization terminated with:
Message: b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
Objective function value: 1.000001
Number of iterations: 44
Number of functions evaluations: 45
```

The gradients can be obtained very easily. For example, if we want the gradients of `loss` with respect to `b`, the following code will create a Tensor for the gradient
```
gradients(loss, b)
```

# More Documentation

1. [The Power of `while_loop` -- Application to Finite Element Analysis](https://github.com/kailaix/ADCME.jl/tree/master/examples/md/while_loop.pdf)
2. [Writing Custom Operators in `ADCME.jl`](https://github.com/kailaix/ADCME.jl/tree/master/examples/md/custom_op.pdf)
3. [TensorFlow Meets PyTorch: Using PyTorch to Create TensorFlow Custom Operators](https://github.com/kailaix/ADCME.jl/tree/master/examples/md/pytorch.pdf)
4. [Four Types of Forward Simulation Operators to Consider in Automatic Differentiation](https://github.com/kailaix/ADCME.jl/tree/master/examples/md/four_types.pdf)

# Research Work Based on ADCME.jl

The following research articles are based on `ADCME.jl`. They should be considered as advanced applications of `ADCME.jl`. If you find them useful, please cite the relevant article(s). 

1. Daniel Z. Huang, **Kailai Xu** (co-first author), Charbel Farhat, Eric Darve. [Predictive Modeling with Learned Constitutive Laws from Indirect Observations](https://arxiv.org/abs/1905.12530)
2. **Kailai Xu**, Eric Darve. [Calibrating Lévy Processes from Observations Based on Neural Networks and Automatic Differentiation](https://arxiv.org/abs/1812.08883)

# LICENSE
Copyright (c) 2019 Kailai Xu

ADCME.jl is released under GNU GENERAL PUBLIC LICENSE Version 3. See [License](https://github.com/kailaix/ADCME.jl/tree/master/LICENSE) for details. 