# Overview

ADCME is suitable for conducting inverse modeling in scientific computing. The purpose of the package is to: (1) provide differentiable programming framework for scientific computing based on TensorFlow automatic differentiation (AD) backend; (2) adapt syntax to facilitate implementing scientific computing, particularly for numerical PDE discretization schemes; (3) supply missing functionalities in the backend (TensorFlow) that are important for engineering, such as sparse linear algebra, constrained optimization, etc. Applications include

- full wavelength inversion

- constitutive modeling in solid mechanics

- learning hidden geophysical dynamics

- physics based machine learning

- parameter estimation in stochastic processes

The package inherents the scalability and efficiency from the well-optimized backend TensorFlow. Meanwhile, it provides access to incooperate existing C/C++ codes via the custom operators. For example, some functionalities for sparse matrices are implemented in this way and serve as extendable "plugins" for ADCME. 

![](./assets/summary.png)

Read more about the methodology, the philosophy, the insights and the perspective about ADCME: [slides](./assets/Slide/ADCME.pdf). Start with [tutorial](./tutorial.md) to solve your own inverse modeling problems.

**Installation**

It is recommended to install ADCME via
```julia
using Pkg; Pkg.add("ADCME")
```

However, in some cases, you may want to install the package and configure the environment manually. 

Step 1: Install `ADCME` on a computer with Internet access and zip all files from the following paths

```julia
julia> using Pkg
julia> Pkg.depots()
```

The files will contain all the dependencies. 

Step 2: Build `ADCME` mannually. 

```julia
using Pkg;
ENV["manual"] = 1
Pkg.build("ADCME")
```

However, in this case you are responsible for configuring the environment by modifying the file

```julia
using ADCME; 
print(joinpath(splitdir(pathof(ADCME))[1], "deps/deps.jl"))
```

**Quick Overview**

Let's consider a simple problem: we want to solve the unconstrained optimization problem

$$f(\mathbf{x}) = \sum_{i=1}^{n-1}\left[ 100(x_{i+1}-x_i^2) + (1-x_i)^2 \right]$$

where $x_i\in [-10,10]$ and $n=100$. 

We solve the problem using the L-BFGS-B method. 

```julia
using ADCME
n = 100
x = Variable(rand(n)) # Use `Variable` to mark the quantity that gets updated in optimization
f = sum(100((x[2:end]-x[1:end-1])^2 + (1-x[1:end-1])^2)) # Use typical Julia syntax 
sess = Session(); init(sess) # Create and initialize a session is mandatory for activating the computational graph
BFGS!(sess, f, var_to_bounds = Dict(x=>[-10.,10.]))
```

To get the value of $\mathbf{x}$, we use [`run`](@ref) to extract the values 

```julia
run(sess, x)
```

The above command will return a value close to  the optimal values $\mathbf{x} = [1\ 1\ \ldots\ 1]$. 

**Contributing**

Contribution and suggestions are always welcome. In addition, we are also looking for research collaborations. You can submit issues for suggestions, questions, bugs, and feature requests, or submit pull requests to contribute directly. You can also contact the authors for research collaboration. 

## Video Lectures and Slides

Here is a collection of video lectures and slides related to ADCME.
1. Inverse Modeling with ADCME. [slides](https://kailaix.github.io/ADCME.jl/dev/assets/Slide/ADCME.pdf)
2. Automatic Differentiation. [slides](https://kailaix.github.io/ADCME.jl/dev/assets/Slide/AD.pdf)
3. Inverse Modeing. [slides](https://kailaix.github.io/ADCME.jl/dev/assets/Slide/Inverse.pdf)
4. Physics Constrained Learning. [video](https://www.youtube.com/watch?v=0r9qekmZGqk&t=480s)

