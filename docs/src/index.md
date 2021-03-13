# Overview

ADCME is suitable for conducting inverse modeling in scientific computing; specifically, ADCME targets **physics informed machine learning**, which leverages machine learning techniques to solve challenging scientific computing problems. The purpose of the package is to: (1) provide differentiable programming framework for scientific computing based on TensorFlow automatic differentiation (AD) backend; (2) adapt syntax to facilitate implementing scientific computing, particularly for numerical PDE discretization schemes; (3) supply missing functionalities in the backend (TensorFlow) that are important for engineering, such as sparse linear algebra, constrained optimization, etc. Applications include

- physics informed machine learning (a.k.a., scientific machine learning, physics informed learning, etc.)

- coupled hydrological and full waveform inversion

- constitutive modeling in solid mechanics

- learning hidden geophysical dynamics

- parameter estimation in stochastic processes

The package inherits the scalability and efficiency from the well-optimized backend TensorFlow. Meanwhile, it provides access to incorporate existing C/C++ codes via the custom operators. For example, some functionalities for sparse matrices are implemented in this way and serve as extendable "plugins" for ADCME. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/summary.png?raw=true)


ADCME is open-sourced with an MIT license. You can find the source codes at 

[https://github.com/kailaix/ADCME.jl](https://github.com/kailaix/ADCME.jl)

Read more about methodology, philosophy, and insights about ADCME: [slides](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/Slide/ADCME.pdf?raw=true). Start with [tutorial](./tutorial.md) to solve your own inverse modeling problems!


## Installation

It is recommended to install ADCME via
```julia
using Pkg
Pkg.add("ADCME")
```

❗ For windows users, please follow the instructions [here](./windows_installation.md). 



!!! info 
    In some cases, you may want to install the package and configure the environment manually. 

    Step 1: Install `ADCME` on a computer with Internet access and zip all files from the following paths

    ```julia
    julia> using Pkg
    julia> Pkg.depots()
    ```

    The files will contain all the dependencies.

    Step 2: 
    Copy the `deps.jl` file from your built ADCME and modify it for your local repository. 

    ```julia
    using ADCME; 
    print(joinpath(splitdir(pathof(ADCME))[1], "deps/deps.jl"))
    ```
    
## Troubleshooting for MacOSX
    
Here are some common problems you may encounter on a Mac computer:

- You may get stuck at building ADCME. It is a [known issue](https://github.com/kailaix/ADCME.jl/issues/64) for some combinations of specific MacOSX and Julia versions. You can verify this by checking whether the following step gets stuck:

```julia
using PyCall
pyimport("tensorflow")
```

One workaround is to use another Julia version; for example, we tested Julia 1.3 on several MacOS versions and it works fine. 


- You may encounter the following warning when you run `using PyPlot`.

```
PyPlot is using `tkagg` backend, which is known to cause crashes on macOS (#410); use the MPLBACKEND environment variable to request a different backend.
```

To fix this problem, add the following line immediately after `using PyPlot`

```julia
using PyPlot
matplotlib.use("agg")
```

The images may not show up but you can save the figure (`savefig("filename.png")`). 


- Your Julia program may crash when you run `BFGS!` and show the following error message.  

```
Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized OMP: Hint: This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
```

This is because `matplotlib` (called by `PyPlot`) and `scipy` (called by `BFGS!`) simultaneously access OpenMP libraries in an unsafe way. To fix this problem, add the following line in the **very beginning** of your script (or run the command right after you enter a Julia prompt)

```julia
ENV["KMP_DUPLICATE_LIB_OK"] = true 
```

- You may see the following error message when you run `ADCME.precompile()`:


```
The C compiler
    "/Users/<YourUsername>/.julia/adcme/bin/clang"
is not able to compile a simple test program.
It fails with the following output: ...
```


This is because your developer tools are not the one required by `ADCME`. To solve this problem, run the following commands in your terminal:

```
rm /Users/<YourUsername>/.julia/adcme/bin/clang
rm /Users/<YourUsername>/.julia/adcme/bin/clang++
ln -s /usr/bin/clang /Users/<YourUsername>/.julia/adcme/bin/clang
ln -s /usr/bin/clang++ /Users/<YourUsername>/.julia/adcme/bin/clang++
```

Here `<YourUsername>` is your user name. 


## Optimization 

ADCME is an all-in-one solver for gradient-based optimization problems. It leverages highly optimized and concurrent/parallel kernels that are implemented in C++ for both the forward computation and gradient computation. Additionally, it provides a friendly user interface to specify the mathematical optimization problem: constructing a computational graph. 

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

The above code will return a value close to  the optimal values $\mathbf{x} = [1\ 1\ \ldots\ 1]$. 


!!! info 
    You can also use [`Optimize!`](@ref) to use other optimizers. For example, if you want to use an optimizer, such as `ConjugateGraidient` from the `Optim` package, simply replace `BFGS!` with `Optimize!` and specify the corresponding optimizer

    ```julia
    using Optim
    Optimize!(sess, loss, optimizer = ConjugateGradient())
    ```


## Machine Learning 

You can also use ADCME to do typical machine learning tasks and leverage the Julia machine learning ecosystem! Here is an example of training a ResNet for digital number recognition.

```julia
using MLDatasets
using ADCME

# load data 
train_x, train_y = MNIST.traindata()
train_x = reshape(Float64.(train_x), :, size(train_x,3))'|>Array
test_x, test_y = MNIST.testdata()
test_x = reshape(Float64.(test_x), :, size(test_x,3))'|>Array

# construct loss function 
ADCME.options.training.training = placeholder(true)
x = placeholder(rand(64, 784))
l = placeholder(rand(Int64, 64))
resnet = Resnet1D(10, num_blocks=10)
y = resnet(x)
loss = mean(sparse_softmax_cross_entropy_with_logits(labels=l, logits=y))

# train the neural network 
opt = AdamOptimizer().minimize(loss)
sess = Session(); init(sess)
for i = 1:10000
    idx = rand(1:60000, 64)
    _, loss_ = run(sess, [opt, loss], feed_dict=Dict(l=>train_y[idx], x=>train_x[idx,:]))
    @info i, loss_
end

# test 
for i = 1:10
    idx = rand(1:10000,100)
    y0 = resnet(test_x[idx,:])
    y0 = run(sess, y0, ADCME.options.training.training=>false)
    pred = [x[2]-1 for x in argmax(y0, dims=2)]
    @info "Accuracy = ", sum(pred .== test_y[idx])/100
end
```




## Contributing

Contribution and suggestions are always welcome. In addition, we are also looking for research collaborations. You can submit issues for suggestions, questions, bugs, and feature requests, or submit pull requests to contribute directly. You can also contact the authors for research collaboration. 
