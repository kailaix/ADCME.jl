# Distributed Scientific Machine Learning using MPI 


Many large-scale scientific computing involves parallel computing. Among many parallel computing models, the MPI  is one of the most popular models. In this section, we describe how ADCME can work with MPI for solving inverse modeling. Specifically, we describe how gradients can be back-propagated via MPI function calls.  

!!! info 
    Message Passing Interface (MPI) is an interface for parallel computing based on message passing models. In the message passing model, a master process assigns work to workers by passing them a message that describes the work. The message may be data or meta information (e.g., operations to perform). A consensus was reached around 1992 and the MPI standard was born. MPI is a definition of interface, and the implementations are left to hardware venders. 

## MPI Support in ADCME

The ADCME solution to distributed computing for scientific machine learning is to provide a set of "data communication" nodes in the computational graph. Each machine (MPI processor) runs an identical computational graph. The computational nodes are executed independently on each processor, and the data communication nodes need to synchronize among different processors. 

These data communication nodes are implemented using MPI APIs. They are not necessarily blocking operations, but because ADCME respects the data dependency of computation, they act like blocking operations and the child operators are executed only when data communication is finished. For example, in the following example,
```julia
b = mpi_op(a)
c = custom_op(b)
```
even though `mpi_op` and `custom_op` can overlap, ADCME still sequentially execute these two operations. 

This blocking behavior simplifies the synchronization logic as well as the implementation of gradient back-propagation while harming little performance. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi1.PNG?raw=true)

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi2.PNG?raw=true)

ADCME provides a set of commonly used MPI operators. See [MPI Operators](https://kailaix.github.io/ADCME.jl/dev/api/#MPI). Basically, they are

* [`mpi_init`](@ref), [`mpi_finalize`](@ref): Initialize and finalize MPI session. 
* [`mpi_rank`](@ref), [`mpi_size`](@ref): Get the MPI rank and size.
* [`mpi_sum`](@ref), [`mpi_bcast`](@ref): Sum and broadcast tensors in different processors. 
* [`mpi_send`](@ref), [`mpi_recv`](@ref): Send and receive operators. 

The above two set of operators support automatic differentiation. They were implemented with MPI adjoint methods, which have existed in academia for decades. 

## Limitations 

Despite that the provided `mpi_*` operations meet most needs,  some sophisticated data communication operations may not be easily expressed using these APIs. For example, when solving the Poisson's equation on a uniform grid, we may decompose the domain into many squares, and two adjacent squares exchange data in each iteration. A sequence of `mpi_send`, `mpi_recv` will likely cause deadlock. 

Just like when it is difficult to use automatic differentiation to implement a forward computation and its gradient back-propagation, we resort to custom operators, it is the same case for MPI. We can design a specialized custom operator for data communication. To resolve the deadlock problem, we found the asynchronous sending, followed by asynchronous receiving, and then followed by waiting, a very general and convenient way to implement custom operators. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpiadjoint.png?raw=true)

Another limitation for ADCME is that currently, for performance, we need to set the total number of threads per MPI process to be 1
```julia
config = tf.ConfigProto(inter_op_parallelism_threads=1)
sess = Session(config = config)
```
Otherwise, there will be significant cost for synchronization between different threads for the MPI operation kernel. Setting `inter_op_parallelism_threads=1` limits ADCME to execute one kernel at a time, although each kernel can still utilize multiple threads. One solution is to use a specialized executed policy for MPI kernels---they should always executed by one thread. Nevertheless, the optimal policy is application dependent and the current model fits a broad range of applications. 




## Implementing Custom Operators using MPI

We can also make custom operators with MPI. Let us consider computing

$$f(\theta) = \sum_{i=1}^n f_i(\theta)$$

Each $f_i$ is a very expensive function so it makes sense to use MPI to split the jobs on different processors. To simplify the problem, we consider 

$$f(\theta) = f_1(\theta) + f_2(\theta) + f_3(\theta) + f_4(\theta)$$

where $f_i(\theta) = \theta^{i-1}$. 


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi.png?raw=true)


Using the ADCME MPI API, we have the following code (`test_simple.jl`)
```julia
using ADCME

mpi_init()
θ = placeholder(1.0)
fθ = mpi_bcast(θ)
l = fθ^mpi_rank()
L = mpi_sum(l)
g = gradients(L, θ)
sess = Session(); init(sess)
L_, g_ = run(sess, [L, g])

if mpi_rank()==0
    @info L_, g_ 
end

mpi_finalize()
```

We run the program with 4 processors

```bash
mpirun -n 4 julia test_simple.jl
```


## Optimization 

For solving inverse problems using distributed computing, an MPI-capable optimizer is required. The ADCME solution to distributed optimization is that the master machine holds, distributes and updates the optimizable variables. The gradients are calculated in the same device where the corresponding forward computation is done. Therefore, for a given serial optimizer, we can refactor it to a distributed one by letting worker nodes wait for instructions from the master node to compute either the objective function or the gradient.

This idea is implemented in the [`ADOPT.jl`](https://github.com/kailaix/ADOPT.jl) package, a customized version of [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). 


In the following, we try to solve 

$$1+\theta +\theta^2+\theta^3 = 2$$

using MPI-enabled LBFGS optimizer. 

```julia
using ADCME

mpi_init()
θ = placeholder(ones(1))
fθ = mpi_bcast(θ)
l = fθ^mpi_rank()
L = (sum(mpi_sum(l))-2)^2
g = gradients(L, fθ)
sess = Session(); init(sess)

f = x->run(sess, L, θ=>x)
g!= (G, x)->(G[:] = run(sess, g, θ=>x))
results = mpi_optimize(f, g!, run(sess, θ), ADOPT.LBFGS())
if mpi_rank()==0
    @info results.minimizer, results.minimum 
end
mpi_finalize()
```

## Configure MPI for ADCME



To enable MPI in ADCME, you need to build ADCME with the following environment variable:

* `MPI_C_LIBRARIES`: the MPI shared library, for example, `C:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Lib\\x64\\msmpi.lib`.
* `MPI_INCLUDE_PATH`: the directory where `mpi.h` is located, for example, `C:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Include`.

You can either add these environment variables to the system path (e.g., via `export` in Linux), or in Julia

```julia-repl
julia> ENV["MPI_C_LIBRARIES"] = ...
julia> ENV["MPI_INCLUDE_PATH"] = ...
pkg> build ADCME
```

Once ADCME is successfully built with these environment variables, you will be able to use ADCME MPI features. The following are examples of using ADCME APIs



### Reduce Sum

```julia
using ADCME

mpi_init()
r = mpi_rank()
a = constant(Float64.(Array(1:10) * r))
b = mpi_sum(a)

L = sum(b)
g = gradients(L, a)
sess = Session(); init(sess)
v, G = run(sess, [b,g])
```

### Broadcast

```julia
using ADCME

mpi_init()
r = mpi_rank()
a = constant(ones(10) * r)
b = mpi_bcast(a, 3)
L = sum(b^2)
L = mpi_sum(L)
g = gradients(L, a)

sess = Session(); init(sess)
v, G = run(sess, [b, G])
```

### Send and Receive
```julia
# mpiexec.exe -n 4 julia .\mpisum.jl
using ADCME

mpi_init()
r = mpi_rank()
a = constant(ones(10) * r)
a = mpi_sendrecv(a, 0, 2)

L = sum(a^2)
g = gradients(L, a)

sess = Session(); init(sess)
v, G = run(sess, [a,g])
```

[`mpi_sendrecv`](@ref) is a lightweight wrapper for [`mpi_send`](@ref) followed by [`mpi_recv`](@ref). Equivalently, we have

```julia
if r==2
    global a
    a = mpi_send(a, 0)
end
if r==0
    global a
    a = mpi_recv(a,2)
end
```

## Solving the Poisson Equation

