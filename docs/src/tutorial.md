

# ADCME Tutorial

> The Gateway to Inverse Modeling with Physics Based Machine Learning

**Overview**

ADCME is an open-source Julia package for inverse modeling in scientific computing using automatic differentiation. The backend of ADCME is the high performance deep learning framework, TensorFlow, which provides parallel computing and automatic differentiation features based on computational graph, but  ADCME augments TensorFlow by functionalities---like sparse linear algebra---essential for scientific computing. ADCME leverages the Julia environment for maximum efficiency of computing. Additionally, the syntax of ADCME is designed from the beginning to be compatible with the Julia syntax, which is friendly for scientific computing. 

**Prerequisites**

The tutorial does not assume readers with experience in deep learning. However, basic knowledge of scientific computing in Julia is required. 

**Tutorial Series**

[What is ADCME? Computational Graph, Automatic Differentiation & TensorFlow]("What is ADCME? Computational Graph, Automatic Differentiation & TensorFlow")

ADCME Basics: Tensor, Type, Operator, Session & Kernel

Mathematical Minimization with ADCME

Sparse Linear Algebra in ADCME

Numerical Scheme in ADCME: Finite Difference Example

Numerical Scheme in ADCME: Finite Element Example

Inverse Modeling in ADCME

Neural Network Tutorial: Combining NN with Numerical Schemes 

Advanced: Automatic Differentiation for Linear Implicit Operations 

Advanced: Automatic Differentiation for Nonlinear Implicit Operators

Advanced: Custom Operators 

Advanced: Debugging 

## What is ADCME? Computational Graph, Automatic Differentiation & TensorFlow

### Computational Graph	

A computational graph is a functional description of the required computation. In the computationall graph, an edge represents a value, such as a scalar, a vector, a matrix or a tensor. A node represents a function whose input arguments are the the incoming edges and output values are are the outcoming edges. Based on the number of input arguments, a function can be nullary, unary, binary, ..., and n-ary; based on the number of output arguments, a function can be single-valued or multiple-valued. 



Computational graphs are directed and acyclic. The acyclicity implies the forward propagation computation is well-defined: we loop over edges in topological order and evaluates the outcoming edges for each node. To make the discussion more concrete, we illustrate the computational graph for 

$$z = \sin(x_1+x_2) + x_2^2 x_3$$

![](./assets/fd.jpeg)



There are in general two programmatic ways to construct computational graphs: static and dynamic declaration. In the static declaration, the computational graph is first constructed symbolically, i.e., no actual numerical arithmetic are executed. Then a bunch of data is fed to the graph for the actual computation. An advantage of static declarations is that they allow for graph optimization such as removing unused branches. Additionally, the dependencies can be analyzed for parallel execution of independent components. Another approach is the dynamic declaration, where the computational graph is constructed on-the-fly as the forward computation is executed. The dynamic declaration interleaves construction and evaluation of the graph, making software development more intuitive. 

### Automatic Differentiation

(This section can be skipped in the first reading.)

An important application of computational graphs is automatic differentiation (AD). In general, there are three modes of AD: reverse-mode, forward-mode, and mixed mode. In this tutorial, we focus on the reverse-mode, which computes the gradients with respect to independent variables by backward propagation, i.e., loop over the edges in reverse topological order starting with a final goal edge. 



To explain how reverse-mode AD works, let's consider constructing a computational graph with independent variables 

$$\{x_1, x_2, \ldots, x_n\}$$

and the forward propagation produces a single output $x_N$, $N>n$. The gradients $\frac{\partial x_N(x_1, x_2, \ldots, x_n)}{\partial x_i}$
$i=1$, $2$, $\ldots$, $n$ are queried. 

The idea is that this algorithm can be decomposed into a sequence of functions $f_i$ ($i=n+1, n+2, \ldots, N$) that can be easily differentiated analytically, such as addition, multiplication, or basic functions like exponential, logarithm and trigonometric functions. Mathematically, we can formulate it as

$$\begin{aligned}
    x_{n+1} &= f_{n+1}(\mathbf{x}_{\pi({n+1})})\\
    x_{n+2} &= f_{n+2}(\mathbf{x}_{\pi({n+2})})\\
    \ldots\\
    x_{N} &= f_{N}(\mathbf{x}_{\pi({N})})\\
\end{aligned}$$

where $\mathbf{x} = \{x_i\}_{i=1}^N$ and $\pi(i)$ are the parents of $x_i$, s.t., $\pi(i) \in \{1,2,\ldots,i-1\}$.



![](./assets/cg.jpeg)

The idea to compute $\partial x_N / \partial x_i$ is to start from $i = N$, and establish recurrences to calculate derivatives with respect to $x_i$ in terms of derivatives with respect to $x_j$, $j >i$. To define these recurrences rigorously, we need to define different functions that differ by the choice of independent variables.

The starting point is to define $x_i$ considering all previous $x_j$, $j < i$, as independent variables. Then:

$$x_i(x_1, x_2, \ldots, x_{i-1}) = f_i(\mathbf{x}_{\pi(i)})$$

Next, we observe that $x_{i-1}$ is a function of previous $x_j$, $j < i-1$, and so on; so that we can recursively define $x_i$ in terms of fewer independent variables, say in terms of $x_1$, ..., $x_k$, with $k < i-1$. This is done recursively using the following definition:

$$x_i(x_1, x_2, \ldots, x_j) = x_i(x_1, x_2, \ldots, x_j, f_{j+1}(\mathbf{x}_{\pi(j+1)})), \quad n < j+1 < i$$

Observe that the function of the left-hand side has $j$ arguments, while the function on the right has $j+1$ arguments. This equation is used to "reduce" the number of arguments in $x_i$.

With these definitions, we can define recurrences for our partial derivatives which form the basis of the back-propagation algorithm. The partial derivatives for

$$x_N(x_1, x_2, \ldots, x_{N-1})$$

are readily available since we can differentiate

$$f_N(\mathbf{x}_{\pi(N)})$$

directly. The problem is therefore to calculate partial derivatives for functions of the type $x_N(x_1, x_2, \ldots, x_i)$ with $i<N-1$. This is done using the following recurrence:

$$\frac{\partial x_N(x_1, x_2, \ldots, x_{i})}{\partial x_i} = \sum_{j\,:\,i\in \pi(j)}
    \frac{\partial x_N(x_1, x_2, \ldots, x_j)}{\partial x_j}
    \frac{\partial x_j(x_1, x_2, \ldots, x_{j-1})}{\partial x_i}$$

with $n < i< N-1$. Since $i \in \pi(j)$, we have $i < j$. So we are defining derivatives with respect to $x_i$ in terms of derivatives with respect to $x_j$ with $j > i$. The last term

$$\frac{\partial x_j(x_1, x_2, \ldots, x_{j-1})}{\partial x_k}$$

is readily available since:

$$x_j(x_1, x_2, \ldots, x_{j-1}) = f_j(\mathbf{x}_{\pi(j)})$$

![](./assets/cg2.jpeg)

The computational cost of this recurrence is proportional to the number of edges in the computational graph (excluding the nodes $1$ through $n$), assuming that the cost of differentiating $f_k$ is $O(1)$. The last step is defining

$$\frac{\partial x_N(x_1, x_2, \ldots, x_n)}{\partial x_i} = \sum_{j\,:\,i\in \pi(j)}
    \frac{\partial x_N(x_1, x_2, \ldots, x_j)}{\partial x_j}
    \frac{\partial x_j(x_1, x_2, \ldots, x_{j-1})}{\partial x_i}$$

with $1 \le i \le n$. Since $n < j$, the first term

$$\frac{\partial x_N(x_1, x_2, \ldots, x_j)}{\partial x_j}$$

has already been computed in earlier steps of the algorithm. The computational cost is equal to the number of edges connected to one of the nodes in $\{1, \dots, n\}$.

We can see that the complexity of the back-propagation is bounded by that of the forward step, up to a constant factor. Reverse mode differentiation is very useful in the penalty method, where the loss function is a scalar, and no other constraints are present. 

As a concrete example, we consider the example of evaluating $\frac{dz(x_1,x_2,x_3)}{dx_i}$, where $z = \sin(x_1+x_2) + x_2^2x_3$. The gradients are  backward propagated exactly in the reverse order of the forward propagation. 

![](./assets/bd.jpeg)

### TensorFlow

Google's TensorFlow provides a convenient way to specify the computational graph statically. TensorFlow  has automatic differentiation features and its performance is optimized for large-scale computing. ADCME is built on TensorFlow by overloading numerical operators and augmenting TensorFlow with essential scientific computing functionalities. We contrast the TensorFlow implementation with the ADCME implementation of computing the objective function and its gradient in the following example.

$$y(x) = \|(AA^T+xI)^{-1}b-c\|^2, \; z = y'(x)$$ 

where $A\in \mathbb{R}^{n\times n}$ is a random matrix, $x,b,c$ are scalars, and $n=10$.

**TensorFlow Implementation**

```python
import tensorflow as tf
import numpy as np 
A = tf.constant(np.random.rand(10,10), dtype=tf.float64)
x = tf.constant(1.0, dtype=tf.float64)
b = tf.constant(np.random.rand(10), dtype=tf.float64)
c = tf.constant(np.random.rand(10), dtype=tf.float64)
B = tf.matmul(A, tf.transpose(A)) + x * tf.constant(np.identity(10))
y = tf.reduce_sum((tf.squeeze(tf.matrix_solve(B, tf.reshape(b, (-1,1))))-c)**2)
z = tf.gradients(y, x)[0]
sess = tf.Session()
sess.run([y, z])
```

**Julia Implementation**

```julia
using ADCME, LinearAlgebra
A = constant(rand(10,10))
x = constant(1.0)
b = rand(10)
c = rand(10)
y = sum(((A*A'+x*diagm(0=>ones(10)))\b - c)^2)
z = gradients(y, x)
sess = Session()
run(sess, [y,z])
```



### Summary

The computational graph and automatic differentiation are the core concepts underlying ADCME. TensorFlow works as the workhorse for optimion and execution of the computational graph in a high performance environment. 

To construct a computational graph for a Julia program, ADCME overloads most numerical operators like `+`, `-`, `*`, `/` and matrix multiplication in Julia by the corresponding TensorFlow operators. Therefore, you will find many similar workflows and concepts as TensorFlow, such as `constant`, `Variable`, `session`, etc. However, not all operators relevant to scientific computing in Julia have its counterparts in TensorFlow. To that end, custom kernels are implemented to supplement TensorFlow, such as sparse linear algebra related functions. 

ADCME aims at providing a easy-to-use, flexible,  and high performance interface to do data processing, implement numerical schemes, and conduct mathematical optimization. It is built not only for academic interest but also for real-life large-scale simulations. 

Like TensorFlow, ADCME works in sessions, in which each session consumes a computational graph. Usually the workflow is split into three steps:

1. Define independent variables. `constant` for tensors that do not require gradients and `Variable` for those requiring gradients. 

   ```julia
   a = constant(0.0)
   ```

2. Construct the computational graph by defining the computation

   ```julia
   L = (a-1)^2
   ```

3. Create a session and run the computational graph

   ```julia
   sess = Session()
   run(sess, L)
   ```

   



## ADCME Basics: Tensor, Type, Operator, Session & Kernel

### Tensors and Operators

`Tensor` is a data structure for storing structured data, such as a scalar, a vector, a matrix or a high dimensional tensor. The name of the ADCME backend, `TensorFlow`, is also derived from its core framework, `Tensor`. Tensors can be viewed as symbolic versions of Julia's `Array`. 



A tensor is a collection of $n$-dimensional arrays. ADCME represents tensors using a `PyObject` handle to the TensorFlow `Tensor` data structure. A tensor has three important properties

- `name`: Each Tensor admits a unique name. 
- `shape`: For scalars, the shape is always an empty tuple `()`; for $n$-dimensional vectors, the shape is `(n,)`; for matrices or higher order tensors, the shape has the form `(n1, n2, ...)`
- `dtype`: The type of the tensors. There is a one-to-one correspondence between most TensorFlow types and Julia types (e.g., `Int64`, `Int32`, `Float64`, `Float32`, `String`, and `Bool`). Therefore, we have overloaded the type name so users have a unified interface. 

![](./assets/tensorspec.png)

An important difference is that `tensor` object stores data in the row-major while Julia's default for `Array` is column major. The difference may affect performance if not carefully dealt with, but more often than not, the difference is not relevant if you do not convert data between Julia and Python often. Here is a representation of ADCME `tensor`

![](./assets/tensors.png)

There are 4 ways to create tensors. 

- [`constant`](@ref). As the name suggests, `constant` creates an immutable tensor from Julia Arrays. 

```julia
constant(1.0)
constant(rand(10))
constant(rand(10,10))
```

- [`Variable`](@ref). In contrast to `constant`, `Variable` creates tensors that are mutable. The mutability allows us to update the tensor values, e.g., in an optimization procedure. It is very important to understand the difference between `constant` and `Variable`: simply put, in inverse modeling, tensors that are defined as `Variable` should be the quantity you want to invert, while `constant` is a way to provide known data.

```julia
Variable(1.0)
Variable(rand(10))
Variable(rand(10,10))
```

- [`placeholder`](@ref). `placeholder` is a convenient way to specify a tensor whose values are to be provided in the runtime. One use case is that you want to try out different values for this tensor and scrutinize the simulation result. 

```julia
placeholder(Float64, shape=[10,10])
placeholder(rand(10)) # default value is `rand(10)`
```

- [`SparseTensor`](@ref). `SparseTensor` is a special data structure to store a sparse matrix. Although it is not very emphasized in machine learning, sparse linear algebra is one of the cores to scientific computing. Thus possessing a strong sparse linear algebra support is the key to success inverse modeling with physics based machine learning. 

```julia
using SparseArrays
SparseTensor(sprand(10,10,0.3))
SparseTensor([1,2,3],[2,2,2],[0.1,0.3,0.5],3,3) # specify row, col, value, number of rows, number of columns
```



Now we know how to create tensors, the next step is to perform mathematical operations on those tensors.

`Operator` can be viewed as a function that takes multiple tensors and outputs multiple tensors. In the computational graph, operators are represented by nodes while tensors are represented by edges. Most mathematical operators, such as `+`, `-`, `*` and `/`, and matrix operators, such as matrix-matrix multiplication, indexing and linear system solve, also work on tensors. 

```julia
a = constant(rand(10,10))
b = constant(rand(10))
a + 1.0 # add 1 to every entry in `a`
a * b # matrix vector production
a * a # matrix matrix production
a .* a # element wise production
inv(a) # matrix inversion
```

### Session

With the aforementioned syntax to create and transform tensors, we have created a computational graph. However, at this point, all the operations are symbolic, i.e., the operators have not been executed yet. 

To trigger the actual computing, the TensorFlow mechanism is to create a session, which drives the graph based optimization (like detecting dependencies) and executes all the operations.  

![](./assets/session.gif)

```julia
a = constant(rand(10,10))
b = constant(rand(10))
c = a * b
sess = Session()
run(sess, c) # syntax for triggering the execution of the graph
```

If your computational graph contains `Variables`, which can be listed via [`get_collection`](@ref), then you must initialize your graph before any `run` command, in which the Variables are populated with initial values

```julia
init(sess)
```

### Kernel

The kernels provide the low level C++ implementation for the operators. ADCME augments users with missing features in TensorFlow that are crucial for scientific computing and tailors the syntax for numerical schemes. Those kernels, depending on their implementation, can be used in CPU, GPU, TPU or heterogenious computing environments. 

All the intensive computations are  done either in Julia or C++, and therefore we can achieve very high performance if the logic is done appropriately. For performance critical part, users may resort to custom kernels using [`custom_op`](@ref), which allows you to incooperate custom designed C++ codes. 

![](./assets/kernel.png)



### Summary

ADCME performances operations on tensors. The actual computations are pushed back to low level C++ kernels via operators. A session is need to drive the executation of the computation. It will be easier for you to analyze computational cost and optimize your codes with this computation model in mind. 





## Numerical Scheme in ADCME: Finite Difference Example

ADCME provides convenient tools to implement numerical schemes. In this tutorial, we will implement a finite difference program and conduct inverse modeling. In the first part, we consider a toy example of estimating parameters in a partial differential equation. In the second part, we showcase a real world application of ADCME to geophysical inversion.

###  Estimating a scalar unknown in the PDE

Consider the following partial differential equation
```math
-bu''(x)+u(x)=f(x)\quad x\in[0,1], u(0)=u(1)=0
```
where 
```math
f(x) = 8 + 4x - 4x^2
```
Assume that we have observed $u(0.5)=1$, we want to estimate $b$. The true value in this case should be $b=1$. We can discretize the system using finite difference method, and the resultant linear system will be
```math
(bA+I)\mathbf{u} = \mathbf{f}
```
where
```math
A = \begin{bmatrix}
        \frac{2}{h^2} & -\frac{1}{h^2} & \dots & 0\\
         -\frac{1}{h^2} & \frac{2}{h^2} & \dots & 0\\
         \dots \\
         0 & 0 & \dots & \frac{2}{h^2}
    \end{bmatrix}, \quad \mathbf{u} = \begin{bmatrix}
        u_2\\
        u_3\\
        \vdots\\
        u_{n}
    \end{bmatrix}, \quad \mathbf{f} = \begin{bmatrix}
        f(x_2)\\
        f(x_3)\\
        \vdots\\
        f(x_{n})
    \end{bmatrix}
```

The idea for implementing the inverse modeling method in ADCME is that we make the unknown $b$ a `Variable` and then solve the forward problem pretending $b$ is known. The following code snippet shows the implementation
```julia
using LinearAlgebra
using ADCME             # (1)

n = 101 # number of grid nodes in [0,1]
h = 1/(n-1)
x = LinRange(0,1,n)[2:end-1]       # (2)

b = Variable(10.0) # we use Variable keyword to mark the unknowns    # (3)
A = diagm(0=>2/h^2*ones(n-2), -1=>-1/h^2*ones(n-3), 1=>-1/h^2*ones(n-3)) 
B = b*A + I  # I stands for the identity matrix
f = @. 4*(2 + x - x^2) 
u = B\f # solve the equation using built-in linear solver
ue = u[div(n+1,2)] # extract values at x=0.5								# (4)

loss = (ue-1.0)^2    # (5)

# Optimization
sess = Session(); init(sess) # (6) 
BFGS!(sess, loss)			# (7)

println("Estimated b = ", run(sess, b)) 
```
The expected output is
```
Estimated b = 0.9995582304494237
```

The detailed explaination is as follow: (1) The first two lines load necessary packages; (2) We split the interval $[0,1]$ into $100$ equal length subintervals; (3) Since $b$ is unknown and needs to be updated during optimization, we mark it as trainable using the `Variable` keyword; (4) Solve the linear system and extract the value at $x=0.5$. here `I` stands for the identity matrix and `@.` denotes element-wise operation. They are `Julia`-style syntax but are also compatible with tensors by overloading; (5) Formulate the loss function; (6) Create and initialize a `TensorFlow` session, which analyzes the computational graph and initializes the tensor values; (7) Finally, we trigger the optimization by invoking `BFGS!`, which wraps the `L-BFGS-B` algorithm. 



### ADSeismic.jl: A General Approach to Seismic Inversion

ADSeismic is a software package for solving seismic inversion problems, such as velocity model estimation, rupture imaging, earthquake location, and source time function retrieval. The governing equation for the acoustic wave equation is  

 $$ \frac{\partial^2 u}{\partial t^2} = \nabla\cdot(c^2 \nabla u) +  f$$

where $u$ is displacement, $f$ is the source term, and $c$ is the spatially varying acoustic velocity. The inversion parameters of interest are $c$ or $f$.
The governing equation for the elastic wave equation is 

$$\begin{aligned}
    \rho \frac{\partial v_i}{\partial t} &= \sigma_{ij, j} + \rho f_i \\ 
    \frac{\partial \sigma_{ij}}{\partial t} &= \lambda v_{k,k} + \mu(v_{i,j} + v_{j,i})
\end{aligned}$$



where $v$ is velocity, $\sigma$ is stress tensor, $\rho$ is density, and $\lambda$ and $\mu$ are the Lam\'e's constants. The inversion parameters in the elastic wave equation case are $\lambda$, $\mu$, $\rho$ or $f$.

The idea is to substitute the unknowns such as the $c$ and $f$ using mutable tensors (with the `Variable` keyword) and implement the finite difference method. The implementation detail is beyond the scope of this tutorial. Basically, when explicit schemes are used, the finite difference scheme can be expressed by a computational graph as follows, where $U$ is the discretization of $u$, $A(\theta)$ is the fintie difference coefficient matrix and $\theta$ is the unknown (in this case, the entries in the coefficient matrix depends on $\theta$ ). The loss function is formulated by matching the predicted wavefield $U_i$ and the observed wavefield $U_i^{\mathrm{obs}}$. 



![adg](./assets/adg.png)

The unknown $\theta$ is sought by solving a minimization problem using L-BFGS-B, using gradients computed in AD. Besides the simplification of implementation, a direct benefit of implementing the numerical in ADCME is that we can leverage multi-GPU computing resources. We distribute the loss function for each scenario (in practice, we can collect many $\{U_i^{\mathbf{obs}}\}$ corresponding to different source functions $f$) onto different GPUs and compute the gradients separately. Using this strategy, we can achieve more than 20 times and 60 times acceleration for acoustic and elastic wave equations respectively.

![](./assets/gpu.png)

Here we show a case in locating the centroid of an earthquake. The red star denotes the location where the earthquake happens and the triangles denote the seismic stations. The subsurface constitutes layers of different properties (the values of $c$ are different), affecting the propagation of the seismic waves. 

| Source/Receiver Location                                     | Forward Simulation                                           |

| ------------------------------------------------------------ | ------------------------------------------------------------ |

|![](./assets/source-vp.png) | ![](./assets/forward_source.gif) |

By running the optimization problem by specifying the earthquake location as `Variable` [^delta], we can locate the centroid of an earthquake. The result is amazingly good. It is worth noting that it requires substantial effort to implement the traditional adjoint-state solver for this problem (e.g., it takes time to manually deriving and implementing the gradients). However, in view of ADCME, the inversion functionality is merely a by-product of the forward simulation codes, which can be reused in many other inversion problems.

![fwi_source](./assets/fwi_source.gif)

[^delta]: Mathematically, $f(t, \mathbf{x})$ is a Delta function in $\mathbf{x}$; to make the inversion problem continuous, we use $f_{\theta}(t, \mathbf{x}) = g(t) \frac{1}{2\pi\sigma^2}\exp(-\frac{\|\mathbf{x}-\theta\|^2}{2\sigma^2})$ to approximate $f(t, \mathbf{x})$; here $\theta\in\mathbb{R}^2$ and $g(t)$ are unknown.

