
# Numerical Scheme in ADCME: Finite Element Example

The purpose of this tutorial is to show how to work with the finite element method (FEM) in ADCME. The tutorial is divided into two part. In the first part, we implement a finite element code for 1D Poisson equation using ADCME without custom operators. In the first part, you will understand how [`while_loop`](@ref) can help avoid creating a computational graph for each element. This is important because for many applications the number of elements in FEM can be enormous. The goal of the second part is to introduce [`customop`](@ref) for FEM. For performance critical applications, you may want to code your own loop over elements. However, in this case, you are responsible to calculate the sensititity of your finite element sensitivity matrix. 

## Why do you need while loop?

In engineering, we usually need to do for loops, e.g., time stepping, finite element matrix assembling, etc. In pseudocode, we have

```julia
x = constant(0.0)
for i = 1:10000
  global x
	x = x + i 
end
```

To do automatic differentiation in ADCME, direct implemnetation in the above way incurs creation of 10000 subgraphs, which requires large memories and long dependency parsing time. 

Instead of relying on programming languages for the dynamic control flow, `TensorFlow` embeds control-flow as operations *inside* the dataflow graph. This is done via `while_loop`, which ADCME inherents from `TensorFlow`. `while_loop` allows for easier graph-based optimization, and reduces time and memory for the computational graph.

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/while_loop_graph.png?raw=true)

Using `while_loop`, the same function can be implemented as follows,

```julia
function func(i, ta)
  xold = read(ta, i)
  x = xold + cast(Float64, i)
  ta = write(ta, i+1, x)
  return i+1, ta
end
i = constant(1, dtype = Int32)
ta = TensorArray(10001)
ta = write(ta, 1, constant(0.0))
_, out = while_loop((i, x)->i<=10000, func, [i, ta])
result = stack(out)
sess = Session()
run(sess,result)
```

Here `TensorArray(10001)` can be viewed as a container, which holds 10001 elements. These elements are tensors and can have sequential dependencies. One restriction on `TensorArray` is that all its elements must have the same type and size. This restriction requires us to "initialize" the `TensorArray` outside `while_loop`. The initialization is done by writting the first entry of `TensorArray` with a tensor (or a Julia numerical value/array), i.e., `ta=write(ta, 1, constant(0.0))`. A second important note is that if we want to have an loop index, `i`, the type of the loop index must be `Int32`. 

The syntax for `while_loop` is 

```julia
i, ta1, ta2, ... = while_loop(condition, body, [i, ta1, ta2, ...])
```

where `ta1`, `ta1`, ..., are different `TensorArrays`.

Inside a `body` function, we can use `read(ta, i)` to read the `i`-th value of the `TensorArray`. Note that the value must exist (written at an earlier time). After performing necessary computation, the result `v` is written the `i+1`-th index of `TensorArray` via `ta = write(ta, i+1, v)`. Note `write` is **not** an in-place function, so you need to update `ta` with its return value. Finally, the input and output of the `body` function must be consistent. 

The `condition` function is a way to specify the stop criterion. It takes inputs `[i, ta1, ta2, ...]` and outputs a **tensor boolean**. For example, `i<10000` is valid because `i` is a tensor, and `i<10000` is interpreted as a tensor operation. 

Finally, to convert `TensorArray` to normal tensors, we can use the function `stack`, which converts a `TensorArray` to a tensor. The first dimension of the converted tensor will be `?`. This is because without actually executing the computational graph, we never know the true size of `TensorArray`. For example, the stop criterio may be reached before the preassigned size. If you need to have a concrete shape of the tensor, you can use [`set_shape`](@ref) or [`reshape`](@ref) to reshape the converted tensor. 




## 1D Example

As a simple example, we consider assemble the external load vector for linear finite elements in 1D. Assume that the load distribution is $f(x)=1-x^2$, $x\in[0,1]$. The goal is to compute a vector $\mathbf{v}$ with $v_i=\int_{0}^1 f(x)\phi_i(x)dx$, where $\phi_i(x)$ is the $i$-th linear element. 

The pseudocode for this problem is shown in the following

```pseudocode
F = zeros(ne+1) // ne is the total number of elements
for e = 1:ne
  add load contribution to F[e] and F[e+1]
end
```

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/externalforce.png?raw=true)

However, if `ne` is very large, writing explicit loops is unwise since it will create `ne` subgraphs. `while_loop` can be very helpful in this case

```julia
using ADCME

ne = 100
h = 1/ne
f = x->1-x^2
function cond0(i, F_arr)
    i<=ne+1
end
function body(i, F_arr)
    fmid = f(cast(i-2, Float64)*h+h/2)
    F = vector([i-1;i], [fmid*h/2;fmid*h/2], ne+1)      # (1)
    F_arr = write(F_arr, i, F)
    i+1, F_arr
end

F_arr = TensorArray(ne+1)
F_arr = write(F_arr, 1, constant(zeros(ne+1))) # (2)
i = constant(2, dtype=Int32)
_, out = while_loop(cond0, body, [i,F_arr]; parallel_iterations=10)
F = sum(stack(out), dims=1)  # (3)
sess = Session(); init(sess)
F0 = run(sess, F)
```

Detailed explaination: (1) [`vector(idx, val, len)`](@ref) creates a length `len` vector with only the indices `idx` nonzero, populated with values `val`, i.e., `v[idx] = val`; (2) it is important to populate the first entry in a TensorArray, partially because of the need to inform `F_arr` of the data type; (3) [`stack`](@ref) extracts the output `out` as a tensor.  

## 2D Example

In this section, we demonstrate how to assemble a finite element matrix based on `while_loop` for a 2D Poisson problem. We consider the following problem
```math
\begin{aligned}
\nabla \cdot ( D\nabla u(\mathbf{x}) ) &= f(\mathbf{x})& \mathbf{x}\in \Omega\\
u(\mathbf{x}) &= 0 & \mathbf{x}\in \partial \Omega
\end{aligned}
```
Here $\Omega$ is the unit disk. We consider a simple case, where
```math
\begin{aligned}
D&=\mathbf{I}\\
f(\mathbf{x})&=-4
\end{aligned}
```
Then the exact solution will be 
```math
u(\mathbf{x}) = 1-x^2-y^2
```
The weak formulation is
```math
\langle \nabla v(\mathbf{x}), D\nabla u(\mathbf{x}) \rangle = \langle f(\mathbf{x}),v(\mathbf{x}) \rangle
```
We  split $\Omega$ into triangles $\mathcal{T}$ and use piecewise linear basis functions. Typically, we would iterate over all elements and compute the local stiffness matrix for each element. However, this could result in a large loop if we use a fine mesh. Instead, we can use `while_loop` to complete the task. 

The implementation is split into two parts: 

- The first part is associated with data preprocessing such as precompute finite element data. The quantities in this part do not require gradients and therefore can leverage the full performance of Julia. 
- The second part is accociated with finite element. Particularly, the quantity of interest is $D$, which we may want to estimate from data in the future. 

```julia
using ADCME, LinearAlgebra, PyCall
using DelimitedFiles
using PyPlot

# read data 
elem = readdlm("meshdata/elem.txt", Int64)
node = readdlm("meshdata/nodes.txt")
dof = readdlm("meshdata/dof.txt", Int64)[:]
elem_ = constant(elem)
ne = size(elem,1)
nv = size(node, 1)

# precompute 
localcoef = zeros(ne, 3, 3)
areas = zeros(ne)
for e = 1:ne 
    el = elem[e,:]
    x1, y1 = node[el[1],:]
    x2, y2 = node[el[2],:]
    x3, y3 = node[el[3],:]
    A = [x1 y1 1.0; x2 y2 1.0; x3 y3 1.0]
    localcoef[e,:,:] = inv(A)
    areas[e] = 0.5*abs(det(A))
end

# compute right hand side using midpoint rule 
rhs = zeros(nv)
for i = 1:ne
    el = elem[i,:]
    rhs[el] .+= 4*areas[i]/3
end

areas = constant(areas)
localcoef = constant(localcoef)
D = constant(diagm(0=>ones(2)))
function body(i, tai, taj, tav)
    el = elem_[i-1]
    a = areas[i-1]
    L = localcoef[i-1]
    LocalStiff = Array{PyObject}(undef, 3, 3)
    for i = 1:3
        for j = 1:3
            LocalStiff[i,j] = a*[L[1,i] L[2,i]]*D*[L[1,j];L[2,j]]|>squeeze
        end
    end
    ii = reshape([el el el], (-1,))
    jj = reshape([el;el;el], (-1,))
    tai = write(tai, i, ii)
    taj = write(taj, i, jj)
    # op = tf.print(el)
    # i = bind(i, op)
    tav = write(tav, i, vcat(LocalStiff[:]...))
    return i+1, tai, taj, tav 
end

i = constant(2, dtype=Int32)
tai = TensorArray(ne+1, dtype=Int64)
taj = TensorArray(ne+1, dtype=Int64)
tav = TensorArray(ne+1)
tai = write(tai, 1, constant(ones(Int64,9)))
taj = write(taj, 1, constant(ones(Int64,9)))
tav = write(tav, 1, constant(zeros(9)))
_, ii, jj, vv = while_loop((i, tas...)->i<=ne+1, body, [i, tai, taj, tav])
ii = reshape(stack(ii),(-1,)); jj = reshape(stack(jj),(-1,)); vv = reshape(stack(vv),(-1,))

A = SparseTensor(ii, jj, vv, nv, nv) # (1)

ndof = [x for x in setdiff(Set(1:nv), Set(dof))]
A = scatter_update(A, dof, ndof, spzero(length(dof), length(ndof)))  # (2)
A = scatter_update(A, ndof, dof, spzero(length(ndof), length(dof)))
A = scatter_update(A, dof, dof, spdiag(length(dof)))
rhs[dof] .= 0.0
sol = A\rhs  # (3)

sess = Session(); init(sess)
S = run(sess, sol)
close("all")
scatter3D(node[:,1], node[:,2], S, marker="^", label = "FEM")
scatter3D(node[:,1], node[:,2], (@. 1-node[:,1]^2-node[:,2]^2), marker = "+", label = "Exact")
legend()
```

The implementation in the `while_loop` part is a standard routine in FEM. Other detailed explaination: (1) We use [`SparseTensor`](@ref) to create a sparse matrix out of the row indices, column indices and values. (2) [`scatter_update`](@ref) sets part of the sparse matrix to a given one. [`spzero`](@ref) and [`spdiag`](@ref) are convenient ways to specify zero and identity sparse matrices. (3) The backslash operator will invoke a sparse solver (the default is SparseLU). 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/while_loop.png?raw=true)

## Sensitivity 

The gradients with respect to the parameters in the finite element coefficient matrix, also known as the **sensitivity**, can be computed using automatic differentiation. For example, to extract the sensitivity of the solution norm with respect to D, we have 

```julia
gradients(sum(sol^2), D)
```

The output is a 2 by 2 sensitivity matrix. 

## Inversion

If we only know the discrete solution, and the form of $D=x\mathbf{I}$, $x>0$. This can be easily done by replacing `D = constant(diagm(0=>ones(2)))` with (the initial guess for $x=2$)

```julia
D = Variable(2.0) .* [1.0 0.0;0.0 1.0]
```

Then, we can estimate $x$ using L-BFGS-B 

```julia
loss = sum((sol - (@. 1-node[:,1]^2-node[:,2]^2))^2)
sess = Session(); init(sess)
BFGS!(sess, loss)
```

The estimated result is 

$$D = \begin{bmatrix}1.0028 & 0.0\\ 0.0 & 1.0028\end{bmatrix}$$




## Summary

Finite element analysis is a powerful tool in numerical PDEs. However, it is more conceptually sophisticated than the finite difference method and requires more implementation efforts. The important lesson we learned from this tutorial is the necessity of `while_loop`, how to separate the computation into pure Julia and ADCME C++ kernels, and how complex numerical schemes can be implemented in ADCME. 


