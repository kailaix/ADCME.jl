# Newton Raphson

Newton-Raphson algorithm is widely used in scientific computing. In ADCME, the function for the algorithm is [`newton_raphson`](@ref). And the signature is
```@docs
newton_raphson
```

As an example, assume we want to solve 
```math
u_i^2 - 1 = 0, i=1,2,\ldots, 10
```
We first need to construct a function 
```julia
function f(θ, u)
    return u^2 - 1, 2*spdiag(u)
end
```
Here $2\texttt{spdiag}(u)$ is the Jacobian matrix for the equation. Then we construct a Newton Raphson solver via
```julia
nr = newton_raphson(f, constant(rand(10)))
```
`nr` is a `NRResult` struct which is runnable and can be materialized by 
```julia
nr = run(sess, nr)
```
The signature for `NRResult` is 
```julia
struct NRResult
    x::Union{PyObject, Array{Float64}} # final solution
    res::Union{PyObject, Array{Float64, 1}} # residual
    u::Union{PyObject, Array{Float64, 2}} # solution history
    converged::Union{PyObject, Bool} # whether it converges
    iter::Union{PyObject, Int64} # number of iterations
end
```
`u`$\in \mathbb{R}^{p\times n}$ where `p` is the solution dimension and `n` is the number of iterations. 

!!! note
    Sometimes we want to construct `f` via some external variables $\theta$, e.g., when $\theta$ is a trainable variable and embeded in the Newton-Raphson solver, we can pass this parameter to `newton_raphson` via the third parameter
    ```julia
    nr = newton_raphson(f, constant(rand(10)),θ)
    ```

!!! note
    `newton_raphson` also accepts a keyword argument `options` through which we can specify special options for the optimization. For example
    ```julia
    nr = newton_raphson(f, constant(rand(10)), missing, 
                options=Dict("verbose"=>true, "tol"=>1e-12))
    ```
    This might be useful for debugging.

In the case we want to apply a linesearch step in our Newton-Raphson solver, we can turn on the `linesearch` option in `options`. However, in this case, we must provide the function value for `f` (assuming we are solving a minimization problem).  
```julia
function f(θ, u)
    return sum(1/3*u^3-u), u^2 - 1, 2*spdiag(u)
end
```

The corresponding driver code is
```julia
nr = newton_raphson(f, constant(rand(10)), missing, 
                options=Dict("verbose"=>false, "tol"=>1e-12, "linesearch"=>true, "ls_αinitial"=>1.0))
```


Finally we consider an advanced usage of the code, where we want to create a custom operator that solves
```math
y^3-x=0
```
We compute the forward using Newton-Raphson and the backward with the implicit function theorem.
```julia
using Random
function myop_(x)
    function f(θ, y)
        y^3 - x, spdiag(3y^2)
    end
    nr = newton_raphson(f, constant(ones(length(x))), options=Dict("verbose"=>true))
    y = nr.x
    function myop_grad(dy, y)
        dy/3y^2
    end
    # move variables to python space
    s = randstring(8)
py"""
y_$$s = $y
grad_$$s = $myop_grad
"""
    # workaround 
    g = py"""lambda dy: grad_$$s(dy, y_$$s)"""
    return y, g
end
tf_myop = tf.custom_gradient(myop_)
```

!!! note
    Here `py"""lambda dy: grad_$$s(dy, y_$$s)"""` is related to a [workaround](https://github.com/JuliaPy/PyCall.jl/issues/367) for converting Julia function to Python function. 
    Also we need to explicitly put Julia object to Python. 

```julia
x = constant(8ones(5))
y = tf_myop(x)
println(run(sess, y))

l = sum(y)
run(sess, gradients(l, x))
```