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
    We can provide options to `newton_raphson` using `ADCME.options.newton_raphson`. For example
    ```julia
    ADCME.options.newton_raphson.verbose = true 
    ADCME.options.newton_raphson.tol = 1e-6
    nr = newton_raphson(f, constant(rand(10)), missing)
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
ADCME.options.newton_raphson.verbose = false
ADCME.options.newton_raphson.linesearch = true
ADCME.options.newton_raphson.tol = 1e-12
ADCME.options.newton_raphson.linesearch_options.αinitial = 1.0
nr = newton_raphson(f, constant(rand(10)), missing
```


Finally we consider the differentiable Newton-Raphson algorithm. Consider we want to construct a map $f:x\mapsto y$, which satisfies
```math
y^3-x=0
```

In a later stage, we also want to evaluate $\frac{dy}{dx}$. To this end, we can use [`newton_raphson_with_grad`](@ref), which provides a differentiable implementation of the Newton-Raphson's algorithm. 

```julia
function f(θ, x)
    x^3 - θ, 3spdiag(x^2)
end

θ = constant([2. .^3;3. ^3; 4. ^3])
x = newton_raphson_with_grad(f, constant(ones(3)), θ)
run(sess, x)≈[2.;3.;4.]
run(sess, gradients(sum(x), θ))≈1/3*[2. .^3;3. ^3; 4. ^3] .^(-2/3)
```

