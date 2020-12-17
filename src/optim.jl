export
AdadeltaOptimizer,
AdagradDAOptimizer,
AdagradOptimizer,
AdamOptimizer,
GradientDescentOptimizer,
RMSPropOptimizer,
minimize,
ScipyOptimizerInterface,
ScipyOptimizerMinimize,
BFGS!,
CustomOptimizer,
newton_raphson,
newton_raphson_with_grad,
NonlinearConstrainedProblem,
Optimize!,
optimize


#### Section 1: TensorFlow Optimizers ####
@doc raw"""
    AdamOptimizer(learning_rate=1e-3;kwargs...)

Constructs an ADAM optimizer. 

# Example 
```julia 
learning_rate = 1e-3
opt = AdamOptimizer(learning_rate).minimize(loss)
sess = Session(); init(sess)
for i = 1:1000
    _, l = run(sess, [opt, loss])
    @info "Iteration $i, loss = $l")
end
```

# Dynamical Learning Rate
We can also use dynamical learning rate. For example, if we want to use a learning rate
$l_t = \frac{1}{1+t}$, we have 

```julia 
learning_rate = placeholder(1.0)
opt = AdamOptimizer(learning_rate).minimize(loss)
sess = Session(); init(sess)
for i = 1:1000
    _, l = run(sess, [opt, loss], lr = 1/(1+i))
    @info "Iteration $i, loss = $l")
end
```

The usage of other optimizers such as [`GradientDescentOptimizer`](@ref), [`AdadeltaOptimizer`](@ref), and so on 
is similar: we can just replace `AdamOptimizer` with the corresponding ones. 
"""
function AdamOptimizer(learning_rate=1e-3;kwargs...)
    return tf.train.AdamOptimizer(;learning_rate=learning_rate,kwargs...)
end

"""
    AdadeltaOptimizer(learning_rate=1e-3;kwargs...)

See [`AdamOptimizer`](@ref) for descriptions.
"""
function AdadeltaOptimizer(learning_rate=1e-3;kwargs...)
    return tf.train.AdadeltaOptimizer(;learning_rate=learning_rate,kwargs...)
end


"""
    AdagradDAOptimizer(learning_rate=1e-3; global_step, kwargs...)

See [`AdamOptimizer`](@ref) for descriptions.
"""
function AdagradDAOptimizer(learning_rate=1e-3; global_step, kwargs...)
    return tf.train.AdagradDAOptimizer(learning_rate, global_step;kwargs...)
end


"""
    AdagradOptimizer(learning_rate=1e-3;kwargs...)

See [`AdamOptimizer`](@ref) for descriptions.
"""
function AdagradOptimizer(learning_rate=1e-3;kwargs...)
    return tf.train.AdagradOptimizer(learning_rate;kwargs...)
end


"""
    GradientDescentOptimizer(learning_rate=1e-3;kwargs...)

See [`AdamOptimizer`](@ref) for descriptions.
"""
function GradientDescentOptimizer(learning_rate=1e-3;kwargs...)
    return tf.train.GradientDescentOptimizer(learning_rate;kwargs...)
end


"""
    RMSPropOptimizer(learning_rate=1e-3;kwargs...)

See [`AdamOptimizer`](@ref) for descriptions.
"""
function RMSPropOptimizer(learning_rate=1e-3;kwargs...)
    return tf.train.RMSPropOptimizer(learning_rate;kwargs...)
end

function minimize(o::PyObject, loss::PyObject; kwargs...)
    o.minimize(loss;kwargs...)
end

"""
    ScipyOptimizerInterface(loss; method="L-BFGS-B", options=Dict("maxiter"=> 15000, "ftol"=>1e-12, "gtol"=>1e-12), kwargs...)

A simple interface for Scipy Optimizer. See also [`ScipyOptimizerMinimize`](@ref) and [`BFGS!`](@ref).
"""
ScipyOptimizerInterface(loss; method="L-BFGS-B", options=Dict("maxiter"=> 15000, "ftol"=>1e-12, "gtol"=>1e-12), kwargs...) = 
            tf.contrib.opt.ScipyOptimizerInterface(loss; method = method, options=options, kwargs...)

"""
    ScipyOptimizerMinimize(sess::PyObject, opt::PyObject; kwargs...)

Minimizes a scalar Tensor. Variables subject to optimization are updated in-place at the end of optimization.

Note that this method does not just return a minimization Op, unlike `minimize`; instead it actually performs minimization by executing commands to control a Session
https://www.tensorflow.org/api_docs/python/tf/contrib/opt/ScipyOptimizerInterface. See also [`ScipyOptimizerInterface`](@ref) and [`BFGS!`](@ref).


- feed_dict: A feed dict to be passed to calls to session.run.
- fetches: A list of Tensors to fetch and supply to loss_callback as positional arguments.
- step_callback: A function to be called at each optimization step; arguments are the current values of all optimization variables packed into a single vector.
- loss_callback: A function to be called every time the loss and gradients are computed, with evaluated fetches supplied as positional arguments.
- run_kwargs: kwargs to pass to session.run.
"""
function ScipyOptimizerMinimize(sess::PyObject, opt::PyObject; kwargs...)
    opt.minimize(sess;kwargs...)
end

@doc raw"""
    CustomOptimizer(opt::Function, name::String)

creates a custom optimizer with struct name `name`. For example, we can integrate `Optim.jl` with `ADCME` by 
constructing a new optimizer
```julia
CustomOptimizer("Con") do f, df, c, dc, x0, x_L, x_U
    opt = Opt(:LD_MMA, length(x0))
    bd = zeros(length(x0)); bd[end-1:end] = [-Inf, 0.0]
    opt.lower_bounds = bd
    opt.xtol_rel = 1e-4
    opt.min_objective = (x,g)->(g[:]= df(x); return f(x)[1])
    inequality_constraint!(opt, (x,g)->( g[:]= dc(x);c(x)[1]), 1e-8)
    (minf,minx,ret) = NLopt.optimize(opt, x0)
    minx
end
```
Here

∘ `f`: a function that returns $f(x)$

∘ `df`: a function that returns $\nabla f(x)$

∘ `c`: a function that returns the constraints $c(x)$

∘ `dc`: a function that returns $\nabla c(x)$

∘ `x0`: initial guess

∘ `nineq`: number of inequality constraints

∘ `neq`: number of equality constraints

∘ `x_L`: lower bounds of optimizable variables

∘ `x_U`: upper bounds of optimizable variables

Then we can create an optimizer with 
```
opt = Con(loss, inequalities=[c1], equalities=[c2])
```
To trigger the optimization, use
```
minimize(opt, sess)
```

Note thanks to the global variable scope of Julia, `step_callback`, `optimizer_kwargs` can actually 
be passed from Julia environment directly.
"""
function CustomOptimizer(opt::Function)
    name = "CustomOptimizer_"*randstring(16)
    name = Symbol(name)
    @eval begin
        @pydef mutable struct $name <: tf.contrib.opt.ExternalOptimizerInterface
            function _minimize(self; initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                packed_bounds, step_callback, optimizer_kwargs)
                local x_L, x_U
                x0 = initial_val # rename 
                nineq, neq = length(inequality_funcs), length(equality_funcs)
                printstyled("[CustomOptimizer] Number of inequalities constraints = $nineq, Number of equality constraints = $neq\n", color=:blue)
                nvar = Int64(sum([prod(self._vars[i].get_shape().as_list()) for i = 1:length(self._vars)]))
                printstyled("[CustomOptimizer] Total number of variables = $nvar\n", color=:blue)
                if isnothing(packed_bounds)
                    printstyled("[CustomOptimizer] No bounds provided, use (-∞, +∞) as default; or you need to provide bounds in the function CustomOptimizer\n", color=:blue)
                    x_L = -Inf*ones(nvar); x_U = Inf*ones(nvar)
                else
                    x_L = vcat([x[1] for x in packed_bounds]...)
                    x_U = vcat([x[2] for x in packed_bounds]...)
                end
                ncon = nineq + neq
                f(x) = loss_grad_func(x)[1]
                df(x) = loss_grad_func(x)[2]
                
                function c(x)
                    inequalities = vcat([inequality_funcs[i](x) for i = 1:nineq]...)
                    equalities = vcat([equality_funcs[i](x) for i=1:neq]...)
                    return Array{eltype(initial_val)}([inequalities;equalities])
                end
                function dc(x)
                    inequalities = [inequality_grad_funcs[i](x) for i = 1:nineq]
                    equalities = [equality_grad_funcs[i](x) for i=1:neq]
                    values = zeros(eltype(initial_val),nvar, ncon)
                    for idc = 1:nineq
                        values[:,idc] = inequalities[idc][1]
                    end
                    for idc = 1:neq
                        values[:,idc+nineq] = equalities[idc][1]
                    end
                    return values[:]
                end
                $opt(f, df, c, dc, x0, x_L, x_U)
            end
        end
        return $name
    end
end


@doc raw"""
    BFGS!(sess::PyObject, loss::PyObject, max_iter::Int64=15000; 
    vars::Array{PyObject}=PyObject[], callback::Union{Function, Nothing}=nothing, method::String = "L-BFGS-B", kwargs...)

`BFGS!` is a simplified interface for **L-BFGS-B** optimizer. See also [`ScipyOptimizerInterface`](@ref).
`callback` is a callback function with signature 
```julia
callback(vs::Array, iter::Int64, loss::Float64)
```
`vars` is an array consisting of tensors and its values will be the input to `vs`.

# Example 1
```julia
a = Variable(1.0)
loss = (a - 10.0)^2
sess = Session(); init(sess)
BFGS!(sess, loss)
```

# Example 2
```julia
θ1 = Variable(1.0)
θ2 = Variable(1.0)
loss = (θ1-1)^2 + (θ2-2)^2
cb = (vs, iter, loss)->begin 
    printstyled("[#iter $iter] θ1=$(vs[1]), θ2=$(vs[2]), loss=$loss\n", color=:green)
end
sess = Session(); init(sess)
cb(run(sess, [θ1, θ2]), 0, run(sess, loss))
BFGS!(sess, loss, 100; vars=[θ1, θ2], callback=cb)
```

# Example 3
Use `bounds` to specify upper and lower bound of a variable. 
```julia
x = Variable(2.0)    
loss = x^2
sess = Session(); init(sess)
BFGS!(sess, loss, bounds=Dict(x=>[1.0,3.0]))
```

!!! note 
    Users can also use other scipy optimization algorithm by providing `method` keyword arguments. For example, you can use the BFGS optimizer 
    ```julia
    BFGS!(sess, loss, method = "BFGS")
    ```
"""->
function BFGS!(sess::PyObject, loss::PyObject, max_iter::Int64=15000; 
    vars::Array{PyObject}=PyObject[], callback::Union{Function, Nothing}=nothing, method::String = "L-BFGS-B", kwargs...)
    __cnt = 0
    __loss = 0
    __var = nothing
    out = []
    function print_loss(l, vs...)
        if !isnothing(callback); __var = vs; end
        if mod(__cnt,1)==0
            println("iter $__cnt, current loss=",l)
        end
        __loss = l
        __cnt += 1
    end
    __iter = 0
    function step_callback(rk)
        if mod(__iter,1)==0
            println("================ STEP $__iter ===============")
        end
        if !isnothing(callback)
            callback(__var, __iter, __loss)
        end
        push!(out, __loss)
        __iter += 1
    end
    kwargs = Dict(kwargs)
    if haskey(kwargs, :bounds)
        kwargs[:var_to_bounds] = kwargs[:bounds]
        delete!(kwargs, :bounds)
    end
    if haskey(kwargs, :var_to_bounds)
        desc = "`bounds` or `var_to_bounds` keywords of `BFGS!` only accepts dictionaries whose keys are Variables"
        for (k,v) in kwargs[:var_to_bounds]
            if !(hasproperty(k, "trainable"))
                error("The tensor $k does not have property `trainable`, indicating it is not a `Variable`. $desc\n")
            end 
            if !k.trainable
                @warn("The variable $k is not trainable, ignoring the bounds")
            end
        end
    end
    opt = ScipyOptimizerInterface(loss, method=method,options=Dict("maxiter"=> max_iter, "ftol"=>1e-12, "gtol"=>1e-12); kwargs...)
    @info "Optimization starts..."
    ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=step_callback, fetches=[loss, vars...])
    out
end

"""
    BFGS!(value_and_gradients_function::Function, initial_position::Union{PyObject, Array{Float64}}, max_iter::Int64=50, args...;kwargs...)

Applies the BFGS optimizer to `value_and_gradients_function`
"""
function BFGS!(value_and_gradients_function::Function, 
    initial_position::Union{PyObject, Array{Float64}}, max_iter::Int64=50, args...;kwargs...)
    tfp.optimizer.bfgs_minimize(value_and_gradients_function, 
        initial_position=initial_position, args...;max_iterations=max_iter, kwargs...)[5]
end

struct NRResult
    x::Union{PyObject, Array{Float64}} # final solution
    res::Union{PyObject, Array{Float64, 1}} # residual
    u::Union{PyObject, Array{Float64, 2}} # solution history
    converged::Union{PyObject, Bool} # whether it converges
    iter::Union{PyObject, Int64} # number of iterations
end

function Base.:run(sess::PyObject, nr::NRResult)
    NRResult(run(sess, [nr.x, nr.res, nr.u, nr.converged, nr.iter])...)
end


function backtracking(compute_gradient::Function , u::PyObject)
    f0, r0, _, δ0 = compute_gradient(u)
    df0 = -sum(r0.*δ0) 
    c1 = options.newton_raphson.linesearch_options.c1
    ρ_hi = options.newton_raphson.linesearch_options.ρ_hi
    ρ_lo = options.newton_raphson.linesearch_options.ρ_lo
    iterations = options.newton_raphson.linesearch_options.iterations
    maxstep = options.newton_raphson.linesearch_options.maxstep
    αinitial = options.newton_raphson.linesearch_options.αinitial

    @assert !isnothing(f0)
    @assert ρ_lo < ρ_hi
    @assert iterations > 0

    function condition(i, ta_α, ta_f)
        f = read(ta_f, i)
        α = read(ta_α, i)
        tf.logical_and(f > f0 + c1 * α * df0, i<=iterations)
    end

    function body(i, ta_α, ta_f)
        α_1 = read(ta_α, i-1)
        α_2 = read(ta_α, i)
        d = 1/(α_1^2*α_2^2*(α_2-α_1))
        f = read(ta_f, i)
        a = (α_1^2*(f - f0 - df0*α_2) - α_2^2*(df0 - f0 - df0*α_1))*d
        b = (-α_1^3*(f - f0 - df0*α_2) + α_2^3*(df0 - f0 - df0*α_1))*d

        α_tmp = tf.cond(abs(a)<1e-10,
            ()->df0/(2b),
            ()->begin
                d = max(b^2-3a*df0, constant(0.0))
                (-b + sqrt(d))/(3a)
            end)


        α_2 = tf.cond(tf.math.is_nan(α_tmp),
                ()->α_2*ρ_hi,
                ()->begin
                    α_tmp = min(α_tmp, α_2*ρ_hi)
                    α_2 = max(α_tmp, α_2*ρ_lo)
                end)

        fnew, _, _, _ = compute_gradient(u - α_2*δ0)
        ta_f = write(ta_f, i+1, fnew)
        ta_α = write(ta_α, i+1, α_2)
        i+1, ta_α, ta_f
    end

    ta_α = TensorArray(iterations)
    ta_α = write(ta_α, 1, constant(αinitial))
    ta_α = write(ta_α, 2, constant(αinitial))

    ta_f = TensorArray(iterations)
    ta_f = write(ta_f, 1, constant(0.0))
    ta_f = write(ta_f, 2, f0)

    i = constant(2, dtype=Int32)

    iter, out_α, out_f = while_loop(condition, body, [i, ta_α, ta_f]; back_prop=false)
    α = read(out_α, iter)
    return α
end

"""
    newton_raphson(func::Function, 
        u0::Union{Array,PyObject}, 
        θ::Union{Missing,PyObject, Array{<:Real}}=missing,
        args::PyObject...) where T<:Real

Newton Raphson solver for solving a nonlinear equation. 
∘ `func` has the signature 
- `func(θ::Union{Missing,PyObject}, u::PyObject)->(r::PyObject, A::Union{PyObject,SparseTensor})` (if `linesearch` is off)
- `func(θ::Union{Missing,PyObject}, u::PyObject)->(fval::PyObject, r::PyObject, A::Union{PyObject,SparseTensor})` (if `linesearch` is on)
where `r` is the residual and `A` is the Jacobian matrix; in the case where `linesearch` is on, the function value `fval` must also be supplied.
∘ `θ` are external parameters.
∘ `u0` is the initial guess for `u`
∘ `args`: additional inputs to the func function 
∘ `kwargs`: keyword arguments to `func`

The solution can be configured via `ADCME.options.newton_raphson`

- `max_iter`: maximum number of iterations (default=100)
- `rtol`: relative tolerance for termination (default=1e-12)
- `tol`: absolute tolerance for termination (default=1e-12)
- `LM`: a float number, Levenberg-Marquardt modification ``x^{k+1} = x^k - (J^k + \\mu^k)^{-1}g^k`` (default=0.0)
- `linesearch`: whether linesearch is used (default=false)

Currently, the backtracing algorithm is implemented.
The parameters for `linesearch` are supplied via `options.newton_raphson.linesearch_options`

- `c1`: stop criterion, ``f(x^k) < f(0) + \\alpha c_1  f'(0)``
- `ρ_hi`: the new step size ``\\alpha_1\\leq \\rho_{hi}\\alpha_0`` 
- `ρ_lo`: the new step size ``\\alpha_1\\geq \\rho_{lo}\\alpha_0`` 
- `iterations`: maximum number of iterations for linesearch
- `maxstep`: maximum allowable steps
- `αinitial`: initial guess for the step size ``\\alpha``
"""
function newton_raphson(func::Function, 
    u0::Union{Array,PyObject}, 
    θ::Union{Missing,PyObject, Array{<:Real}}=missing,
    args::PyObject...; kwargs...) where T<:Real
    f = (θ, u)->func(θ, u, args...; kwargs...)
    if length(size(u0))!=1
        error("ADCME: Initial guess must be a vector")
    end
    if length(u0)===nothing
        error("ADCME: The length of the initial guess must be determined at compilation.")
    end
    u = convert_to_tensor(u0)

    max_iter = options.newton_raphson.max_iter
    verbose = options.newton_raphson.verbose
    oprtol = options.newton_raphson.rtol
    optol = options.newton_raphson.tol
    LM = options.newton_raphson.LM
    linesearch = options.newton_raphson.linesearch

    function condition(i,  ta_r, ta_u)
        if verbose; @info "(2/4)Parsing Condition..."; end
        if_else(tf.math.logical_and(tf.equal(i,2), tf.less(i, max_iter+1)), 
            constant(true),
            ()->begin
                tol = read(ta_r, i-1)
                rel_tol = read(ta_r, i-2)
                if verbose
                    op = tf.print("Newton iteration ", i-2, ": r (abs) =", tol, " r (rel) =", rel_tol, summarize=-1)
                    tol = bind(tol, op)
                end
                return tf.math.logical_and(
                    tf.math.logical_and(tol>=optol, rel_tol>=oprtol),
                    i<=max_iter
                )
            end
        )
    end
    function body(i, ta_r, ta_u)
        local δ, val, r_
        if verbose; @info "(3/4)Parsing Main Loop..."; end
        u_ = read(ta_u, i-1)

        function compute_gradients(x)
            val = nothing
            out = f(θ, x)
            if length(out)==2
                r_, J = out
            else
                val, r_, J = out
            end
            if LM>0.0 # Levenberg-Marquardt
                μ = LM
                μ = convert_to_tensor(μ)
                δ = (J + μ*spdiag(size(J,1)))\r_ 
            else
                δ = J\r_
            end
            return val, r_, J, δ
        end



        if linesearch
            if verbose; @info "Perform Linesearch..."; end
            step_size = backtracking(compute_gradients, u_)
        else
            step_size = 1.0
        end
        val, r_, _, δ = compute_gradients(u_)
        ta_r = write(ta_r, i, norm(r_))
        δ = step_size * δ
        new_u = u_ - δ

        if verbose && linesearch
            op = tf.print("Current Step Size = ", step_size)
            new_u = bind(new_u, op)
        end
        ta_u = write(ta_u, i, new_u)     
        i+1, ta_r, ta_u
    end
    
    
    if verbose; @info "(1/4)Intializing TensorArray..."; end
    out = f(θ, u)
    r0 = length(out)==2 ? out[1] : out[2]
    tol0 = norm(r0)
    if verbose
        op = tf.print("Newton-Raphson with absolute tolerance = $optol and relative tolerance = $oprtol")
        tol0 = bind(tol0, op)
    end

    ta_r = TensorArray(max_iter)
    ta_u = TensorArray(max_iter)
    ta_u = write(ta_u, 1, u)
    ta_r = write(ta_r, 1, tol0)
    i = constant(2, dtype=Int32)
    i_, ta_r_, ta_u_ = while_loop(condition, body, [i, ta_r, ta_u], back_prop=false)
    r_out, u_out = stack(ta_r_), stack(ta_u_)
    
    if verbose; @info "(4/4)Postprocessing Results..."; end
    sol = if_else(
        tf.less(tol0,optol),
        u,
        u_out[i_-1]
    )
    res = if_else(
        tf.less(tol0,optol),
        reshape(tol0, 1),
        tf.slice(r_out, [1],[i_-2])
    )
    u_his = if_else(
        tf.less(tol0,optol),
        reshape(u, 1, length(u)),
        tf.slice(u_out, [0; 0], [i_-2; length(u)])
    )
    iter = if_else(
        tf.less(tol0,optol),
        constant(1),
        cast(Int64,i_)-2
    )
    converged = if_else(
        tf.less(i_, max_iter),
        constant(true),
        constant(false)
    )
    # it makes no sense to take the gradients
    sol = stop_gradient(sol)
    res = stop_gradient(res)
    NRResult(sol, res, u_his', converged, iter)
end

"""
    newton_raphson_with_grad(f::Function, 
    u0::Union{Array,PyObject}, 
    θ::Union{Missing,PyObject, Array{<:Real}}=missing,
    args::PyObject...) where T<:Real

Differentiable Newton-Raphson algorithm. See [`newton_raphson`](@ref).

Use `ADCME.options.newton_raphson` to supply options. 

# Example 
```julia
function f(θ, x)
    x^3 - θ, 3spdiag(x^2)
end

θ = constant([2. .^3;3. ^3; 4. ^3])
x = newton_raphson_with_grad(f, constant(ones(3)), θ)
run(sess, x)≈[2.;3.;4.]
run(sess, gradients(sum(x), θ))
```
"""
function newton_raphson_with_grad(func::Function, 
    u0::Union{Array,PyObject}, 
    θ::Union{Missing,PyObject, Array{<:Real}}=missing,
    args::PyObject...; kwargs...) where T<:Real
    f = ( θ, u, args...) -> func(θ, u, args...; kwargs...)
    function forward(θ, args...)
        nr = newton_raphson(f, u0, θ, args...)
        return nr.x 
    end

    function backward(dy, x, θ, xargs...)
        θ = copy(θ)
        args = [copy(z) for z in xargs]
        r, A = f(θ, x, args...)
        dy = tf.convert_to_tensor(dy)
        g = independent(A'\dy)
        s = sum(r*g)
        gs = [-gradients(s, z) for z in args]
        if length(args)==0
            -gradients(s, θ)
        else
            -gradients(s, θ), gs...
        end
    end

    if !isa(θ, PyObject)
        @warn("θ is not a PyObject, no gradients is available")
        return forward(θ, args...)
    end
    fn = register(forward, backward)
    return fn(θ, args...)
end

@doc raw"""
    NonlinearConstrainedProblem(f::Function, L::Function, θ::PyObject, u0::Union{PyObject, Array{Float64}}; options::Union{Dict{String, T}, Missing}=missing) where T<:Integer

Computes the gradients ``\frac{\partial L}{\partial \theta}``
```math
\min \ L(u) \quad \mathrm{s.t.} \ F(\theta, u) = 0
```
`u0` is the initial guess for the numerical solution `u`, see [`newton_raphson`](@ref).

Caveats:
Assume `r, A = f(θ, u)` and `θ` are the unknown parameters,
`gradients(r, θ)` must be defined (backprop works properly)

Returns:
It returns a tuple (`L`: loss, `C`: constraints, and `Graidents`)
```math
\left(L(u), u, \frac{\partial L}{\partial θ}\right)
```

# Example 
We want to solve the following constrained optimization problem 
$$\begin{aligned}\min_\theta &\; L(u) = (u-1)^3\\ \text{s.t.} &\; u^3 + u = \theta\end{aligned}$$
The solution is $\theta = 2$. The Julia code is 
```julia
function f(θ, u)
    u^3 + u - θ, spdiag(3u^2+1) 
end
function L(u) 
    sum((u-1)^2)
end
pl = Variable(ones(1))
l, θ, dldθ = NonlinearConstrainedProblem(f, L, pl, ones(1))
```

We can coupled it with a mathematical optimizer 
```julia
using Optim 
sess = Session(); init(sess)
BFGS!(sess, l, dldθ, pl) 
```
"""
function NonlinearConstrainedProblem(f::Function, L::Function, θ::Union{Array{Float64,1},PyObject},
     u0::Union{PyObject, Array{Float64}}) where T<:Real
    θ = convert_to_tensor(θ)
    nr = newton_raphson(f, u0, θ)
    r, A = f(θ, nr.x)
    l = L(nr.x)
    top_grad = tf.convert_to_tensor(gradients(l, nr.x))
    A = A'
    g = A\top_grad
    g = independent(g) # preventing gradients backprop
    l, nr.x, tf.convert_to_tensor(-gradients(sum(r*g), θ))
end


@doc raw"""
    BFGS!(sess::PyObject, loss::PyObject, grads::Union{Array{T},Nothing,PyObject}, 
    vars::Union{Array{PyObject},PyObject}; kwargs...) where T<:Union{Nothing, PyObject}

Running BFGS algorithm
``\min_{\texttt{vars}} \texttt{loss}(\texttt{vars})``
The gradients `grads` must be provided. Typically, `grads[i] = gradients(loss, vars[i])`. 
`grads[i]` can exist on different devices (GPU or CPU). 

# Example 1
```julia
import Optim # required
a = Variable(0.0)
loss = (a-1)^2
g = gradients(loss, a)
sess = Session(); init(sess)
BFGS!(sess, loss, g, a)
```

# Example 2
```julia 
import Optim # required
a = Variable(0.0)
loss = (a^2+a-1)^2
g = gradients(loss, a)
sess = Session(); init(sess)
cb = (vs, iter, loss)->begin 
    printstyled("[#iter $iter] a = $vs, loss=$loss\n", color=:green)
end
BFGS!(sess, loss, g, a; callback = cb)
```
"""
function BFGS!(sess::PyObject, loss::PyObject, grads::Union{Array{T},Nothing,PyObject}, 
    vars::Union{Array{PyObject},PyObject}; kwargs...) where T<:Union{Nothing, PyObject}
    Optimize!(sess, loss; vars=vars, grads = grads, kwargs...)
end


"""
    Optimize!(sess::PyObject, loss::PyObject, max_iter::Int64 = 15000;
    vars::Union{Array{PyObject},PyObject, Missing} = missing, 
    grads::Union{Array{T},Nothing,PyObject, Missing} = missing, 
    optimizer = missing,
    callback::Union{Function, Missing}=missing,
    x_tol::Union{Missing, Float64} = missing,
    f_tol::Union{Missing, Float64} = missing,
    g_tol::Union{Missing, Float64} = missing, kwargs...) where T<:Union{Nothing, PyObject}

An interface for using optimizers in the Optim package or custom optimizers. 

- `sess`: a session;

- `loss`: a loss function;

- `max_iter`: maximum number of max_iterations;

- `vars`, `grads`: optimizable variables and gradients 

- `optimizer`: Optim optimizers (default: LBFGS)

- `callback`: callback after each linesearch completion (NOT one step in the linesearch)

Other arguments are passed to Options in Optim optimizers. 


We can also construct a custom optimizer. For example, to construct an optimizer out of Ipopt:

```julia
import Ipopt
x = Variable(rand(2))
loss = (1-x[1])^2 + 100(x[2]-x[1]^2)^2

function opt(f, g, fg, x0, kwargs...)
    prob = createProblem(2, -100ones(2), 100ones(2), 0, Float64[], Float64[], 0, 0,
                     f, (x,g)->nothing, (x,G)->g(G, x), (x, mode, rows, cols, values)->nothing, nothing)
    prob.x = x0 
    Ipopt.addOption(prob, "hessian_approximation", "limited-memory")
    status = Ipopt.solveProblem(prob)
    println(Ipopt.ApplicationReturnStatus[status])
    println(prob.x)
    Ipopt.freeProblem(prob)
    nothing
end

sess = Session(); init(sess)
Optimize!(sess, loss, optimizer = opt)
```
"""
function Optimize!(sess::PyObject, loss::PyObject, max_iter::Int64 = 15000;
    vars::Union{Array{PyObject},PyObject, Missing} = missing, 
    grads::Union{Array{T},Nothing,PyObject, Missing} = missing, 
    optimizer = missing,
    callback::Union{Function, Missing}=missing,
    x_tol::Union{Missing, Float64} = missing,
    f_tol::Union{Missing, Float64} = missing,
    g_tol::Union{Missing, Float64} = missing,
    return_feval_loss::Bool = false, kwargs...) where T<:Union{Nothing, PyObject}
    
    vars = coalesce(vars, get_collection())
    grads = coalesce(grads, gradients(loss, vars))
    if isa(vars, PyObject); vars = [vars]; end
    if isa(grads, PyObject); grads = [grads]; end
    if length(grads)!=length(vars); error("ADCME: length of grads and vars do not match"); end

    if !all(is_variable.(vars))
        error("ADCME: the input `vars` should be trainable variables")
    end

    idx = ones(Bool, length(grads))
    pynothing = pytypeof(PyObject(nothing))
    for i = 1:length(grads)
        if isnothing(grads[i]) || pytypeof(grads[i])==pynothing
            idx[i] = false
        end
    end
    grads = grads[idx]
    vars = vars[idx]
    sizes = []
    for v in vars
        push!(sizes, size(v))
    end
    grds = vcat([tf.reshape(g, (-1,)) for g in grads]...)
    vs = vcat([tf.reshape(v, (-1,)) for v in vars]...); x0 = run(sess, vs)
    pl = placeholder(x0)
    n = 0
    assign_ops = PyObject[]
    for (k,v) in enumerate(vars)
        push!(assign_ops, assign(v, tf.reshape(pl[n+1:n+prod(sizes[k])], sizes[k])))
        n += prod(sizes[k])
    end
    
    __loss = 0.0
    __losses = Float64[]
    __feval_loss = Float64[]
    __iter = 0
    __value = nothing
    __ls_iter = 0
    function f(x)
        run(sess, assign_ops, pl=>x)
        __loss = run(sess, loss)
        __ls_iter += 1
        push!(__feval_loss, __loss)
        options.training.verbose && (println("iter $__ls_iter, current loss = $__loss"))
        return __loss
    end

    function g!(G, x)
        run(sess, assign_ops, pl=>x)
        G[:] = run(sess, grds)
        __value = x
    end

    function fg(G, x)
        run(sess, assign_ops, pl=>x)
        __loss,  G[:] = run(sess, [loss, grds])
        __ls_iter += 1
        __value = x
        return __loss
    end

    
    if isa(optimizer, Function)
        if length(kwargs)>0
            kwargs_ = copy(kwargs)
        else
            kwargs_ = Dict{Symbol, Any}()
        end
        kwargs_[:max_iter] = max_iter
        __losses = optimizer(f, g!, fg, x0; kwargs_...)
        if return_feval_loss
            __losses = (__losses, __feval_loss)
        end
        return __losses
    end

    if !isdefined(Main, :Optim)
        error("Package Optim.jl must be imported in the main module using `import Optim` or `using Optim`")
    end

    function callback1(x)
        @info x.iteration, x.value
        __iter = x.iteration
        __loss = x.value
        push!(__losses, __loss)
        if options.training.verbose
            println("================== STEP $__iter ==================")
        end
        if !ismissing(callback)
            callback(__value, __iter, __loss)
        end
        false
    end

    method = coalesce(optimizer, Main.Optim.LBFGS())

    @info "Optimization starts..."
    res = Main.Optim.optimize(f, g!, x0, method, Main.Optim.Options(
        ; store_trace = false, 
        show_trace = false, 
        callback=callback1,
        iterations = max_iter,
        x_tol = coalesce(x_tol, 1e-12),
        f_tol = coalesce(f_tol, 1e-12),
        g_tol = coalesce(g_tol, 1e-12),
         kwargs...))

    if return_feval_loss
        __losses = (__losses, __feval_loss)
    end
    return __losses
end