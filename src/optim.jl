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
CustomOptimizer

function AdamOptimizer(learning_rate=1e-3;kwargs...)
    return tf.train.AdamOptimizer(;learning_rate=learning_rate,kwargs...)
end

function AdadeltaOptimizer(learning_rate=1e-3;kwargs...)
    return tf.train.AdadeltaOptimizer(;learning_rate=learning_rate,kwargs...)
end

function AdagradDAOptimizer(learning_rate=1e-3; global_step, kwargs...)
    return tf.train.AdagradDAOptimizer(learning_rate, global_step;kwargs...)
end

function AdagradOptimizer(learning_rate=1e-3;kwargs...)
    return tf.train.AdagradOptimizer(learning_rate;kwargs...)
end

function GradientDescentOptimizer(learning_rate=1e-3;kwargs...)
    return tf.train.GradientDescentOptimizer(learning_rate;kwargs...)
end

function RMSPropOptimizer(learning_rate=1e-3;kwargs...)
    return tf.train.RMSPropOptimizer(learning_rate;kwargs...)
end

function minimize(o::PyObject, loss::PyObject; kwargs...)
    o.minimize(loss;kwargs...)
end

"""
ScipyOptimizerInterface(
    loss,
    var_list=None,
    equalities=None,
    inequalities=None,
    var_to_bounds=None,
    **optimizer_kwargs
)
https://www.tensorflow.org/api_docs/python/tf/contrib/opt/ScipyOptimizerInterface
"""
ScipyOptimizerInterface(loss; method="L-BFGS-B", options=Dict("maxiter"=> 15000, "ftol"=>1e-12, "gtol"=>1e-12), kwargs...) = 
            tf.contrib.opt.ScipyOptimizerInterface(loss; method = method, options=options, kwargs...)

"""
ScipyOptimizerMinimize(
    session=None,
    feed_dict=None,
    fetches=None,
    step_callback=None,
    loss_callback=None,
    **run_kwargs
)
Minimize a scalar Tensor.

Variables subject to optimization are updated in-place at the end of optimization.

Note that this method does not just return a minimization Op, unlike `minimize`; instead it actually performs minimization by executing commands to control a Session
https://www.tensorflow.org/api_docs/python/tf/contrib/opt/ScipyOptimizerInterface

kwargs
======
-- feed_dict: A feed dict to be passed to calls to session.run.
-- fetches: A list of Tensors to fetch and supply to loss_callback as positional arguments.
-- step_callback: A function to be called at each optimization step; arguments are the current values of all optimization variables flattened into a single vector.
-- loss_callback: A function to be called every time the loss and gradients are computed, with evaluated fetches supplied as positional arguments.
-- run_kwargs: kwargs to pass to session.run.
"""
function ScipyOptimizerMinimize(sess::PyObject, opt::PyObject; kwargs...)
    opt.minimize(sess;kwargs...)
end

@doc """
CustomOptimizer(opt::Function, name::String)

creates a custom optimizer with struct name `name`. For example, we can integrate `Optim.jl` with `ADCME` by 
constructing a new optimizer
```julia
CustomOptimizer("Con") do f, df, c, dc, x0, nineq, neq, x_L, x_U
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
Then we can create an optimizer with 
```
opt = Con(loss, inequalities=[c1], equalities=[c2])
```
To trigger the optimization, use
```
opt.minimize(sess)
```
or 
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
                x_L = vcat([x[1] for x in packed_bounds]...)
                x_U = vcat([x[2] for x in packed_bounds]...)
                x0 = initial_val # rename 
                nineq, neq = length(inequality_funcs), length(equality_funcs)
                nvar = Int64(sum([prod(self._vars[i].get_shape().as_list()) for i = 1:length(self._vars)]))
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
                $opt(f, df, c, dc, x0, nineq, neq, x_L, x_U)
            end
        end
        return $name
    end
end


@doc """
BFGS!(sess::PyObject, loss::PyObject, max_iter::Int64=15000; kwargs...)

`BFGS!` is a simplified interface for BFGS optimizer. 
"""->
function BFGS!(sess::PyObject, loss::PyObject, max_iter::Int64=15000; kwargs...)
    __cnt = 0
    __loss = 0
    out = []
    function print_loss(l)
        if mod(__cnt,1)==0
            println("iter $__cnt, current loss=",l)
        end
        __loss = l
        __cnt += 1
    end
    __iter = 0
    function step_callback(rk)
        if mod(__iter,1)==0
            println("================ ITER $__iter ===============")
        end
        push!(out, __loss)
        __iter += 1
    end
    opt = ScipyOptimizerInterface(loss, method="L-BFGS-B",options=Dict("maxiter"=> max_iter, "ftol"=>1e-12, "gtol"=>1e-12))
    @info "Optimization starts..."
    ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=step_callback, fetches=[loss])
    out
end


 