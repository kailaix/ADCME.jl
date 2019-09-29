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
NonlinearConstrainedProblem,
verify_jacobian,
verify_NonlinearConstrainedProblem

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
                local x_L, x_U
                x0 = initial_val # rename 
                nineq, neq = length(inequality_funcs), length(equality_funcs)
                nvar = Int64(sum([prod(self._vars[i].get_shape().as_list()) for i = 1:length(self._vars)]))
                if isnothing(packed_bounds)
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

"""
    newton_raphson(f::Function, u::Union{Array,PyObject}, θ::Union{Missing,PyObject}; options::Union{Dict{String, T}, Missing}=missing)

Newton Raphson solver for solving a nonlinear equation. 
`f` has the signature `f(u::PyObject, θ::Union{Missing,PyObject})->(r::PyObject, A::Union{PyObject,SparseTensor})`
where `r` is the residual and `A` is the Jacobian matrix. 
`θ` are external parameters.
`u0` is the initial guess for `u`
`options`:
- "max_iter": maximum number of iterations (default=100)
- "verbose": whether details are printed (default=false)
- "rtol": relative tolerance for termination (default=1e-12)
- "tol": absolute tolerance for termination (default=1e-12)
"""
function newton_raphson(f::Function, u0::Union{Array,PyObject}, θ::Union{Missing,PyObject, Array{<:Real}}=missing; options::Union{Dict{String, T}, Missing}=missing) where T<:Real
    options_ = Dict(
            "max_iter"=>100,
            "verbose"=>false,
            "rtol"=>1e-12,
            "tol"=>1e-12
        )
    if !ismissing(options)
        for k in keys(options)
            options_[k] = options[k]
        end
    end
    options = options_
    if length(size(u0))!=1
        error("ADCME: Initial guess must be a vector")
    end
    if length(u0)===nothing
        error("ADCME: The length of the initial guess must be determined at compilation.")
    end
    u = convert_to_tensor(u0)

    function condition(i,  ta_r, ta_u)
        if options["verbose"]; @info "(2/4)Parsing Condition..."; end
        if_else(tf.math.logical_and(tf.equal(i,2), tf.less(i, options["max_iter"]+1)), 
            constant(true),
            ()->begin
                tol = read(ta_r, i-1)
                rel_tol = read(ta_r, i-2)
                if options["verbose"]
                    op = tf.print("Iteration =",i-1, "| Tol =", tol, "( $(options["tol"]) )", "| Rel_Tol =", rel_tol, 
                        "( $(options["rtol"]) )", summarize=-1)
                    tol = bind(tol, op)
                end
                return tf.math.logical_and(
                    tf.math.logical_and(tol>=options["tol"], rel_tol>=options["rtol"]),
                    i<=options["max_iter"]
                )
            end
        )
    end
    function body(i, ta_r, ta_u)
        if options["verbose"]; @info "(3/4)Parsing Main Loop..."; end
        u_ = read(ta_u, i-1)
        r_, J = f(θ, u_)
        ta_r = write(ta_r, i, norm(r_))
        δ = J\r_
        new_u = u_ - δ
        # op = tf.print(i,"==>\n", u_, "\n", norm(r_),"\n", summarize=-1)
        # i = bind(i, op)
        ta_u = write(ta_u, i, new_u)     
        i+1, ta_r, ta_u
    end
    
    
    if options["verbose"]; @info "(1/4)Intializing TensorArray..."; end
    r0, _ = f(θ, u)
    tol0 = norm(r0)
    if options["verbose"]
        op = tf.print("Iteration = 1", "| Tol =", tol0, "( $(options["tol"]) )", "| Rel_Tol = ---", 
        "( $(options["rtol"]) )", summarize=-1)
        tol0 = bind(tol0, op)
    end

    ta_r = TensorArray(options["max_iter"])
    ta_u = TensorArray(options["max_iter"])
    ta_u = write(ta_u, 1, u)
    ta_r = write(ta_r, 1, tol0)
    i = constant(2, dtype=Int32)
    i_, ta_r_, ta_u_ = while_loop(condition, body, [i, ta_r, ta_u])
    r_out, u_out = stack(ta_r_), stack(ta_u_)
    
    if options["verbose"]; @info "(4/4)Postprocessing Results..."; end
    sol = if_else(
        tf.less(tol0,options["tol"]),
        u,
        u_out[i_-1]
    )
    res = if_else(
        tf.less(tol0,options["tol"]),
        reshape(tol0, 1),
        tf.slice(r_out, [1],[i_-2])
    )
    u_his = if_else(
        tf.less(tol0,options["tol"]),
        reshape(u, 1, length(u)),
        tf.slice(u_out, [0; 0], [i_-2; length(u)])
    )
    iter = if_else(
        tf.less(tol0,options["tol"]),
        constant(1),
        cast(Int64,i_)-2
    )
    converged = if_else(
        tf.less(i_, options_["max_iter"]),
        constant(true),
        constant(false)
    )
    # it makes no sense to take the gradients
    sol = stop_gradient(sol)
    res = stop_gradient(res)
    NRResult(sol, res, u_his', converged, iter)
end


function verify_jacobian(sess::PyObject, f::Function, θ::Union{Array{Float64}, PyObject, Missing}, u0::Array{Float64}, args...)
    u = placeholder(Float64, shape=[length(u0)])
    L, J = f(θ, u)
    L_ = run(sess, L, u=>u0, args...)
    J_ = run(sess, J, u=>u0, args...)
    v = rand(length(u0))
    γs = 1.0 ./ 10 .^ (1:5)
    v1 = Float64[]
    v2 = Float64[]
    for i = 1:5
        L__ = run(sess, L, u=>u0+v*γs[i], args...)
        push!(v1, norm(L__-L_))
        push!(v2, norm(L__-L_-γs[i]*J_*v))
    end
    close("all")
    loglog(γs, abs.(v1), "*-", label="finite difference")
    loglog(γs, abs.(v2), "+-", label="automatic linearization")
    loglog(γs, γs.^2 * 0.5*abs(v2[1])/γs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
    loglog(γs, γs * 0.5*abs(v1[1])/γs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")
    plt.gca().invert_xaxis()
    legend()
    xlabel("\$\\gamma\$")
    ylabel("Error")
end


@doc raw"""
    NonlinearConstrainedProblem(f::Function, L::Function, θ::PyObject, u0::Union{PyObject, Array{Float64}}; options::Union{Dict{String, T}, Missing}=missing) where T<:Integer

Computes the gradients ``\frac{\partial L}{\partial \theta}``
```math
\begin{align}
\min & \ L(u)\\ 
\mathrm{s.t.} & \ F(\theta, u) = 0
\end{align}
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

"""
function NonlinearConstrainedProblem(f::Function, L::Function, θ::Union{Array{Float64,1},PyObject},
     u0::Union{PyObject, Array{Float64}}; options::Union{Dict{String, T}, Missing}=missing) where T<:Real
    θ = convert_to_tensor(θ)
    nr = newton_raphson(f, u0, θ, options = options)
    r, A = f(θ, nr.x)
    l = L(nr.x)
    top_grad = tf.convert_to_tensor(gradients(l, nr.x))
    A = A'
    g = A\top_grad
    g = stop_gradient(g) # preventing gradients backprop
    l, nr.x, tf.convert_to_tensor(-gradients(sum(r*g), θ))
end

function verify_NonlinearConstrainedProblem(sess::PyObject, f::Function, L::Function, θ::Union{PyObject,Array{Float64,1}, Float64}, 
    u0::Union{PyObject, Array{Float64}}, args...; options::Union{Dict{String, T}, Missing}=missing) where T<:Real
    if isa(θ, PyObject)
        θ = run(sess, θ, args...)
    end
    x = placeholder(Float64, shape=[length(θ)])
    l, u, g = NonlinearConstrainedProblem(f, L, x, u0; options=options)
    L_ = run(sess, l, x=>θ, args...)
    J_ = run(sess, g, x=>θ, args...)
    v = rand(length(x))
    γs = 1.0 ./ 10 .^ (1:5)
    v1 = Float64[]
    v2 = Float64[]
    for i = 1:5
        L__ = run(sess, l, x=>θ+v*γs[i], args...)
        # @show L__,L_,J_, v
        push!(v1, L__-L_)
        push!(v2, L__-L_-γs[i]*sum(J_.*v))
    end
    close("all")
    loglog(γs, abs.(v1), "*-", label="finite difference")
    loglog(γs, abs.(v2), "+-", label="automatic linearization")
    loglog(γs, γs.^2 * 0.5*abs(v2[1])/γs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
    loglog(γs, γs * 0.5*abs(v1[1])/γs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")
    plt.gca().invert_xaxis()
    legend()
    xlabel("\$\\gamma\$")
    ylabel("Error")
end