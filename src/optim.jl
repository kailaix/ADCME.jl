export
AdadeltaOptimizer,
AdagradDAOptimizer,
AdagradOptimizer,
AdamOptimizer,
GradientDescentOptimizer,
RMSPropOptimizer,
minimize,
ScipyOptimizerInterface,
ScipyOptimizerMinimize

function AdamOptimizer(learning_rate=1e-3;kwargs...)
    return tf.train.AdamOptimizer(;kwargs...)
end

function AdadeltaOptimizer(learning_rate=1e-3;kwargs...)
    return tf.train.AdadeltaOptimizer(;kwargs...)
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