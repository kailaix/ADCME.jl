import LinearAlgebra:cond
export
Graph,
reset_default_graph,
get_default_graph,
get_collection,
add_to_collection,
finalize_graph,
enable_eager_execution,
value,
control_dependencies,
has_gpu,
while_loop,
cond,
stop_gradient,
tensor,
RegisterGradient




Graph() = tf.Graph()
reset_default_graph() = tf.compat.v1.reset_default_graph()
get_default_graph() = tf.get_default_graph()
enable_eager_execution() = tf.enable_eager_execution()
value(o::PyObject) = o.numpy()
function RegisterGradient(args...;kwargs...)
    try
        tfops.RegisterGradient(args...;kwargs...)
    catch e
        @warn(e)
    end
end

"""
finalize(s::PyObject)

The method can help to catch leaks like this: it marks a graph as read-only, and raises an exception if anything is added to the graph
Reference: https://riptutorial.com/tensorflow/example/13426/use-graph-finalize---to-catch-nodes-being-added-to-the-graph
"""
finalize_graph(s::PyObject) = s.graph.finalize()


# TensorFlow Graph Collections
function get_collection(name, args...;kwargs...)
    if !(name in [GLOBAL_VARIABLES, TRAINABLE_VARIABLES, UPDATE_OPS])
        return get_collection(GLOBAL_VARIABLES, scope=name)
    else
        return tf.get_collection(name, args...;kwargs...)
    end
end
add_to_collection(args...;kwargs...) = tf.get_collection(args...;kwargs...)


function tensor(s::String)
    tf.get_default_graph().get_tensor_by_name(s)
end

function jlargs(kwargs)
    kwargs = Dict{Any, Any}(kwargs)
    if :axis in keys(kwargs)
        @error("axis is not a valid keyword, using dims instead (base = 1)")
    end
    if :dtype in keys(kwargs)
        kwargs[:dtype] = DTYPE[kwargs[:dtype]]
    end
    if :dims in keys(kwargs)
        kwargs[:axis] = kwargs[:dims] .- 1
        delete!(kwargs, :dims)
    end
    kwargs
end

# control_dependencies can be used to fix the memory problem
# https://stackoverflow.com/questions/39350164/tensorflow-parallel-for-loop-results-in-out-of-memory
function control_dependencies(f, ops)
    if isa(ops, PyObject)
        ops = [ops]
    end
    @pywith tf.control_dependencies(ops) begin
        f()
    end
end


function Base.:bind(op::PyObject, ops...)
    local op1
    control_dependencies(ops) do 
        op1 = tf.identity(op)
    end
    return op1
end

function while_loop(condition::Union{PyObject,Function}, body::Function, loop_vars::Union{PyObject, Array{Any}, Array{PyObject}};
        parallel_iterations=10, kwargs...)
    @warn "TensorArray must be initialized (writedown at index 1) outside" maxlog=1
    if isa(loop_vars, PyObject)
        lv = [loop_vars]
    else
        lv = loop_vars
    end
    if get_dtype(loop_vars[1])!=Int32
        error("Loop index must be Int32, got $(get_dtype(loop_vars[1]))")
    end

    res = tf.while_loop(condition, body, loop_vars=lv; parallel_iterations=parallel_iterations, kwargs...)

    if isa(loop_vars, PyObject)
        return res[1]
    else
        return res
    end
end

function cond(condition::Union{PyObject,Function}, fn1::Function, fn2::Function, args...;kwargs...)
    tf.cond(condition, fn1, fn2, args...;kwargs...)
end

function has_gpu()
    s = tf.test.gpu_device_name()
    if length(s)==0
        return false
    else
        return true
    end
end

function stop_gradient(o::PyObject, args...;kwargs...)
    tf.stop_gradient(o, args...;kwargs...)
end
