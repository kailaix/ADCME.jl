export
reset_default_graph,
get_collection,
add_collection,
enable_eager_execution,
control_dependencies,
has_gpu,
while_loop,
if_else,
stop_gradient,
tensor,
tensorname

# only for eager eager execution
enable_eager_execution() = tf.enable_eager_execution()
Base.:values(o::PyObject) = o.numpy()

"""
    reset_default_graph()

Resets the graph by removing all the operators. 
"""
reset_default_graph() = tf.compat.v1.reset_default_graph()
"""
    get_collection(name::Union{String, Missing})

Returns the collection with name `name`. If `name` is `missing`, returns all the trainable variables.
"""
function get_collection(name::Union{String, Missing}=missing)
    if ismissing(name)
        res = tf.compat.v1.get_collection(TRAINABLE_VARIABLES)
    else
        res = tf.get_default_graph()._collections[name]
    end
    return unique(res)
end

"""
    add_collection(name::String, v::PyObject)

Adds `v` to the collection with name `name`. If `name` does not exist, a new one is created.
"""
function add_collection(name::String, v::PyObject)
    tf.get_default_graph().add_to_collection(name, v)
    nothing
end

"""
    add_collection(name::String, vs::PyObject...)

Adds operators `vs` to the collection with name `name`. If `name` does not exist, a new one is created.
"""
function add_collection(name::String, vs::PyObject...)
    for v in vs
        add_collection(name, v)
    end
    nothing
end

"""
    tensor(s::String)

Returns the tensor with name `s`. See [`tensorname`](@ref).
"""
function tensor(s::String)
    tf.get_default_graph().get_tensor_by_name(s)
end

"""
    tensorname(o::PyObject)

Returns the name of the tensor. See [`tensor`](@ref).
"""
function tensorname(o::PyObject)
    o.name
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
        if isa(kwargs[:axis], Array)
            kwargs[:axis] = Tuple(kwargs[:axis])
        end
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

"""
    bind(op::PyObject, ops...)

Adding operations `ops` to the dependencies of `op`. The function is useful when we want to execute `ops` but `ops` is not 
in the dependency of the final output. For example, if we want to print `i` each time `i` is evaluated
```julia
i = constant(1.0)
op = tf.print(i)
i = bind(i, op)
```
"""
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

function if_else_v1(condition::Union{PyObject}, fn1, fn2, args...;kwargs...)
    fn1_ = ifelse(isa(fn1, Function), fn1, ()->fn1)
    fn2_ = ifelse(isa(fn2, Function), fn2, ()->fn2)
    tf.cond(condition, fn1_, fn2_, args...;kwargs...)
end 

function if_else_v2(condition::PyObject, fn1::Union{Nothing, PyObject, Array}, 
        fn2::Union{Nothing, PyObject, Array})
    fn1 = convert_to_tensor(fn1)
    fn2 = convert_to_tensor(fn2)
    tf.compat.v2.where(condition, fn1, fn2) 
end 


"""
    if_else(condition::Union{PyObject,Array,Bool}, fn1, fn2, args...;kwargs...)

- If `condition` is a scalar boolean, it outputs `fn1` or `fn2` (a function with no input argument or a tensor) based on whether `condition` is true or false.
- If `condition` is a boolean array, if returns `condition .* fn1 + (1 - condition) .* fn2`
"""
function if_else(condition::Union{PyObject,Array,Bool}, fn1, fn2, args...;kwargs...)
    if isa(condition, Array) || isa(condition, Bool)
        condition = convert_to_tensor(condition)
    end
    if isa(condition, Function) || (eltype(condition)<:Bool && length(size(condition))==0)
        if_else_v1(condition, fn1, fn2, args...;kwargs...)
    else
        if_else_v2(condition, fn1, fn2)
    end
end

"""
    has_gpu()

Checks if GPU is available.

!!! note
```
ADCME will use GPU automatically if GPU is available. To disable GPU, set the environment variable `ENV["CUDA_VISIBLE_DEVICES"]=""`
before importing ADCME 
```
"""
function has_gpu()
    s = tf.test.gpu_device_name()
    if length(s)==0
        return false
    else
        return true
    end
end


""" 
    stop_gradient(o::PyObject, args...;kwargs...)

Disconnects `o` from gradients backpropagation. 
"""
function stop_gradient(o::PyObject, args...;kwargs...)
    tf.stop_gradient(o, args...;kwargs...)
end
