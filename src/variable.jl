import Base:lastindex, getindex
export
constant,
Variable,
cell,
set_shape,
get_dtype,
get_variable,
placeholder,
variable_scope,
gradients,
gradient_magnitude,
hessian,
constant_initializer,
glorot_normal_initializer,
random_normal_initializer,
random_uniform_initializer,
truncated_normal_initializer,
uniform_unit_scaling_initializer, 
variance_scaling_initializer,
sym,
spd,
tensor,
convert_to_tensor,
hessian_vector,
TensorArray,
gradient_checkpointing,
zeros_like,
ones_like,
gradients_colocate,
is_variable,
jacobian

"""
    constant(value; kwargs...)

Constructs a non-trainable tensor from `value`.
"""
function constant(value; kwargs...)
    if isa(value, PyObject)
        return value
    end
    if isa(value, Char)
        value = string(value)
    end
    kwargs = jlargs(kwargs)
    if !(:dtype in keys(kwargs))
        kwargs[:dtype] = DTYPE[eltype(value)]
    end
    tf.constant(value; kwargs...)
end

"""
    Variable(initial_value;kwargs...)

Constructs a trainable tensor from `value`. 
"""
function Variable(initial_value;kwargs...)
    kwargs = jlargs(kwargs)
    if !(:dtype in keys(kwargs))
        kwargs[:dtype] = DTYPE[eltype(initial_value)]
    end
    tf.Variable(initial_value; kwargs...)
end

"""
    cell(arr::Array, args...;kwargs...)

Construct a cell tensor. 
# Example
```julia-REPL
julia> r = cell([[1.],[2.,3.]])
julia> run(sess, r[1])
1-element Array{Float32,1}:
 1.0
julia> run(sess, r[2])
2-element Array{Float32,1}:
 2.0
 3.0
```
"""
function cell(arr::Array, args...;kwargs...)
    kwargs = jlargs(kwargs)
    if !(:dtype in keys(kwargs))
        kwargs[:dtype] = DTYPE[eltype(arr[1])]
    end
    tf.ragged.constant(arr, args...;kwargs...)
end

"""
    copy(o::PyObject)

Creates a tensor that has the same value that is currently stored in a variable.

!!! note
    The output is a graph node that will have that value when evaluated. Any time you evaluate it, it will grab the current value of `o`. 
"""
function Base.:copy(o::PyObject)
    return tf.identity(o)
end

function get_variable(name::String; kwargs...)
    kwargs = jlargs(kwargs)
    tf.compat.v1.get_variable(name;kwargs...)
end


"""
    get_variable(o::Union{PyObject, Bool, Array{<:Number}}; 
        name::Union{String, Missing} = missing, 
        scope::String = "")

Creates a new variable with initial value `o`. If `name` exists, `get_variable` returns the variable instead of create a new one.
"""
function get_variable(o::Union{PyObject, Bool, Number, Array{<:Number}}; 
    name::Union{String, Missing} = missing, 
    scope::String = "")
    local v
    o = constant(o)
    if ismissing(name)
        name = "unnamed_"*randstring(10)
    end
    variable_scope(scope) do 
        v = tf.compat.v1.get_variable(name = name, initializer=o, dtype=DTYPE[get_dtype(o)])
    end
    return v
end


"""
    get_variable(dtype::Type;
    shape::Union{Array{<:Integer}, NTuple{N, <:Integer}}, 
    name::Union{Missing,String} = missing
    scope::String = "")

Creates a new variable with initial value `o`. If `name` exists, `get_variable` returns the variable instead of create a new one.
"""
function get_variable(dtype::Type;
     shape::Union{Array{<:Integer}, NTuple{N, <:Integer}}, 
     name::Union{Missing,String} = missing,
     scope::String = "") where N
    local v
    dtype = DTYPE[dtype]
    if ismissing(name)
        name = "unnamed_"*randstring(10)
    end
    variable_scope(scope) do 
        v = tf.compat.v1.get_variable(name = name, shape=shape, dtype = dtype)
    end
    return v
end

get_variable(dtype::Type,
    shape::Union{Array{<:Integer}, NTuple{N, <:Integer}}, 
    name::Union{Missing,String} = missing,
    scope::String = "") where N = get_variable(dtype, shape=shape, name=name, scope = scope)

"""
    placeholder(dtype::Type; kwargs...)

Creates a placeholder of the type `dtype`.
# Example 
```julia
a = placeholder(Float64, shape=[20,10])
b = placeholder(Float64, shape=[]) # a scalar 
c = placeholder(Float64, shape=[nothing]) # a vector
```
"""
function placeholder(dtype::Type; kwargs...)
    dtype = DTYPE[dtype]
    kwargs = Dict{Any,Any}(kwargs)
    if !(:shape in keys(kwargs))
        kwargs[:shape] = []
    end
    tf.compat.v1.placeholder(dtype;kwargs...)
end

"""
    placeholder(o::Union{Number, Array, PyObject}; kwargs...)

Creates a placeholder of the same type and size as `o`. `o` is the default value. 
"""
function placeholder(o::Union{Number, Array, PyObject}; kwargs...)
    o = convert_to_tensor(o; kwargs...)
    tf.compat.v1.placeholder_with_default(o, shape=size(o))
end

function variable_scope(f, name_or_scope; reuse=AUTO_REUSE, kwargs...)
    @pywith tf.variable_scope(name_or_scope;reuse = reuse, kwargs...) begin
        f()
    end
end

function set_shape(o::PyObject, shape)
    o.set_shape(shape)
    o
end

function get_dtype(o::PyObject)
    for (k,v) in DTYPE
        if occursin(string(v),string(o.dtype))
            return k
        end
    end
    return nothing
end

Base.:eltype(o::PyObject) = get_dtype(o)

@deprecate get_shape size
get_shape(o::PyObject, i::Union{Int64,Nothing}) = size(o,i)

# compatible with Julia size
function PyCall.:size(o::PyObject, i::Union{Int64, Nothing}=nothing)
    d = o.shape.dims
    if d===nothing
        return nothing
    end
    if length(d)==0
        if i!=nothing
            return 1
        else
            return ()
        end
    else
        s = o.get_shape().as_list()
        s = [isnothing(x) ? x : Int64(x) for x in s]
        if i==nothing
            return Tuple(s)
        elseif 0<i<=length(s)
            return s[i]
        else
            return 1
        end
    end
end

function PyCall.:length(o::PyObject) 
    # If `o.shape.dims` is invalid, it is not a TensorFlow object.
    if !hasproperty(o, :graph)
        return PyCall.@pycheckz ccall((PyCall.@pysym :PySequence_Size), Int, (PyCall.PyPtr,), o)
    end
    if any(isnothing.(size(o)))
        return nothing
    else
        return prod(size(o))
    end
end

function Base.:ndims(o::PyObject)
    return length(size(o))
end

@doc raw"""
    gradients(ys::PyObject, xs::PyObject; kwargs...)

Computes the gradients of `ys` w.r.t `xs`. 

- If `ys` is a scalar, `gradients` returns the gradients with the same shape as `xs`.
- If `ys` is a vector, `gradients` returns the Jacobian $\frac{\partial y}{\partial x}$

!!! note
    The second usage is not suggested since `ADCME` adopts reverse mode automatic differentiation. 
    Although in the case `ys` is a vector and `xs` is a scalar, `gradients` cleverly uses forward mode automatic differentiation,
    it requires that the second order gradients are implemented for relevant operators. 
"""
function gradients(ys::PyObject, xs::PyObject; kwargs...)
    s1 = size(ys)
    s2 = size(xs)
    kwargs = jlargs(kwargs)
    if isnothing(s1) && isnothing(s2)
        error("s1, s2 should be rank 0, 1, 2")
    end
    if length(s1)==0
        ret = tf.gradients(ys, xs; kwargs...)
        if !isnothing(ret) && !isnothing(ret[1])
            return tf.convert_to_tensor(ret[1])
        else
            return nothing
        end
    elseif length(s1)==1 && length(s2)==0
        return gradients10(ys, xs; kwargs...)
    elseif length(s1)==1 && length(s2)==1
        return gradients11(ys, xs)
    elseif length(s1)==2 && length(s2)==0
        grad = gradients(vec(ys), xs; kwargs...)
        reshape(grad, s1...)
    else
        error("gradients: Invalid argument")
    end 
end

function gradients(ys::PyObject, xs::Array{T}; kwargs...) where T <: Union{Any, PyObject}
    zs = Array{PyObject}(undef, length(xs))
    for i = 1:length(zs)
        zs[i] = gradients(ys, xs[i]; kwargs...)
    end
    zs
end

function gradients(ys::Array{T}, xs::PyObject; kwargs...) where T <: Union{Any, PyObject}
    zs = Array{PyObject}(undef, length(ys))
    for i = 1:length(ys)
        zs[i] = gradients(ys[i], xs; kwargs...)
    end
    zs
end

function gradients10(ys::PyObject, xs::PyObject; kwargs...)
    kwargs = jlargs(kwargs)
    try
        u = Variable(rand(length(ys)), trainable=false)
        g = tf.gradients(ys, xs, grad_ys=u; kwargs...)
        g==nothing && (return nothing)
        r = tf.gradients(g[1], u; unconnected_gradients="zero", kwargs...)[1]
    catch
        gradients_v(ys, xs; kwargs...)
    end
end

function gradients_v(ys::PyObject, xs::PyObject;kwargs...)
    kwargs = jlargs(kwargs)
    if length(size(ys))!=1
        error("ys should be a n dimensional vector function")
    end
    if length(size(xs))!=0
        error("xs should be a scalar")
    end
    n = length(ys)
    function condition(i, ta)
        i<=n
    end
    function body(i,ta)
        ta = write(ta, i, tf.gradients(ys[i], xs, unconnected_gradients="zero", kwargs...)[1])
        i+1, ta
    end
    ta = TensorArray(n)
    i = constant(1, dtype=Int32)
    _, ta = while_loop(condition, body, [i,ta], parallel_iterations=10)
    out = stack(ta)
end

# https://stackoverflow.com/questions/50244270/computing-jacobian-matrix-in-tensorflow
function gradients11(ys::PyObject, xs::PyObject; kwargs...)
    n = size(ys,1)
    function condition(i, ta)
        i <= n 
    end
    function body(i, ta)
        ta = write(ta, i, tf.convert_to_tensor(gradients(ys[i], xs; unconnected_gradients="zero")))
        i+1, ta
    end
    ta = TensorArray(n)
    i = constant(1, dtype=Int32)
    _, ta = while_loop(condition, body, [i,ta],parallel_iterations=10)
    stack(ta)
end

"""
    gradients_colocate(loss::PyObject, xs::Union{PyObject, Array{PyObject}}, args...;use_locking::Bool = true, kwargs...)

Computes the gradients of a **scalar** loss function `loss` with respect to `xs`. The gradients are colocated with respect to the forward pass. 
This function is usually in distributed computing. 
"""
function gradients_colocate(loss::PyObject, xs::Union{PyObject, Array{PyObject}}, args...;use_locking::Bool = true, kwargs...)
    flag = false
    if isa(xs, PyObject)
        xs = [xs]
        flag = true
    end
    opt = tf.train.Optimizer(use_locking, "default_optimizer_"*randstring(8))
    grads_and_vars = opt.compute_gradients(
        loss, var_list=xs,
        colocate_gradients_with_ops=true)
    grads = [x[1] for x in grads_and_vars]
    flag && (grads = grads[1])
    return grads
end

@doc raw"""
    jacobian(y::PyObject, x::PyObject)

Computes the Jacobian matrix 
$$J_{ij} = \frac{\partial y_i}{\partial x_j}$$
"""
function jacobian(y::PyObject, x::PyObject)
    @assert length(size(y))==1
    @assert length(size(x))==1
    tfj = pyimport("tensorflow.python.ops.parallel_for.gradients")
    tfj.jacobian(y, x)
end

function hessian_vector(f, xs, v; kwargs...)
    kwargs = jlargs(kwargs)
    gradients_impl = pyimport("tensorflow.python.ops.gradients_impl")
    gradients_impl._hessian_vector_product(f, [xs], [v]; kwargs...)[1]
end

@doc raw"""
    hessian(ys::PyObject, xs::PyObject; kwargs...)

`hessian` computes the hessian of a scalar function f with respect to vector inputs xs. 

# Example 
```julia
x = constant(rand(10))
y = 0.5 * sum(x^2)
o = hessian(y, x)

sess = Session(); init(sess)
run(sess, o) # should be an identity matrix
```
"""
function hessian(ys::PyObject, xs::PyObject; kwargs...)
    if length(size(ys))==0 && length(size(xs))==1
        return tf.hessians(ys, xs)[1]
    end
    kwargs = jlargs(kwargs)
    s1 = size(ys)
    s2 = size(xs)
    if s1==nothing || s2 == nothing || (length(s1)!=0 && length(s2)!=1)
        error("Invalid input arguments")
    end
    h = tf.gradients(ys, xs; kwargs...)
    if h==nothing
        return nothing
    else
        h = h[1]
    end
    vs = Array{PyObject}(undef, s2[1])
    for i = 1:s2[1]
        # verbose && (@info "_hessian... $i/$(s2[1])")
        vs[i] = gradients(get(h,i-1), xs)
        if isnothing(vs[i])
            vs[i] = constant(zeros(get_dtype(ys), s2[1]))
        end
    end
    stack(vs, dims=1)
end

# initializers
constant_initializer(args...;kwargs...) = tf.constant_initializer(args...;kwargs...)
glorot_normal_initializer(args...;kwargs...) = tf.glorot_normal_initializer(args...;kwargs...)
glorot_uniform_initializer(args...;kwargs...) = tf.glorot_uniform_initializer(args...;kwargs...)
random_normal_initializer(args...;kwargs...) = tf.random_normal_initializer(args...;kwargs...)
random_uniform_initializer(args...;kwargs...) = tf.random_uniform_initializer(args...;kwargs...)
truncated_normal_initializer(args...;kwargs...) = tf.truncated_normal_initializer(args...;kwargs...)
uniform_unit_scaling_initializer(args...;kwargs...) = tf.uniform_unit_scaling_initializer(args...;kwargs...)
variance_scaling_initializer(args...;kwargs...) = tf.uniform_unit_scaling_initializer(args...;kwargs...)

#--------------------------------------------------------------------------------------------------------
# Indexing

function PyCall.:lastindex(o::PyObject)
    return size(o,1)
end

function PyCall.:lastindex(o::PyObject, i::Int64)
    return size(o,i)
end

# rank 1 tensor
function getindex(o::PyObject, r::Union{Colon, Array{Bool,1}, BitArray{1}, Array{Int64,1},UnitRange{Int64}, StepRange{Int64, Int64}})
    if typeof(r)==Colon
        return vec(o)
    elseif typeof(r)==Array{Bool,1} || typeof(r)==BitArray{1}
        return getindex(o, findall(r))
    elseif typeof(r)==UnitRange{Int64} || typeof(r)==StepRange{Int64, Int64}
        return getindex(o, collect(r))
    elseif typeof(r)==Array{Int64,1}
        return tf.gather(o, r.-1)
    end
end

function getindex(o::PyObject, i::PyObject)
    return tf.gather(o, i-1)
end

# rank 2 tensor
# drawback: only access the 1st dimension of a tensor
function getindex(o::PyObject, i1::Union{Int64, Colon, Array{Bool,1},BitArray{1}, Array{Int64,1},UnitRange{Int64}, StepRange{Int64, Int64}}, c::Colon)
    if typeof(i1)==Colon
        return o
    elseif typeof(i1)==Array{Bool,1}|| typeof(i1)==BitArray{1}
        return getindex(o, findall(i1), c)
    elseif typeof(i1)==UnitRange{Int64} || typeof(i1)==StepRange{Int64, Int64}
        i1 = (i1|>collect) 
    end
    tf.gather(o, i1.-1)
end

# less efficient
function _to_range_array(o::PyObject, 
    i::Union{Int64, Colon, Array{Bool,1},BitArray{1},  UnitRange{Int64}, Array{Int64,1}, StepRange{Int64, Int64}},
    d::Int64)
    if typeof(i)==Int64
        return [i]
    elseif typeof(i)==Colon 
        if isnothing(size(o,d))
            error(ArgumentError("Dimension $d of `o` is `nothing`. You need to set a concrete dimension, e.g., using `set_shape`"))
        end
        return collect(1:size(o,d))
    elseif typeof(i)<:StepRange || typeof(i)<:UnitRange
        return collect(i)
    elseif typeof(i)==Array{Bool,1}|| typeof(i)==BitArray{1}
        return findall(i)
    else
        return i
    end
end

function getindex(o::PyObject, i1::Union{Int64, Colon, Array{Bool,1},BitArray{1}, Array{Int64,1},UnitRange{Int64}, StepRange{Int64, Int64}}, 
    i2::Union{Int64, Array{Bool,1},Array{Int64,1},UnitRange{Int64}, StepRange{Int64, Int64}})
    sdim1 = typeof(i1)==Int64
    sdim2 = typeof(i2)==Int64
    if typeof(i1)==Int64 && typeof(i2)==Int64
        return squeeze(tf.strided_slice(o, (i1[1]-1,i2[1]-1),(i1[1],i2[1]),(1,1)), dims=(1,2))
    end
    i1 = _to_range_array(o, i1, 1)
    i2 = _to_range_array(o, i2, 2)
    temp = Base.Iterators.product(i2,i1)|>collect
    indices = [vec([x[2] for x in temp]) vec([x[1] for x in temp])] .- 1
    p = tf.gather_nd(o, indices)
    p = tf.reshape(p', (length(i1), length(i2)))

    if sdim1 
        return squeeze(p, dims=1)
    elseif sdim2
        return squeeze(p, dims=2)
    else
        return p
    end
end

function getindex(o::PyObject, i1, i2, i3)
    sdims = Int64[]
    if isa(i1, Int64)
         push!(sdims, 1)
         i1 = i1:i1
    end
    if isa(i2, Int64)
        push!(sdims, 2)
        i2 = i2:i2
   end
    if isa(i3, Int64)
        push!(sdims, 3)
        i3 = i3:i3
    end
    res = getindex(o, i1, i2, i3)
    squeeze(res, dims=sdims)
end

function getindex(o::PyObject, 
    i1::Union{Colon, Array{Bool,1},BitArray{1}, Array{Int64,1},UnitRange{Int64}, StepRange{Int64, Int64}}, 
    i2::Union{Colon, Array{Bool,1},BitArray{1}, Array{Int64,1},UnitRange{Int64}, StepRange{Int64, Int64}}, 
    i3::Union{Colon, Array{Bool,1},BitArray{1}, Array{Int64,1},UnitRange{Int64}, StepRange{Int64, Int64}})
    if length(size(o))!=3
        error(DimensionMismatch("input tensor is $(length(size(o))) dimensional, but expected 3"))
    end
    i1 = _to_range_array(o, i1, 1)
    i2 = _to_range_array(o, i2, 2)
    i3 = _to_range_array(o, i3, 3)
    idx = Int64[]
    for i = 1:length(i1)
        for j = 1:length(i2)
            for k = 1:length(i3)
                ii = i3[k] + (i2[j]-1)*size(o,3) + (i1[i]-1)*size(o,2)*size(o,3)
                push!(idx, ii)
            end
        end
    end
    p = reshape(o, (-1,))[idx]
    reshape(p, (length(i1), length(i2), length(i3)))
end




function Base.:getindex(o::PyObject, i::PyObject, j::Union{Int64, Colon, Array{Bool,1},BitArray{1}, Array{Int64,1},UnitRange{Int64}, StepRange{Int64, Int64}})
    flag = false
    if isa(j, Colon) 
        return o[i]
    end
    if length(size(i))!=0 || !(get_dtype(i)<:Integer)
        error(ArgumentError("Only integer `i` is supported"))
    end
    if isnothing(size(o,2))
        error(ArgumentError("Dimension 2 of `o` is `nothing`. You need to set a concrete dimension, e.g., using `set_shape`"))
    end
    if isa(j, Integer)
        flag = true 
    end
    j = _to_range_array(o, j, 2)
    idx = (i-1)*size(o,2) + j
    ret = tf.reshape(o, (-1,))[idx]
    if flag
        return get(ret, 0)
    else
        return ret
    end
end

function Base.:getindex(o::PyObject, i::Union{Int64, Colon, Array{Bool,1},BitArray{1}, Array{Int64,1},UnitRange{Int64}, StepRange{Int64, Int64}}, j::PyObject)
    flag = false
    if length(size(j))!=0 || !(get_dtype(j)<:Integer)
        error(ArgumentError("Only integer `j` is supported"))
    end
    if isnothing(size(o,2))
        error(ArgumentError("Dimension 2 of `o` is `nothing`. You need to set a concrete dimension, e.g., using `set_shape`"))
    end
    if isa(i, Integer)
        flag = true 
    end
    i = _to_range_array(o, i, 1)
    idx = (i .- 1)*size(o,2) + j
    ret = tf.reshape(o, (-1,))[idx]
    if flag
        return get(ret, 0)
    else
        return ret
    end
end

#-------------------------------------------------------------------------------------------------------

# https://stackoverflow.com/questions/46718356/tensorflow-symmetric-matrix
function sym(o::Union{Array{<:Real}, PyObject})
    convert_to_tensor(1/2 * (o + o'))
end

function spd(o::Union{Array{<:Real}, PyObject})
    if length(size(o))!=2 || size(o,1)!=size(o,2)
        error("Input `o` must be a square matrix")
    end
    convert_to_tensor(o * o')
end


"""
    tensor(v::Array{T,2}; dtype=Float64, sparse=false) where T
"""
function tensor(v::Array{T,1}; dtype=Float64, sparse=false) where T
    local ret
    N = length(v)
    ret = Variable(zeros(dtype, N), trainable=false)
    for i = 1:N
        if sparse && isa(v[i], Number) && v[i]≈0
            continue
        end
        if isa(v[i], Number)
            v[i] = dtype(v[i])
        end
        ret = scatter_add(ret, i, v[i])
    end
    ret
end

"""
    tensor(v::Array{T,2}; dtype=Float64, sparse=false) where T
    
Convert a generic array `v` to a tensor. For example, 
```julia
v = [0.0 constant(1.0) 2.0
    constant(2.0) 0.0 1.0]
u = tensor(v)
```
`u` will be a ``2\\times 3`` tensor. 
!!! note 
    This function is expensive. Use with caution.
"""
function tensor(v::Array{T,2}; dtype=Float64, sparse=false) where T
    local ret
    M, N = size(v)
    ret = Variable(zeros(dtype, M, N), trainable=false)
    for i = 1:M
        for j = 1:N
            if sparse && isa(v[i,j], Number) && v[i,j]≈0
                continue
            end
            if isa(v[i,j], Number)
                v[i,j] = dtype(v[i,j])
            end
            ret = scatter_add(ret, i, j, v[i, j])
        end
    end
    ret
end

"""
    TensorArray(size_::Int64=0, args...;kwargs...)

Constructs a tensor array for [`while_loop`](@ref).  
"""
function TensorArray(size_::Int64=0, args...;kwargs...)
    kwargs = jlargs(kwargs)
    if !(haskey(kwargs, :dtype))
        kwargs[:dtype] = tf.float64
    end
    if !haskey(kwargs, :dynamic_size)
        kwargs[:dynamic_size] = false
    end
    if !haskey(kwargs, :clear_after_read)
        kwargs[:clear_after_read] = false
    end
    kwargs[:size] = size_
    tf.TensorArray(args...;kwargs...)
end

""" 
    read(ta::PyObject, i::Union{PyObject,Integer})

Reads data from [`TensorArray`](@ref) at index `i`.
"""
function Base.:read(ta::PyObject, i::Union{PyObject,Integer})
    ta.read(i-1)
end

""" 
    write(ta::PyObject, i::Union{PyObject,Integer}, obj)

Writes data `obj` to [`TensorArray`](@ref) at index `i`.
"""
function Base.:write(ta::PyObject, i::Union{PyObject,Integer}, obj::PyObject)
    ta.write(i-1, obj)
end

Base.:write(ta::PyObject, i::Union{PyObject,Integer}, obj::Union{Array{<:Real}, Real}) = write(ta, i, constant(obj))

"""
    convert_to_tensor(o::Union{PyObject, Number, Array{T}, Missing, Nothing}; dtype::Union{Type, Missing}=missing) where T<:Number
    convert_to_tensor(os::Array, dtypes::Array)

Converts the input `o` to tensor. If `o` is already a tensor and `dtype` (if provided) is the same as that of `o`, the operator does nothing.
Otherwise, `convert_to_tensor` converts the numerical array to a constant tensor or casts the data type.
`convert_to_tensor` also accepts multiple tensors. 

# Example
```julia
convert_to_tensor([1.0, constant(rand(2)), rand(10)], [Float32, Float64, Float32])
```
"""
function convert_to_tensor(o::Union{PyObject, Number, Array{T}, Missing, Nothing}; 
    dtype::Union{Type, Missing}=missing) where T<:Number
    if ismissing(o) || isnothing(o)
        return o
    end
    if isa(o, PyObject)
        if ismissing(dtype) || dtype==eltype(o)
            return o
        else
            return cast(o, dtype)
        end
    else
        if !ismissing(dtype)
            return constant(o, dtype=dtype)
        else
            return constant(o)
        end
    end
end

function convert_to_tensor(os::Array, dtypes::Array)
    [convert_to_tensor(o, dtype=d) for (o, d) in zip(os, dtypes)]
end


"""
    gradient_checkpointing(type::String="speed")

Uses checkpointing scheme for gradients. 
- 'speed':  checkpoint all outputs of convolutions and matmuls. these ops are usually the most expensive,
    so checkpointing them maximizes the running speed
    (this is a good option if nonlinearities, concats, batchnorms, etc are taking up a lot of memory)
- 'memory': try to minimize the memory usage
    (currently using a very simple strategy that identifies a number of bottleneck tensors in the graph to checkpoint)
- 'collection': look for a tensorflow collection named 'checkpoints', which holds the tensors to checkpoint
"""
function gradient_checkpointing(type::String="speed")
    pyfile = "$(ADCME.LIBDIR)/memory_saving_gradients.py"
    if !isfile(pyfile)
        @info "Downloading memory_saving_gradients.py..."
        download("https://raw.githubusercontent.com/cybertronai/gradient-checkpointing/master/memory_saving_gradients.py",
                pyfile)
    end
    py"""exec(open($pyfile).read())"""
    gradients_speed = py"gradients_speed"
    gradients_memory = py"gradients_memory"
    gradients_collection = py"gradients_collection"
    if type=="speed"
        tf.__dict__["gradients"] = gradients_speed
    elseif type=="memory"
        tf.__dict__["gradients"] = gradients_memory
    elseif type=="collection"
        tf.__dict__["gradients"] = gradients_collection
    else
        error("ADCME: $type not defined")
    end
    @info "Loaded: $type"
end

@doc raw"""
    gradient_magnitude(l::PyObject, o::Union{Array, PyObject})

Returns the gradient sum 

$$\sqrt{\sum_{i=1}^n \|\frac{\partial l}{\partial o_i}\|^2}$$

This function is useful for debugging the training
"""
function gradient_magnitude(l::PyObject, o::Union{Array, PyObject})
    if isa(o, PyObject)
        o = [o]
    end
    tf.global_norm(gradients(l, o))
end

"""
    zeros_like(o::Union{PyObject,Real, Array{<:Real}}, args...; kwargs...)

Returns a all-zero tensor, which has the same size as `o`.

# Example
```julia
a = rand(100,10)
b = zeros_like(a)
@assert run(sess, b)≈zeros(100,10)
```
"""
function zeros_like(o::Union{PyObject,Real, Array{<:Real}}, args...; kwargs...)
    kwargs = jlargs(kwargs)
    tf.zeros_like(o, args...;kwargs...)
end

"""
    ones_like(o::Union{PyObject,Real, Array{<:Real}}, args...; kwargs...)

Returns a all-one tensor, which has the same size as `o`.

# Example
```julia
a = rand(100,10)
b = ones_like(a)
@assert run(sess, b)≈ones(100,10)
```
"""
function ones_like(o::Union{PyObject,Real, Array{<:Real}}, args...; kwargs...)
    kwargs = jlargs(kwargs)
    tf.ones_like(o, args...;kwargs...)
end

"""
    is_variable(o::PyObject)

Determines whether `o` is a trainable variable.
"""
function is_variable(o::PyObject)
    hasproperty(o, :trainable) && o.trainable
end