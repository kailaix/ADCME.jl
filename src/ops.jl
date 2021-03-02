import Base:*, broadcast, reshape, exp, log, tanh, sum, 
    adjoint, inv, argmax, argmin, ^, max, maximum, min, minimum,
    vec, \, cos, sin, sign, map, prod, reverse
import LinearAlgebra: tr, diag, det, norm, diagm, dot, I, svd, tril, triu
import Statistics: mean, std
import FFTW: fft, ifft
export 
*,
^,
einsum,
sigmoid,
tanh,
mean,
log,
exp,
softplus,
softmax,
softsign,
sum,
relu,
relu6,
squeeze,
adjoint,
diag,
diagm,
det,
inv,
triangular_solve,
argmin,
argmax,
max,
min,
group_assign,
assign,
maximum,
minimum,
cast,
group,
clip,
scatter_add,
scatter_sub,
scatter_update,
stack,
concat,
unstack,
norm,
cvec,
rvec,
vec,
sqrt,
mean,
pad,
leaky_relu,
fft, 
ifft,
I,
svd,
vector,
pmap,
std,
lgamma,
topk,
argsort,
batch_matmul,
dot,
set_shape,
selu, 
elu,
tr,
tril, 
triu,
solve_batch,
swish, hard_sigmoid, hard_swish, concat_elu, concat_hard_swish, concat_relu, fourier,
rollmean, rollsum, rollvar, rollstd,
softmax_cross_entropy_with_logits


@doc raw"""
    batch_matmul(o1::PyObject, o2::PyObject)

Computes `o1[i,:,:] * o2[i, :]` or `o1[i,:,:] * o2[i, :, :]` for each index `i`.
"""
function batch_matmul(o1::PyObject, o2::PyObject)
    flag = false
    if length(size(o2))==2
        flag = true
        o2 = tf.expand_dims(o2, 2)
    end
    if length(size(o1))!=3 || length(size(o2))!=3
        error("The size of o1 or o2 is not valid.")
    end
    out = tf.matmul(o1, o2)
    if flag 
        squeeze(out)
    else
        out 
    end
end

batch_matmul(o1::Array{<:Real}, o2::PyObject) = batch_matmul(constant(o1), o2)
batch_matmul(o1::PyObject, o2::Array{<:Real}) = batch_matmul(o1, constant(o2))
batch_matmul(o1::Array{<:Real}, o2::Array{<:Real}) = batch_matmul(constant(o1), constant(o2))

function PyCall.:*(o1::PyObject, o2::PyObject)
    s1 = size(o1)
    s2 = size(o2)
    if s1==nothing || s2==nothing
        error("o1 and o2 should be tensors of rank 0, 1, 2")
    end
    if length(s1) == 0 || length(s2)==0
        return tf.multiply(o1, o2)
    end
    if length(s1)==2 && length(s2)==2
        return tf.matmul(o1, o2)
    elseif length(s1)==2 && length(s2)==1
        return tf.einsum("nm,m->n", o1, o2)
    elseif length(s1)==2 && length(s2)==0
        return tf.multiply(o1, o2)
    elseif length(s1)==1 && length(s2)==2
        error("[rand 1] x [rank 2] not defined")
    elseif length(s1)==1 && length(s2)==1
        return tf.multiply(o1, o2)
    else
        @warn("Unusual usage of multiplication. Check carefully")
        tf.matmul(o1,o2)
    end
end

Base.:*(o1::PyObject, o2::AbstractArray{<:Real}) = *(o1, constant(Array(o2), dtype=get_dtype(o1)))
Base.:*(o1::AbstractArray{<:Real}, o2::PyObject) = *(constant(Array(o1), dtype=get_dtype(o2)), o2)
Base.:*(o1::Number, o2::PyObject) = *(constant(o1, dtype=get_dtype(o2)), o2)
Base.:*(o1::PyObject, o2::Number) = *(o1, constant(o2, dtype=get_dtype(o1)))

Base.Broadcast.broadcasted(::typeof(*), o1::PyObject, o2::AbstractArray{<:Real}) = tf.multiply(o1, Array(o2))
Base.Broadcast.broadcasted(::typeof(*), o1::AbstractArray{<:Real}, o2::PyObject) = tf.multiply(Array(o1), o2)
Base.Broadcast.broadcasted(::typeof(*), o1::PyObject, o2::PyObject) = tf.multiply(o1, o2)
Base.Broadcast.broadcasted(::typeof(*), o1::PyObject, o2::Number) = tf.multiply(o1, o2)
Base.Broadcast.broadcasted(::typeof(*), o1::Number, o2::PyObject) = tf.multiply(o1, o2)

Base.Broadcast.broadcasted(::typeof(/), o1::PyObject, o2::AbstractArray{<:Real}) = tf.divide(o1, Array(o2))
Base.Broadcast.broadcasted(::typeof(/), o1::AbstractArray{<:Real}, o2::PyObject) = tf.divide(Array(o1), o2)
Base.Broadcast.broadcasted(::typeof(/), o1::PyObject, o2::PyObject) = tf.divide(o1, o2)
Base.Broadcast.broadcasted(::typeof(/), o1::PyObject, o2::Number) = tf.divide(o1, o2)
Base.Broadcast.broadcasted(::typeof(/), o1::Number, o2::PyObject) = tf.divide(o1, o2)

Base.Broadcast.broadcasted(::typeof(+), o1::PyObject, o2::AbstractArray{<:Real}) = o1 + Array(o2)
Base.Broadcast.broadcasted(::typeof(+), o1::AbstractArray{<:Real}, o2::PyObject) = Array(o1) + o2
Base.Broadcast.broadcasted(::typeof(+), o1::PyObject, o2::PyObject) = o1 + o2
Base.Broadcast.broadcasted(::typeof(+), o1::PyObject, o2::Number) = o1 + o2
Base.Broadcast.broadcasted(::typeof(+), o1::Number, o2::PyObject) = o1 + o2

Base.Broadcast.broadcasted(::typeof(-), o1::PyObject, o2::AbstractArray{<:Real}) = o1 - Array(o2)
Base.Broadcast.broadcasted(::typeof(-), o1::AbstractArray{<:Real}, o2::PyObject) = Array(o1) - o2
Base.Broadcast.broadcasted(::typeof(-), o1::PyObject, o2::PyObject) = o1 - o2
Base.Broadcast.broadcasted(::typeof(-), o1::PyObject, o2::Number) = o1 - o2
Base.Broadcast.broadcasted(::typeof(-), o1::Number, o2::PyObject) = o1 - o2


warn_broadcast_pow() = error(".^ is disabled due to eager evaluation. Use ^ instead.")
Base.Broadcast.broadcasted(::typeof(^), o1::PyObject, o2::Union{AbstractArray{<:Real},Number}) = warn_broadcast_pow()
Base.Broadcast.broadcasted(::typeof(^), o1::PyObject, o2::PyObject) = warn_broadcast_pow()
Base.Broadcast.broadcasted(::typeof(^), o1::Union{AbstractArray{<:Real},Number}, o2::PyObject) = warn_broadcast_pow()

function einsum(equation, args...; kwargs...)
    tf.einsum(equation, args...; kwargs...)
end

"""
    reshape(o::PyObject, s::Union{Array{<:Integer}, Tuple{Vararg{<:Integer, N}}}) where N 
    reshape(o::PyObject, s::Integer; kwargs...)
    reshape(o::PyObject, m::Integer, n::Integer; kwargs...)
    reshape(o::PyObject, ::Colon, n::Integer)
    reshape(o::PyObject, n::Integer, ::Colon)

Reshapes the tensor according to row major if the "TensorFlow style" syntax is used; otherwise 
reshaping according to column major is assumed. 

# Example
```julia
reshape(a, [10,5]) # row major 
reshape(a, 10, 5) # column major 
```
"""
function reshape(o::PyObject, s::Union{Array{<:Integer}, Tuple{Vararg{<:Integer, N}}}) where N 
    tf.reshape(o, s)
end

function reshape(o::PyObject, s::Integer; kwargs...)
    if length(size(o))>=2
        return tf.reshape(o', [s]; kwargs...)
    end
    tf.reshape(o, [s]; kwargs...)
end

function reshape(o::PyObject, m::Integer, n::Integer; kwargs...)
    if length(size(o))==1
        return tf.reshape(o, [n,m]; kwargs...)'
    elseif length(size(o))==2
        return tf.reshape(o', [n,m]; kwargs...)'
    end
    tf.reshape(o, [m, n]; kwargs...)
end

reshape(o::PyObject, ::Colon, n::Integer) = reshape(o, -1, n)
reshape(o::PyObject, n::Integer, ::Colon) = reshape(o, n, -1)

"""
    rvec(o::PyObject; kwargs...)

Vectorizes the tensor `o` to a row vector, assuming column major.
"""
function rvec(o::PyObject; kwargs...)
    s = size(o)
    if length(s)==0
        return reshape(o, 1, 1, kwargs...)
    elseif length(s)==1
        return reshape(o, 1, s[1], kwargs...)
    elseif length(s)==2
        return reshape(o, 1, s[1]*s[2], kwargs...)
    else
        error("Invalid argument")
    end
end

"""
    rvec(o::PyObject; kwargs...)

Vectorizes the tensor `o` to a column vector, assuming column major.
"""
function cvec(o::PyObject;kwargs...)
    s = size(o)
    if length(s)==0
        return reshape(o, 1, 1,kwargs...)
    elseif length(s)==1
        return reshape(o, s[1], 1,kwargs...)
    elseif length(s)==2
        return reshape(o, s[1]*s[2], 1,kwargs...)
    else
        error("Invalid argument")
    end
end

"""
    vec(o::PyObject;kwargs...)

Vectorizes the tensor `o` assuming column major. 
"""
function vec(o::PyObject;kwargs...)
    s = size(o)
    if length(s)==0
        return reshape(o, 1, kwargs...)
    elseif length(s)==1
        return o
    elseif length(s)>=2
        return reshape(o, s[1]*s[2])
    end
end

"""
    set_shape(o::PyObject, s::Union{Array{<:Integer}, Tuple{Vararg{<:Integer, N}}}) where N
    set_shape(o::PyObject, s::Integer...)
    
Sets the shape of `o` to `s`. `s` must be the actual shape of `o`. This function is used to convert a 
tensor with unknown dimensions to a tensor with concrete dimensions. 

# Example 
```julia
a = placeholder(Float64, shape=[nothing, 10])
b = set_shape(a, 3, 10)
run(sess, b, a=>rand(3,10)) # OK 
run(sess, b, a=>rand(5,10)) # Error
run(sess, b, a=>rand(10,3)) # Error
```
"""
function set_shape(o::PyObject, s::Union{Array{<:Integer}, Tuple{Vararg{<:Integer, N}}}) where N 
    o.set_shape(s)
    return o 
end

set_shape(o::PyObject, s::Integer...) = set_shape(o, s)


function sigmoid(x::Real)
    return 1/(1+exp(-x))
end

function sigmoid(o::PyObject; kwargs...)
    tf.math.sigmoid(o; kwargs...)
end

relu(x::Real) = max(zero(x), x)
function relu(o::PyObject; kwargs...)
    tf.nn.relu(o; kwargs...)
end


relu6(x::Real) = min(relu(x), one(x)*oftype(x, 6))
relu6(o::PyObject; kwargs...) = tf.nn.relu6(o; kwargs...)


function tan(o::PyObject; kwargs...)
    tf.math.tan(o; kwargs...)
end

function leaky_relu(x::Real, a = oftype(x / 1, 0.2))
    max(a * x, x / one(x))
end

function leaky_relu(o::PyObject; kwargs...)
    tf.nn.leaky_relu(o; kwargs...)
end

function tanh(o::PyObject; kwargs...)
    tf.tanh(o; kwargs...)
end

function selu(x::Real)
    λ = oftype(x / 1, 1.0507009873554804934193349852946)
    α = oftype(x / 1, 1.6732632423543772848170429916717)
    λ * ifelse(x > 0, x / one(x), α * (exp(x) - one(x)))
end

selu(o::PyObject; kwargs...) = tf.nn.selu(o; kwargs...)

elu(x, α = one(x)) = ifelse(x ≥ 0, x / one(x), α * (exp(x) - one(x)))
elu(o::PyObject; kwargs...) = tf.nn.elu(o; kwargs...)

softsign(x::Real) = x / (one(x) + abs(x))
softsign(o::PyObject; kwargs...) = tf.nn.softsign(o; kwargs...)



function argmax(o::PyObject; kwargs...)
    kwargs = jlargs(kwargs)
    tf.argmax(o; kwargs...) + 1
end

function Base.:sqrt(o::PyObject; kwargs...)
    kwargs = jlargs(kwargs)
    tf.sqrt(o)
end

function argmin(o::PyObject; kwargs...)
    kwargs = jlargs(kwargs)
    tf.argmin(o; kwargs...) + 1
end

function max(o1::PyObject, o2::PyObject; kwargs...)
    tf.maximum(o1, o2; kwargs...)
end

function min(o1::PyObject, o2::PyObject; kwargs...)
    tf.minimum(o1, o2; kwargs...)
end

function maximum(o::PyObject; kwargs...)
    kwargs = jlargs(kwargs)
    tf.reduce_max(o; kwargs...) 
end

function minimum(o::PyObject; kwargs...)
    kwargs = jlargs(kwargs)
    tf.reduce_min(o; kwargs...) 
end

function std(o::PyObject; kwargs...)
    kwargs = jlargs(kwargs)
    tf.math.reduce_std(o; kwargs...) 
end

function cast(x::PyObject, dtype::Type;kwargs...)
    dtype = DTYPE[dtype]
    tf.cast(x, dtype; kwargs...)
end

function cast(dtype::Type, x::PyObject;kwargs...)
    dtype = DTYPE[dtype]
    tf.cast(x, dtype; kwargs...)
end

softplus(x::Real) = ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))
function softplus(x;kwargs...)
    tf.math.softplus(x; kwargs...)
end

function log(o::PyObject; kwargs...)
    tf.math.log(o; kwargs...)
end

function exp(o::PyObject; kwargs...)
    tf.exp(o; kwargs...)
end

function cos(o::PyObject; kwargs...)
    tf.cos(o; kwargs...)
end

function sin(o::PyObject; kwargs...)
    tf.sin(o; kwargs...)
end

function sign(o::PyObject; kwargs...)
    tf.sign(o; kwargs...)
end

function softmax(o::PyObject; kwargs...)
    kwargs = jlargs(kwargs)
    tf.math.softmax(o; kwargs...)
end

function sum(o::PyObject; kwargs...)
    kwargs = jlargs(kwargs)
    tf.reduce_sum(o; kwargs...)
end

function mean(o::PyObject; kwargs...)
    kwargs = jlargs(kwargs)
    tf.reduce_mean(o; kwargs...)
end

function prod(o::PyObject; kwargs...)
    kwargs = jlargs(kwargs)
    tf.reduce_prod(o; kwargs...)
end
function squeeze(o::PyObject; kwargs...)
    kwargs = jlargs(kwargs)
    tf.squeeze(o;kwargs...)
end

"""
    pad(o::PyObject, paddings::Array{Int64, 2}, args...; kwargs...)

Pads `o` with values on the boundary. 
# Example 
```julia 
o = rand(3,3)
o = pad(o, [1 4      # first dimension
             2 3])   # second dimension
run(sess, o)
```
Expected:
```
8×8 Array{Float64,2}:
 0.0  0.0  0.0       0.0       0.0       0.0  0.0  0.0
 0.0  0.0  0.250457  0.666905  0.823611  0.0  0.0  0.0
 0.0  0.0  0.23456   0.625145  0.646713  0.0  0.0  0.0
 0.0  0.0  0.552415  0.226417  0.67802   0.0  0.0  0.0
 0.0  0.0  0.0       0.0       0.0       0.0  0.0  0.0
 0.0  0.0  0.0       0.0       0.0       0.0  0.0  0.0
 0.0  0.0  0.0       0.0       0.0       0.0  0.0  0.0
 0.0  0.0  0.0       0.0       0.0       0.0  0.0  0.0
```
"""
function pad(o::Union{Array{<:Real}, PyObject}, paddings::Array{Int64, 2}, args...; kwargs...)
    o = constant(o)
    tf.pad(o, paddings, args...; kwargs...)
end

function assign(o::PyObject,value, args...; kwargs...)
    tf.compat.v1.assign(o, value, args...;kwargs...)
end

function group(args...; kwargs...)
    tf.group(args...; kwargs...)
end

assign(o::Array{PyObject}, value::Array, args...;kwargs...) = group_assign(o, value, args...; kwargs...)


@deprecate group_assign assign
function group_assign(os::Array{PyObject}, values, args...; kwargs...)
    ops = Array{PyObject}(undef, length(os))
    for i = 1:length(os)
        ops[i] = tf.compat.v1.assign(os[i], values[i], args...; kwargs...)
    end
    ops
end


"""
    adjoint(o::PyObject; kwargs...) 

Returns the conjugate adjoint of `o`. 
When the dimension of `o` is greater than 2, only the last two dimensions are permuted, i.e., `permutedims(o, [1,2,...,n,n-1])`
"""
function adjoint(o::PyObject; kwargs...) 
    if length(size(o))==0
        return o 
    elseif length(size(o))==1
        return rvec(o)
    else
        return tf.linalg.adjoint(o; kwargs...)
    end
end
diagm(o::PyObject; kwargs...) = tf.linalg.diag(o; kwargs...)
diag(o::PyObject; kwargs...) = tf.linalg.diag_part(o; kwargs...)
det(o::PyObject; kwargs...) = tf.linalg.det(o; kwargs...)
inv(o::PyObject; kwargs...) = tf.linalg.inv(o; kwargs...)

@doc raw"""
    solve_batch(A::Union{PyObject, Array{<:Real, 2}}, rhs::Union{PyObject, Array{<:Real,2}})

Solves $$Ax = b$$ for a batch of right hand sides. 

- `A`: a $m\times n$ matrix, where $m\geq n$
- `rhs`: a $n_b\times m$ matrix. Each row is a new right hand side to solve. 

The returned value is a $n_b\times n$ matrix. 

# Example
```julia
a = rand(10,5)
b = rand(100, 10)
sol = solve_batch(a, b)
@assert run(sess, sol) ≈ (a\b')'
```

!!! note 
    Internally, the matrix $A$ is factorized first and then the factorization is used to solve multiple right hand side.
"""
function solve_batch(A::Union{PyObject, Array{<:Real, 2}}, rhs::Union{PyObject, Array{<:Real,2}})
    solve_batched_rhs_ = load_system_op("solve_batched_rhs"; multiple=false)
    a,rhs = convert_to_tensor([A,rhs], [Float64,Float64])
    sol = solve_batched_rhs_(a,rhs)
    if size(a, 2)!=nothing && size(rhs,1) != nothing
        sol = set_shape(sol, (size(rhs,1), size(a,2)))
    end
    return sol 
end

function solve(matrix, rhs; kwargs...)
    rhs = constant(rhs)
    matrix = constant(matrix)
    flag = false
    if length(size(rhs))==1
        flag = true
        rhs = reshape(rhs, size(rhs, 1), 1)
    end
    if size(matrix,1)==size(matrix,2)
        ret = tf.linalg.solve(matrix, rhs; kwargs...)
    else
        # @show matrix, rhs
        ret = tf.linalg.lstsq(matrix, rhs;kwargs...)
    end
    if flag
        ret = squeeze(ret, dims=2)
    end
    return ret
end

Base.:\(o1::PyObject, o2::PyObject) = solve(o1, o2)
Base.:\(o1::PyObject, o2::Array) = solve(o1, o2)
Base.:\(o1::Array, o2::PyObject) = solve(o1, o2)


function triangular_solve(matrix, rhs; kwargs...)
    flag = false
    if length(size(rhs))==1
        flag = true
        rhs = reshape(rhs, size(rhs, 1), 1)
    end
    ret = tf.linalg.triangular_solve(matrix, rhs; kwargs...)
    if flag
        ret = squeeze(ret, dims=2)
    end
    return ret
end

# reference: https://blog.csdn.net/LoseInVain/article/details/79638183
function concat(o::Union{PyObject,Array{PyObject}}, args...;kwargs...)
    if isa(o, PyObject)
        @warn "Only one input is consumed by concat" maxlog=1
        return o
    end
    kwargs = jlargs(kwargs)
    if length(size(o[1]))==0
        return tf.stack(o)
    end
    tf.concat(o, args...; kwargs...)
end

function stack(o::Array{PyObject}, args...;kwargs...)
    kwargs = jlargs(kwargs)
    tf.stack(o, args...; kwargs...)
end

Base.:vcat(args::PyObject...) = concat([args...],0)
function Base.:hcat(args::PyObject...)
    if length(size(args[1]))>=2 
        concat([args...],1) 
    elseif length(size(args[1]))==1
        stack([args...],dims=2)
    else 
        vcat(args...)'
    end
end

"""
    stack(o::PyObject)

Convert a `TensorArray` `o` to a normal tensor. The leading dimension is the size of the tensor array. 
"""
function stack(o::PyObject)
    o.stack()
end

function unstack(o::PyObject, args...;kwargs...)
    kwargs = jlargs(kwargs)
    tf.unstack(o, args...; kwargs...)
end



function _jlindex2indices(len::Int64, 
    indices::Union{PyObject, Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}})
    if isa(indices, BitArray{1}) || isa(indices, Array{Bool,1})
        indices = findall(indices)
    elseif isa(indices, UnitRange{Int64}) || isa(indices, StepRange{Int64, Int64})
        indices = collect(indices)
    elseif isa(indices, Colon)
        indices = Array(1:len)
    elseif isa(indices, Int64)
        indices = [indices]
    end
    if isa(indices, PyObject)
        return indices - 1
    else
        return constant(indices .- 1)
    end
end


for (op1, op2) = [(:_scatter_add, :tensor_scatter_nd_add), (:_scatter_sub, :tensor_scatter_nd_sub), 
                    (:_scatter_update,:tensor_scatter_nd_update)]
    @eval begin 
        function $op1(ref::PyObject, 
            indices::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
            updates::Union{Array{<:Real}, Real, PyObject})
            updates = convert_to_tensor(updates, dtype=get_dtype(ref))
            @assert length(size(updates)) <= 1
            @assert length(size(ref))==1
            if length(size(updates))==0
                updates = reshape(updates, (-1,))
            end
            indices = _jlindex2indices(length(ref),indices)
            indices = reshape(indices, (-1,1))
            # @info ref, indices, updates
            tf.$op2(ref, indices, updates)
        end
    end
end

"""
    scatter_update(a::PyObject, 
        indices::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
        updates::Union{Array{<:Real}, Real, PyObject})

Updates array `a`
```
a[indices] = updates
```

# Example
Julia:
```julia
A[[1;2;3]] = rand(3)
A[2] = 1.0
```

ADCME:
```
A = scatter_update(A, [1;2;3], rand(3))
A = scatter_update(A, 2, 1.0)
```
"""
scatter_update(a::PyObject, 
    indices::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
    updates::Union{Array{<:Real}, Real, PyObject}) = _scatter_update(a, indices, updates)

"""
    scatter_sub(a::PyObject, 
        indices::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
        updates::Union{Array{<:Real}, Real, PyObject})

Updates array `a`
```
a[indices] -= updates
```

# Example
Julia:
```julia
A[[1;2;3]] -= rand(3)
A[2] -= 1.0
```

ADCME:
```
A = scatter_sub(A, [1;2;3], rand(3))
A = scatter_sub(A, 2, 1.0)
```
"""
scatter_sub(a::PyObject, 
    indices::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
    updates::Union{Array{<:Real}, Real, PyObject}) = _scatter_sub(a, indices, updates)


"""
    scatter_add(a::PyObject, 
        indices::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
        updates::Union{Array{<:Real}, Real, PyObject})

Updates array `add`
```
a[indices] += updates
```

# Example
Julia:
```julia
A[[1;2;3]] += rand(3)
A[2] += 1.0
```

ADCME:
```
A = scatter_add(A, [1;2;3], rand(3))
A = scatter_add(A, 2, 1.0)
```
"""
scatter_add(a::PyObject, 
    indices::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
    updates::Union{Array{<:Real}, Real, PyObject}) = _scatter_add(a, indices, updates)


for (op1, op2) = [(:scatter_update2, :scatter_update), (:scatter_add2, :scatter_add), 
    (:scatter_sub2,:scatter_sub)]
    @eval begin 
        function $op1(A::PyObject, 
            xind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
            yind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
            updates::Union{Array{<:Real}, Real, PyObject})
            m, n = size(A)
            updates = convert_to_tensor(updates, dtype=get_dtype(A))
            @assert length(size(A))==2
            xind = _jlindex2indices(m,xind)
            yind = _jlindex2indices(n,yind)
            if length(size(updates))==0
                updates = reshape(updates, (1,1))
            end
            indices = reshape(repeat(xind, 1, length(yind)), (-1,)) * n + repeat(yind, length(xind)) 
            out = $op2(reshape(A, (-1,)), indices + 1, reshape(updates, (-1,)))
            reshape(out, (m, n))
        end
    end
end

"""
    scatter_update(A::PyObject, 
        xind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
        yind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
        updates::Union{Array{<:Real}, Real, PyObject})

```julia
A[xind, yind] = updates
```
"""
scatter_update(A::PyObject, 
    xind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
    yind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
    updates::Union{Array{<:Real}, Real, PyObject}) = scatter_update2(A, xind, yind, updates)

"""
    scatter_add(A::PyObject, 
        xind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
        yind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
        updates::Union{Array{<:Real}, Real, PyObject})

```julia
A[xind, yind] += updates
```
"""
scatter_add(A::PyObject, 
    xind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
    yind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
    updates::Union{Array{<:Real}, Real, PyObject}) = scatter_add2(A, xind, yind, updates)

"""
    scatter_add(A::PyObject, 
        xind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
        yind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
        updates::Union{Array{<:Real}, Real, PyObject})

```julia
A[xind, yind] -= updates
```
"""
scatter_sub(A::PyObject, 
    xind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
    yind::Union{Colon, Int64, Array{Int64}, BitArray{1}, Array{Bool,1}, UnitRange{Int64}, StepRange{Int64, Int64}, PyObject},
    updates::Union{Array{<:Real}, Real, PyObject}) = scatter_sub2(A, xind, yind, updates)

function norm(o::PyObject, args...;kwargs...)
    kwargs = jlargs(kwargs)
    tf.norm(o, args...;kwargs...)
end

function Base.:diff(o::PyObject; dims::Union{Int64,Nothing}=nothing)
    if dims==nothing
        if length(size(o))!=1
            error("expect rank=1")
        end
        return o[2:end]-o[1:end-1]
    elseif length(size(o))==2
        if dims==1
            return o[2:end,:]-o[1:end-1,:]
        elseif dims==2
            return o[:,2:end]-o[:,1:end-1]
        end
    else
        error("Arguments not understood")
    end
end

clip(o::PyObject, vmin, vmax, args...;kwargs...) = tf.clip_by_value(o, vmin, vmax, args...;kwargs...)

@doc raw"""
    clip(o::Union{Array{Any}, Array{PyObject}}, vmin, vmax, args...;kwargs...)

Clips the values of `o` to the range [`vmin`, `vmax`]

# Example
```julia 
a = constant(3.0)
a = clip(a, 1.0, 2.0)
b = constant(rand(3))
b = clip(b, 0.5, 1.0)
```
"""
function clip(o::Union{Array{Any}, Array{PyObject}}, vmin, vmax, args...;kwargs...)
    out = Array{PyObject}(undef, length(o))
    for i = 1:length(o)
        out[i] = clip(o[i], vmin, vmax, args...;kwargs...)
    end
    out
end

function fft(o::PyObject, args...; kwargs...)
    if length(size(o))==1
        tf.fft(o, args...; kwargs...)
    elseif length(size(o))==2
        tf.fft2d(o, args...; kwargs...)
    elseif length(size(o))==3
        tf.fft3d(o, args...; kwargs...)
    else
        error("FFT for d>=4 not supported")
    end
end


# mimic the Julia SVD 
struct TFSVD
    S::PyObject
    U::PyObject
    V::PyObject
    Vt::PyObject
end

"""
    svd(o::PyObject, args...; kwargs...)

Returns a `TFSVD` structure which holds the following data structures
```julia
S::PyObject
U::PyObject
V::PyObject
Vt::PyObject
```
We have the equality
``o = USV'``

# Example 
```julia
A = rand(10,20)
r = svd(constant(A))
A2 = r.U*diagm(r.S)*r.Vt # The value of `A2` should be equal to `A`
```
"""
function svd(o::PyObject, args...; kwargs...)
    s,u,v = tf.linalg.svd(o)
    TFSVD(s, u, v, v')
end

function ifft(o::PyObject, args...; kwargs...)
    if length(size(o))==1
        tf.ifft(o, args...; kwargs...)
    elseif length(size(o))==2
        tf.ifft2d(o, args...; kwargs...)
    elseif length(size(o))==3
        tf.ifft3d(o, args...; kwargs...)
    else
        error("IFFT for d>=4 not supported")
    end
end

function Base.:real(o::PyObject, args...; kwargs...)
    tf.real(o, args...; kwargs...)
end

function Base.:imag(o::PyObject, args...; kwargs...)
    tf.imag(o, args...; kwargs...)
end

@doc raw"""
    map(fn::Function, o::Union{Array{PyObject},PyObject};
    kwargs...)

Applies `fn` to each element of `o`. 
- `o`∈`Array{PyObject}` : returns `[fn(x) for x in o]`
- `o`∈PyObject : splits `o` according to the first dimension and then applies `fn`. 

# Example
```julia
a = constant(rand(10,5))
b = map(x->sum(x), a) # equivalent to `sum(a, dims=2)`
```

!!! note 
    If `fn` is a multivariate function, we need to specify the output type using `dtype` keyword. For example, 
    ```julia
    a = constant(ones(10))
    b = constant(ones(10))
    fn = x->x[1]+x[2]
    c = map(fn, [a, b], dtype=Float64)
    ```
"""
function map(fn::Function, o::Union{Array{PyObject},PyObject};
         kwargs...)
    # if `o` is not a tensorflow tensor, roll back to normal `map`
    if (isa(o, Array) && !hasproperty(o[1], :graph)) ||
        (isa(o, PyObject) && !hasproperty(o, :graph))
        return fn.(o)
    end
    kwargs = jlargs(kwargs)
    tf.map_fn(fn, o;kwargs...)
end

"""
    pmap(fn::Function, o::Union{Array{PyObject}, PyObject})

Parallel for loop. There should be no data dependency between different iterations.

# Example
```julia
x = constant(ones(10))
y1 = pmap(x->2.0*x, x)
y2 = pmap(x->x[1]+x[2], [x,x])
y3 = pmap(1:10, x) do z
    i = z[1]
    xi = z[2]
    xi + cast(Float64, i)
end
run(sess, y1)
run(sess, y2)
run(sess, y3)
```
"""
function pmap(fn::Function, o::Union{Array{PyObject}, PyObject})
    tf.compat.v1.vectorized_map(fn, o)
end

function pmap(fn::Function, range_::Union{Array{Int64},UnitRange{Int64},StepRange{Int64}, PyObject}, 
        o::Union{Array{PyObject}, PyObject})
    ipt = convert_to_tensor(collect(range_))
    if isa(o, PyObject)
        return tf.compat.v1.vectorized_map(fn, [ipt,o])
    end
    if length(o)==0
        return tf.compat.v1.vectorized_map(fn, ipt)
    end
    if length(o)>0
        tf.compat.v1.vectorized_map(fn, [ipt, o...])
    end
end

dot(x::PyObject, y::PyObject) = sum(x.*y)
dot(x::PyObject, y::AbstractArray{<:Real}{<:Real}) = sum(x.*constant(y))
dot(x::AbstractArray{<:Real}{<:Real}, y::PyObject) = sum(constant(x).*y)

import PyCall: +, -
function +(o::PyObject, I::UniformScaling{Bool})
    @assert size(o,1)==size(o,2)
    o + diagm(0=>ones(size(o,1)))
end

function -(o::PyObject, I::UniformScaling{Bool})
    @assert size(o,1)==size(o,2)
    o - diagm(0=>ones(size(o,1)))
end

Base.:+(I::UniformScaling{Bool}, o::PyObject) = o+I
Base.:-(I::UniformScaling{Bool}, o::PyObject) = -(o-I)

function Base.:findall(o::PyObject)
    if !(length(size(o)) in [1,2])
        error("ADCME: input tensor must have rank 1 or 2")
    end
    if !(eltype(o) <: Bool)
        error("ADCME: input tensor must have boolean types")
    end
    if length(size(o))==2
        tf.compat.v2.where(o) + 1
    else
        o = reshape(o, :, 1)
        res = findall(o)
        res'[1,:]
    end
end 

"""
    vector(i::Union{Array{T}, PyObject, UnitRange, StepRange}, v::Union{Array{Float64},PyObject},s::Union{Int64,PyObject})

Returns a vector `V` with length `s` such that
```
V[i] = v
```
"""
function vector(i::Union{Array{T}, PyObject, UnitRange, StepRange}, v::Union{Array{Float64},PyObject},
                 s::Union{Int64,PyObject}) where T<:Integer
    if isa(i, UnitRange) || isa(i, StepRange)
        i = collect(i)
    end
    i = convert_to_tensor(i)
    s = convert_to_tensor(s)
    i = reshape(i - 1,:,1)
    s = reshape(s, 1)
    tf.scatter_nd(i, v, s)
end

function Base.:repeat(o::PyObject, i::Int64, j::Int64)
    if length(size(o))==0
        return o*ones(eltype(o), i, j)
    end
    if length(size(o))==1
        o = reshape(o, :, 1)
    end
    if length(size(o))!=2
        error("ADCME: size of `o` must be 0, 1, 2")
    end
    tf.tile(o, (i,j))
end

function Base.:repeat(o::PyObject, i::Int64)
    if length(size(o))==0
        o * ones(eltype(o),i)
    elseif length(size(o))==1
        squeeze(repeat(o, i, 1))
    elseif length(size(o))==2
        repeat(o, i, 1)
    else
        error("ADCME: rank of `o` must be 0, 1, 2")
    end
end

function lgamma(o::Union{T, PyObject}, args...;kwargs...) where T<:Real 
    tf.math.lgamma(convert_to_tensor(o), args...;kwargs...)
end

@doc raw"""
    Base.:sort(o::PyObject; 
    rev::Bool=false, dims::Integer=-1, name::Union{Nothing,String}=nothing)

Sort a multidimensional array `o` along the given dimension. 
- `rev`: `true` for DESCENDING and `false` (default) for ASCENDING
- `dims`: `-1` for last dimension. 
"""
function Base.:sort(o::PyObject; 
    rev::Bool=false, dims::Integer=-1, name::Union{Nothing,String}=nothing)
    direction = rev == false ? "ASCENDING" : "DESCENDING"
    axis = -1
    if dims!=-1
        axis = dims - 1
    end
    tf.compat.v1.sort(o, axis=axis, direction=direction, name = name)
end

"""
    topk(o::PyObject, k::Union{PyObject,Integer}=1;
        sorted::Bool=true, name::Union{Nothing,String}=nothing)

Finds values and indices of the `k` largest entries for the last dimension.
If `sorted=true` the resulting k elements will be sorted by the values in descending order.
"""
function topk(o::PyObject, k::Union{PyObject,Integer}=1;
        sorted::Bool=true, name::Union{Nothing,String}=nothing)
    k = convert_to_tensor(k, dtype=Int32)
    tf.compat.v1.math.top_k(o, k, sorted=sorted, name = name)
end

"""
    argsort(o::PyObject; 
    stable::Bool = false, rev::Bool=false, dims::Integer=-1, name::Union{Nothing,String}=nothing)

Returns the indices of a tensor that give its sorted order along an axis.
"""
function argsort(o::PyObject; 
    stable::Bool = false, rev::Bool=false, dims::Integer=-1, name::Union{Nothing,String}=nothing)
    direction = rev == false ? "ASCENDING" : "DESCENDING"
    axis = -1
    if dims!=-1
        axis = dims + 1
    end
    tf.compat.v1.argsort(
        o,
        axis=axis,
        direction=direction,
        stable=stable,
        name=name
    ) + 1
end

function tr(o::PyObject)
    tf.linalg.trace(o)
end



function tril(o::PyObject, num::Int64 = 0)
    flag = false
    if length(size(o))==2
        flag = true 
        o = tf.expand_dims(o, 0)
    end
    tri_lu_ = load_system_op("tri_lu"; multiple=false)
    u,num,lu = convert_to_tensor([o,num,1], [Float64,Int64,Int64])
    out = tri_lu_(u,num,lu)
    if flag 
        return out[1]
    else
        return out 
    end
end

function triu(o::PyObject, num::Int64 = 0)
    flag = false
    if length(size(o))==2
        flag = true 
        o = tf.expand_dims(o, 0)
    end
    tri_lu_ = load_system_op("tri_lu"; multiple=false)
    u,num,lu = convert_to_tensor([o,num,0], [Float64,Int64,Int64])
    out = tri_lu_(u,num,lu)
    if flag 
        return out[1]
    else
        return out 
    end
end

"""
    split(o::PyObject, 
        num_or_size_splits::Union{Integer, Array{<:Integer}, PyObject}; kwargs...)
    
Splits `o` according to `num_or_size_splits`

# Example 1
```julia
a = constant(rand(10,8,6))
split(a, 5)
```
Expected output:
```
5-element Array{PyCall.PyObject,1}:
 PyObject <tf.Tensor 'split_5:0' shape=(2, 8, 6) dtype=float64>
 PyObject <tf.Tensor 'split_5:1' shape=(2, 8, 6) dtype=float64>
 PyObject <tf.Tensor 'split_5:2' shape=(2, 8, 6) dtype=float64>
 PyObject <tf.Tensor 'split_5:3' shape=(2, 8, 6) dtype=float64>
 PyObject <tf.Tensor 'split_5:4' shape=(2, 8, 6) dtype=float64>
```

# Example 2
```julia
a = constant(rand(10,8,6))
split(a, [4,3,1], dims=2)
```
Expected output:
```
3-element Array{PyCall.PyObject,1}:
 PyObject <tf.Tensor 'split_6:0' shape=(10, 4, 6) dtype=float64>
 PyObject <tf.Tensor 'split_6:1' shape=(10, 3, 6) dtype=float64>
 PyObject <tf.Tensor 'split_6:2' shape=(10, 1, 6) dtype=float64>
```

# Example 3
```julia
a = constant(rand(10,8,6))
split(a, 3, dims=3)
```
Expected output:
```
3-element Array{PyCall.PyObject,1}:
 PyObject <tf.Tensor 'split_7:0' shape=(10, 8, 2) dtype=float64>
 PyObject <tf.Tensor 'split_7:1' shape=(10, 8, 2) dtype=float64>
 PyObject <tf.Tensor 'split_7:2' shape=(10, 8, 2) dtype=float64>
```
"""
function Base.:split(o::PyObject, 
    num_or_size_splits::Union{Integer, Array{<:Integer}, PyObject}; kwargs...)
    kwargs = jlargs(kwargs)
    tf.split(o, num_or_size_splits; kwargs...)
end

"""
    reverse(o::PyObject, kwargs...)

Given a tensor `o`, and an index `dims` representing the set of dimensions of tensor to reverse.

# Example 
```julia
a = rand(10,2)
A = constant(a)
@assert run(sess, reverse(A, dims=1)) == reverse(a, dims=1)
@assert run(sess, reverse(A, dims=2)) == reverse(a, dims=2)
@assert run(sess, reverse(A, dims=-1)) == reverse(a, dims=2)
```
"""
function reverse(o::PyObject; dims::Integer = -1)
    if dims==-1
        dims = length(size(o))
    end
    dims = convert_to_tensor([dims-1], dtype=Int32)
    tf.reverse(o, dims)
end

function swish(x)
    return x*sigmoid(x)
end

function hard_swish(x)
    return x*sigmoid(100*x)
end 

function hard_sigmoid(x)
    return min(1.0, maximum(0, x+0.5))
end

function concat_relu(x)
    relu([x -x])
end

function concat_elu(x)
    elu([x -x])
end

function concat_hard_swish(x)
    hard_swish([x -x])
end

function fourier(x, terms::Int64=10)
    list = []
    for i = 1:terms 
        push!(list, sin(x))
        push!(list, cos(x))
    end
    hcat(list...)
end

function __rollfunction(u, window::Int64, op)
    rolling_functions_ = load_op_and_grad(libadcme,"rolling_functions")
    u,window_ = convert_to_tensor(Any[u,window], [Float64,Int64])

    @assert isnothing(size(u)) || length(size(u))==1
    @assert length(u)>=window
    @assert window>1 

    out = rolling_functions_(u,window_,op)
    if !isnothing(length(u))
        set_shape(out, (length(u) - window + 1,))
    else
        out 
    end
end

@doc raw"""
    rollmean(u, window::Int64)

Returns the rolling mean given a window size `m`

$$o_k = \frac{\sum_{i=k}^{k+m-1} u_i}{m}$$

## Rolling functions in ADCME:
- [`rollmean`](@ref): rolling mean 
- [`rollsum`](@ref): rolling sum 
- [`rollvar`](@ref): rolling variance 
- [`rollstd`](@ref): rolling standard deviation
"""
function rollmean(u, window::Int64)
    __rollfunction(u, window, "mean")
end

@doc raw"""
    rollsum(u, window::Int64)

Returns the rolling sum given a window size `m`

$$o_k = \sum_{i=k}^{k+m-1} u_i$$


## Rolling functions in ADCME:
- [`rollmean`](@ref): rolling mean 
- [`rollsum`](@ref): rolling sum 
- [`rollvar`](@ref): rolling variance 
- [`rollstd`](@ref): rolling standard deviation
"""
function rollsum(u, window::Int64)
    __rollfunction(u, window, "sum")
end

@doc raw"""
    rollvar(u, window::Int64)

Returns the rolling variance given a window size `m`

$$o_k = \frac{\sum_{i=k}^{k+m-1} (u_i - m_i)^2}{m-1}$$

Here $m_i$ is the rolling mean computed using [`rollmean`](@ref)


## Rolling functions in ADCME:
- [`rollmean`](@ref): rolling mean 
- [`rollsum`](@ref): rolling sum 
- [`rollvar`](@ref): rolling variance 
- [`rollstd`](@ref): rolling standard deviation
"""
function rollvar(u, window::Int64)
    __rollfunction(u, window, "var")
end

@doc raw"""
    rollstd(u, window::Int64)

Returns the rolling standard deviation given a window size `m`

$$o_k = \sqrt{\frac{\sum_{i=k}^{k+m-1} (u_i - m_i)^2}{m-1}}$$

Here $m_i$ is the rolling mean computed using [`rollmean`](@ref)


## Rolling functions in ADCME:
- [`rollmean`](@ref): rolling mean 
- [`rollsum`](@ref): rolling sum 
- [`rollvar`](@ref): rolling variance 
- [`rollstd`](@ref): rolling standard deviation
"""
function rollstd(u, window::Int64)
    __rollfunction(u, window, "std")
end

"""
    softmax_cross_entropy_with_logits(logits::Union{Array, PyObject}, labels::Union{Array, PyObject})

Computes softmax cross entropy between `logits` and `labels`

`logits` is typically the output of a linear layer. For example,

```
logits = [
    0.124575  0.511463   0.945934
    0.538054  0.0749339  0.187802
    0.355604  0.052569   0.177009
    0.896386  0.546113   0.456832
]
labels = [2;1;2;3]
```

!!! info 
    The values of `labels` are from  {1,2,...,`num_classes`}. Here `num_classes` is the number of columns in `logits`.

The predicted labels associated with `logits` is 
```
argmax(softmax(logits), dims = 2)
```

Labels can also be one hot vectors 
```
labels = [0 1
          1 0
          0 1
          0 1]
```
"""
function softmax_cross_entropy_with_logits(logits::Union{Array, PyObject}, labels::Union{Array, PyObject})
    logits = convert_to_tensor(logits, dtype = Float64)
    labels = convert_to_tensor(labels, dtype = Int64) 
    if ndims(labels)==1
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels - 1)
    else
        tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    end
end