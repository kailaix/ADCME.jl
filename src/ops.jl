import Base:*, broadcast, reshape, exp, log, tanh, sum, 
    adjoint, inv, argmax, argmin, ^, max, maximum, min, minimum,
    vec, \, cos, sin, sign, map, prod
import LinearAlgebra: diag, det, norm, diagm, dot, I, svd
import Statistics: mean
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
sum,
relu,
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
scatter_div,
scatter_max,
scatter_min,
scatter_mul,
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
pmap


function PyCall.:*(o1::PyObject, o2::PyObject)
    s1 = size(o1)
    s2 = size(o2)
    if s1==nothing || s2==nothing
        error("o1 and o2 should be tensors of rank 0, 1, 2")
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
    elseif length(s1)==1 && length(s2)==0
        return tf.multiply(o1, o2)
    elseif length(s1)==0 && length(s2)==2
        return tf.multiply(o1, o2)
    elseif length(s1)==0 && length(s2)==1
        return tf.multiply(o1, o2)
    elseif length(s1)==0 && length(s2)==0
        return tf.multiply(o1, o2)
    else
        @warn("Unusual usage of multiplication. Check carefully")
        tf.multiply(o1,o2)
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


function reshape(o::PyObject, s::Integer; kwargs...)
    if length(size(o))==2
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


function _tfreshape(o::PyObject, s...; kwargs...)
    if length(size(o))==2
        return tf.reshape(o', [s...]; kwargs...)
    end
    tf.reshape(o, [s...]; kwargs...)
end

function sigmoid(o::PyObject; kwargs...)
    tf.math.sigmoid(o; kwargs...)
end

function relu(o::PyObject; kwargs...)
    tf.nn.relu(o; kwargs...)
end

function tan(o::PyObject; kwargs...)
    tf.math.tan(o; kwargs...)
end

function leaky_relu(o::PyObject; kwargs...)
    tf.nn.leaky_relu(o; kwargs...)
end

function tanh(o::PyObject; kwargs...)
    tf.tanh(o; kwargs...)
end

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

function cast(x::PyObject, dtype::Type;kwargs...)
    dtype = DTYPE[dtype]
    tf.cast(x, dtype; kwargs...)
end

function cast(dtype::Type, x::PyObject;kwargs...)
    dtype = DTYPE[dtype]
    tf.cast(x, dtype; kwargs...)
end

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
function pad(o::PyObject, paddings, args...; kwargs...)
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

function vec(o::PyObject;kwargs...)
    s = size(o)
    if length(s)==0
        return reshape(o, 1,kwargs...)
    elseif length(s)==1
        return o
    elseif length(s)==2
        return tf.reshape(tf.linalg.adjoint(o), (s[1]*s[2],),kwargs...)
    else
        error("Invalid argument")
    end
end

# linear algebra
function adjoint(o::PyObject; kwargs...) 
    if length(size(o))<=1
        return rvec(o)
    end
    tf.linalg.adjoint(o; kwargs...)
end
diagm(o::PyObject; kwargs...) = tf.linalg.diag(o; kwargs...)
diag(o::PyObject; kwargs...) = tf.linalg.diag_part(o; kwargs...)
det(o::PyObject; kwargs...) = tf.linalg.det(o; kwargs...)
inv(o::PyObject; kwargs...) = tf.linalg.inv(o; kwargs...)

function solve(matrix, rhs; kwargs...)
    flag = false
    if isa(rhs, Array)
        rhs = constant(rhs)
    end
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
Base.:hcat(args::PyObject...) = length(size(args[1]))>=2 ? concat([args...],1) : stack([args...],dims=2)

# for TensorArray
function stack(o::PyObject)
    o.stack()
end

function unstack(o::PyObject, args...;kwargs...)
    kwargs = jlargs(kwargs)
    tf.unstack(o, args...; kwargs...)
end

for (op1, op2) = [(:scatter_add, :tensor_scatter_nd_add), (:scatter_sub, :tensor_scatter_nd_sub), (:scatter_update,:tensor_scatter_nd_update)]
    @eval begin
        function $op1(ref::PyObject, indices::PyObject, updates)
            if length(size(ref))==1
                indices = reshape(indices-1, length(indices), 1)
            else
                error("Only 1D $op1 is implemented")
            end
            tf.$op2(ref, indices, updates)
        end
    end
end

# https://github.com/tensorflow/tensorflow/issues/2358#issuecomment-274590896
for (op1, op2) = [(:scatter_add, :add), (:scatter_sub, :subtract), (:scatter_mul,:multiply), (:scatter_div, div)]
    @eval begin
        function $op1(ref::PyObject, indices, updates; kwargs...)
            if isa(indices, BitArray{1}) || isa(indices, Array{Bool,1})
                indices = findall(indices)
            elseif isa(indices, UnitRange{Int64}) || isa(indices, StepRange{Int64, Int64})
                indices = collect(indices)
            elseif isa(indices, Colon)
                indices = (1:size(ref,1) |> collect)
            end
            
            indices = indices .- 1
            
            if isa(indices, Number)
                indices = reshape([indices], 1, 1)
            else
                indices = reshape(indices, length(indices), 1)
            end
            
            if isa(updates, Number)
                updates = reshape([updates], 1)
            elseif isa(updates, Array)
                updates = reshape(updates, length(updates))
            elseif isa(updates, PyObject)
                if length(size(updates))==0
                    updates = reshape(updates, 1)
                else
                    updates = reshape(updates, size(updates,1))
                end
            end
            ref_shape = size(ref)
            scattered_updates = tf.scatter_nd(indices, updates, ref_shape)
            output = tf.$op2(ref, scattered_updates)
        end
    end
end

const Index = Union{Int64, Array{Int64,1}, UnitRange{Int64}, StepRange{Int64,Int64}}

function _sub2ind_scatter_update(idx::Index, idy::Index, M::Int64, N::Int64)
    if isa(idx, Int64)
        idx = [idx]
    end
    if isa(idy, Int64)
        idy = [idy]
    end
    idx = collect(idx)
    idy = collect(idy)
    if maximum(idx)>M || maximum(idy)>N || minimum(idx)<1 || minimum(idy)<1
        error("Invalid argument for idx and idy")
    end
    ind = zeros(Int64, length(idx)*length(idy))
    for i = 1:length(idx)
        for j = 1:length(idy)
            # ind[(i-1)*length(idy)+j] = idy[j] + (idx[i]-1)*N
            ind[i + (j-1)*length(idx)] = idx[i] + (idy[j]-1)*M
        end
    end
    ind
end

# matrix row major
for op = [:scatter_add, :scatter_sub, :scatter_mul, :scatter_div]
    @eval begin
        function $op(ref::PyObject, idx::Index, idy::Index, updates, 
                    M::Union{Nothing,Int64}=nothing, N::Union{Nothing,Int64}=nothing; kwargs...)
            if length(size(ref))==2 &&  (M!=nothing) && (N!=nothing)
                error("No need to provide M, N for a matrix")
            end
            is_matrix = false
            if M==nothing || N==nothing
                if length(size(ref))!=2
                    error("M,N not provided, ref must be a matrix")
                end
                M, N = size(ref)
                ref = reshape(ref, M*N)
                is_matrix = true
            end
            indices = _sub2ind_scatter_update(idx, idy, M, N)
            if isa(updates, Array)
                updates = updates[:]
            elseif isa(updates, PyObject)
                updates = reshape(updates, length(idx)*length(idy))
            end
            ref = $op(ref, indices, updates; kwargs...)
            if is_matrix
                ref = reshape(ref, M, N)
            end
            return ref
        end
    end
end

function norm(o::PyObject, args...;kwargs...)
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

function map(fn::Function, o::PyObject; kwargs...)
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