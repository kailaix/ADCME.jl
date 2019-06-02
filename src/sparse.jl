using SparseArrays
export sparsetensor,
sparse_dense_matmul,
sparse_eval

function sparsetensor(ii::Array{T}, jj::Array{T}, vv::Union{PyObject,Array},
        m::Union{Int64, Nothing}=nothing, n::Union{Int64, Nothing}=nothing; 
        row_major=true, reorder = true) where T<:Integer
    ii = Array{Int32}(ii); jj = Array{Int32}(jj)
    if isnothing(m) || isnothing(n)
        m = maximum(ii)
        n = maximum(jj)
    end

    if !reorder
        return tf.SparseTensor(indices = indices .- 1, values = vv, dense_shape=[m;n])
    end

    d = Dict{Tuple{Int32, Int32}, PyObject}()
    for i = 1:length(ii)
        if haskey(d, (ii[i],jj[i]))
            d[(ii[i],jj[i])] += vv[i]
        else
            d[(ii[i],jj[i])] = isa(vv[i], PyObject) ? vv[i] : constant(vv[i])
        end
    end
    N = length(d)
    indices = zeros(Int32, N, 2)
    keys_ = keys(d)|>collect
    for i = 1:N
        indices[i,:] = [keys_[i][1];keys_[i][2]]
    end
    if row_major==true
        linearized_indices = indices[:,2] + (indices[:,1] .- 1)*m
        Idx = sortperm(linearized_indices)
        indices = indices[Idx,:]
    else
        linearized_indices = indices[:,1] + (indices[:,2] .- 1)*n
        Idx = sortperm(linearized_indices)
        indices = indices[Idx,:]
    end
    values = Array{PyObject}(undef, N)
    for i = 1:N
        values[i] = d[(indices[i,1], indices[i,2])]
    end
    indices = indices .- 1
    values = tensor(values)
    tf.SparseTensor(indices = indices, values = values, dense_shape=[m;n])
end

function Base.:Array(o::PyObject)
    o = tf.sparse_reorder(o) # convert colume major to row major
    tf.sparse_tensor_to_dense(o)
end

"""
Multiply SparseTensor (of rank 2) "A" by dense matrix "B"
"""
function sparse_dense_matmul(o1::PyObject, o2::Union{Array, PyObject}; kwargs...)
    if length(size(o2))==1
        input_b = reshape(o2, length(o2), 1)
    elseif length(size(o2))==2
        input_b = o2
    else
        error("b must be rank 1 or 2")
    end
    input_b = cast(input_b, get_dtype(o1))
            
    res = tf.sparse_tensor_dense_matmul(o1, input_b; kwargs...)
    if length(size(o2))==1
        return reshape(res, length(res))
    end
    return res
end

function sparse_eval(sess::PyObject, sp::PyObject)
    indices, values, s = run(sess, sp)
    sparse(indices[:,1] .+ 1, indices[:,2] .+ 1, values, s[1], s[2])
end