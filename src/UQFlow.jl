
import Statistics: mean, cov
import LinearAlgebra: diagm, cholesky
export UQNode, mean, cov, sample, ml_estimator, stddev, UQOp

@doc """
UQNode(loc::Union{Number, Array, PyObject}, scale::Union{Number, Array, PyObject})

Creates a UQNode object that has uncertainty information. The embedded TensorFlow structure of UQNode 
should always be a number or a 1D array
"""
mutable struct UQNode
    loc::PyObject
    cov::PyObject
    function UQNode(loc::Union{Number, AbstractArray, PyObject}, cov::Union{Number, AbstractArray, PyObject})
        if isa(loc, Number)
            loc = [loc]
        elseif isa(loc, PyObject) && length(size(loc))==0
            loc = reshape(loc,1)
        end
        if isa(cov, Number)
            cov = reshape([cov],1,1)
        elseif isa(cov, PyObject) && length(size(cov))==0
            cov = reshape(cov,1,1)
        end
        new(constant(loc), constant(cov))
    end
end

function mean(u::UQNode)
    u.loc
end

function cov(u::UQNode)
    u.cov
end

function Base.:run(sess::PyObject, u::UQNode, args...;kwargs...)
    run(sess, [u.loc, u.cov], args...;kwargs...)
end

function sample(sess::PyObject, u::UQNode, nsp::Int64=1)
    n = length(u)
    a = randn(n, nsp)
    μ, cv = run(sess, u)
    C = cholesky(cv)
    C.L * a + repeat(μ, 1, nsp)
end

# function sample(u::UQNode, n::Integer=1)
#     u.sample(n)
# end

function stddev(u::UQNode)
    sqrt(tf.linalg.diag_part(u.cov))
end

function ml_estimator(value::Union{AbstractArray, PyObject})
    if isa(value, AbstractArray)
        value = constant(value)
        return ml_estimator(value)
    end

    if length(size(value))==1
        error("1D Array is an ambiguous input for `ml_estimator`. Reshape it to 2D Array first.")
    end

    n = size(value,1)
    C = tfp.stats.covariance(value)*n/(n-1)
    μ = mean(value, dims=1)
    μ, C
end

################ The following operations are exact ################

function Base.:-(o::UQNode)
    UQNode(-o.loc, o.cov)
end

function Base.:+(u::UQNode)
    u
end

function Base.:+(u1::UQNode, u2::UQNode)
    UQNode(u1.loc + u2.loc, u1.cov+u2.cov)
end

function Base.:+(u1::Union{Number, AbstractArray, PyObject}, u2::UQNode)
    UQNode(u1 + u2.loc, u2.cov)
end

function Base.:+(u1::UQNode, u2::Union{Number, AbstractArray, PyObject})
    UQNode(u1.loc + u2, u1.cov)
end

function Base.:-(u1::UQNode, u2::UQNode)
    u1 + (-u2)
end

function Base.:-(u1::Union{Number, AbstractArray, PyObject}, u2::UQNode)
    u1 + (-u2)
end

function Base.:-(u1::UQNode, u2::Union{Number, AbstractArray, PyObject})
    u1 + (-u2)
end

function Base.:*(u1::Union{Number, AbstractArray}, u2::UQNode)
    if isa(u1, PyObject) && length(size(u1))==0
        return UQNode(u1 * u2.loc, u1^2*u2.cov)
    end
    UQNode(u1 * u2.loc, u1*u2.cov*u1')
end

function Base.:*(u1::PyObject, u2::UQNode)
    if isa(u1, PyObject) && length(size(u1))==0
        return UQNode(u1 * u2.loc, u1^2*u2.cov)
    end
    UQNode(u1 * u2.loc, u1*u2.cov*u1')
end


function Base.:*(u1::UQNode, u2::Union{PyObject,Number})
    if isa(u2, PyObject) && length(size(u2))>0
        error("Only `UQNode*Scalar` is allowed in this case")
    end
    UQNode(u2 * u1.loc, u2^2*u1.cov)
end

function Base.:/(u1::UQNode, u2::Union{PyObject,Number})
    if isa(u2, PyObject) && length(size(u2))>0
        error("Only `UQNode/Scalar` is allowed in this case")
    end
    u1 * (1/u2)
end

function Base.Broadcast.broadcasted(::typeof(*), o1::Union{Number, AbstractArray, PyObject}, o2::UQNode)
    if isa(o1, Number)
        return o1 * o2
    elseif isa(o1, AbstractArray)
        o1 = diagm(0=>o1)
        return o1 * o2 
    else
        return diag(o1)*o2
    end
end

function Base.Broadcast.broadcasted(::typeof(*), o1::UQNode, o2::Union{Number, AbstractArray, PyObject})
    o2 .* o1
end

function Base.Broadcast.broadcasted(::typeof(/), o1::UQNode, o2::Union{Number, AbstractArray, PyObject})
    if isa(o2, Number) || isa(o2, AbstractArray)
        o2 = constant(o2)
    end
    return (1/o2) .* o1
end

################ Create new ops ################
mutable struct UQOp
    forward::Function
    grad::Union{Function, Missing}
    function UQOp(forward::Function)
        new(forward, missing)
    end
    function (op::UQOp)(o::UQNode)
        local A
        loc = op.forward(o.loc)
        if isa(op.grad, Missing)
            A = gradients(loc, o.loc)
        else
            A = op.grad(o.loc, loc)
        end
        cov = A*o.cov*A'
        UQNode(loc, cov)
    end
end

# an example
function Base.:^(o1::UQNode, o2::Union{Number, PyObject})
    if isa(o2, PyObject) && length(size(o2))>0
        error("Only `UQNode/Scalar` is allowed in this case")
    end
end

function Base.:length(u::UQNode)
    length(u.loc)
end

function Base.:size(u::UQNode)
    return (length(u),)
end

function Base.:size(u::UQNode, r::Integer)
    return size(u)[r]
end

# indexing
function lastindex(u::UQNode)
    return u.event_shape
end

function getindex(u::UQNode, r::Union{Colon, Int64, Array{Bool,1}, BitArray{1}, Array{Int64,1},UnitRange{Int64}, StepRange{Int64, Int64}})
    if typeof(r)==Colon
        return u
    elseif typeof(r)==Array{Bool,1} || typeof(r)==BitArray{1}
        return getindex(u, findall(r))
    elseif typeof(r)==UnitRange{Int64} || typeof(r)==StepRange{Int64, Int64}
        return getindex(u, collect(r))
    elseif typeof(r)==Int64
        return getindex(u, [r])
    elseif typeof(r)==Array{Int64,1}
        A = zeros(length(r), length(u))
        for i = 1:length(r)
            A[i, r[i]] = 1.0
        end
        return A*u
    end
end


function Base.:vcat(args::UQNode...) 
    linop_blocks = [tf.linalg.LinearOperatorFullMatrix(u.cov) for u in args]
    linop_block_diagonal = tf.linalg.LinearOperatorBlockDiag(linop_blocks)
    C = linop_block_diagonal.to_dense()
    μ = vcat([u.loc for u in args]...)
    UQNode(μ, C)
end
    
