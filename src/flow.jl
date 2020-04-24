export AffineConstantFlow, AffineHalfFlow, 
       Invertible1x1Conv, NormalizingFlow, NormalizingFlowModel
       
abstract type FlowOp end 

#------------------------------------------------------------------------------------------
mutable struct AffineConstantFlow <: FlowOp
    dim::Int64
    s::PyObject
    t::PyObject
end

function AffineConstantFlow(dim::Int64, name::Union{String, Missing}=missing; scale::Bool=true, shift::Bool=true)
    if ismissing(name)
        name = randstring(10)
    end
    s = missing
    t = missing
    if scale
        s = get_variable(Float64, shape=[1,dim], name=name*"_s")
    end
    if shift
        t = get_variable(Float64, shape=[1,dim], name=name*"_t")
    end
    AffineConstantFlow(dim, s, t)
end


function ActNorm(x, name::Union{String, Missing}=missing)
    dim = size(x,2)
    s = -log(std(x, dims=1))
    t = -mean(x .* exp(s), dims=1)
    AffineConstantFlow(dim, reshape(s, 1, :), reshape(t, 1, :))
end

function forward(fo::AffineConstantFlow, x)
    s = ismissing(fo.s) ? tf.zeros_like(x) : fo.s
    t = ismissing(fo.t) ? tf.zeros_like(x) : fo.t
    z = x .* exp(s) + t 
    log_det = sum(s, dims=2)
    return z, log_det
end

function backward(fo::AffineConstantFlow, x)
    s = ismissing(fo.s) ? tf.zeros_like(x) : fo.s
    t = ismissing(fo.t) ? tf.zeros_like(x) : fo.t
    x = (z-t) .* exp(-s)
    log_det = sum(-s, dims=2)
    return x, log_det
end

#------------------------------------------------------------------------------------------
mutable struct Invertible1x1Conv <: FlowOp
    dim
    P
    L
    S
    U
end

function Invertible1x1Conv(dim, name=missing)
    if ismissing(name)
        name = randstring(10)
    end
    Q = lu(randn(dim,dim))
    P, L, U = Q.P, Q.L, Q.U 
    L = get_variable(L, name=name*"_L")
    S = get_variable(diag(U), name=name*"_S")
    U = get_variable(triu(U, 1), name=name*"_U")
    Invertible1x1Conv(dim, P, L, S, U)
end

function _assemble_W(fo::Invertible1x1Conv)
    L = tril(L, -1) + diagm(0=>ones(dim))
    U = triu(U, 1)
    W = P * L * (U + diagm(S))
    return W
end

function forward(fo::Invertible1x1Conv, x)
    W = _assemble_W(fo)
    z = x*W 
    log_det = sum(log(abs(fo.S)))
    return z, log_det
end

function backward(fo::Invertible1x1Conv, z)
    W = _assemble_W(fo)
    Winv = inv(W)
    x = z*Winv
    log_det = -sum(log(abs(S)))
    return x, log_det 
end
    
#------------------------------------------------------------------------------------------
mutable struct AffineHalfFlow <: FlowOp
    dim::Int64
    parity::Bool
    s_cond::Function
    t_cond::Function
end

"""
    AffineHalfFlow(dim::Int64, parity::Bool, s_cond::Union{Function, Missing} = missing, t_cond::Union{Function, Missing} = missing)

Creates an `AffineHalfFlow` operator. 
"""
function AffineHalfFlow(dim::Int64, parity::Bool, s_cond::Union{Function, Missing} = missing, t_cond::Union{Function, Missing} = missing)
    if ismissing(s_cond)
        s_cond = x->constant(zeros(size(x,1), dim÷2))
    end
    if ismissing(t_cond)
        t_cond = x->constant(zeros(size(x,1), dim÷2))
    end
    AffineHalfFlow(dim, parity, s_cond, t_cond)
end

function forward(fo::AffineHalfFlow, x::Union{Array{<:Real}, PyObject})
    x = constant(x)
    x0 = x[:,1:2:end]
    x1 = x[:,2:2:end]
    if fo.parity
        x0, x1 = x1, x0 
    end
    s = fo.s_cond(x0)
    t = fo.t_cond(x0)
    z0 = x0
    z1 = exp(s) * x1 + t 
    if fo.parity
        z0, z1 = z1, z0 
    end
    z = [z0 z1]
    log_det = sum(s, dims=2)
    return z, log_det
end

function backward(fo::AffineHalfFlow, z::Union{Array{<:Real}, PyObject})
    z = constant(z)
    z0 = z[:,1:2:end]
    z1 = z[:,2:2:end]
    if fo.parity
        z0, z1 = z1, z0 
    end
    s = fo.s_cond(z0)
    t = fo.t_cond(z0)
    x0 = z0 
    x1 = (z1-t)*exp(-s)
    if parity
        x0, x1 = x1, x0
    end
    x = [x0 x1]
    log_det = sum(-s, dims=2)
    return x, log_det
end

#------------------------------------------------------------------------------------------
mutable struct NormalizingFlow <: FlowOp
    flows
end

function forward(fo::NormalizingFlow, x)
    m = size(x,1)
    log_det = constant(zeros(m))
    zs = PyObject[x]
    for flow in fo.flows
        x, ld = forward(flow, x)
        log_det += ld 
        push!(zs, x)
    end
    return zs, log_det
end

function backward(fo::NormalizingFlow, z)
    m = size(z,1)
    log_det = constant(zeros(m))
    xs = [z]
    for flow in reverse(fo.flows)
        z, ld = backward(flow, z)
        log_det += ld 
        push!(xs, z)
    end
    return xs, log_det
end

#------------------------------------------------------------------------------------------
mutable struct NormalizingFlowModel
    prior
    flow::NormalizingFlow
end

function NormalizingFlowModel(prior, flows::Array{FlowOp})
    NormalizingFlowModel(prior, NormalizingFlow(flows))
end

function forward(nf::NormalizingFlow, x)
    zs, log_det = forward(nf.flow, x)
    prior_logprob = log_prob(fo.prior, zs[end])
    return zs, prior_logprob, log_det
end

function backward(nf::NormalizingFlow, z)
    xs, log_det = backward(nf.flow, z)
    return xs, log_det
end

function sample(nf::NormalizingFlow, num_samples)
    z = sample(nf.prior, num_samples)
    xs, _ = backward(flow, z)
    return xs 
end