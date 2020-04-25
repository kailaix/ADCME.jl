export AffineConstantFlow, AffineHalfFlow, SlowMAF, MAF,
       Invertible1x1Conv, NormalizingFlow, NormalizingFlowModel,
       autoregressive_network

abstract type FlowOp end 

#------------------------------------------------------------------------------------------
mutable struct AffineConstantFlow <: FlowOp
    dim::Int64
    s::Union{Missing,PyObject}
    t::Union{Missing,PyObject}
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

function backward(fo::AffineConstantFlow, z)
    s = ismissing(fo.s) ? tf.zeros_like(z) : fo.s
    t = ismissing(fo.t) ? tf.zeros_like(z) : fo.t
    x = (z-t) .* exp(-s)
    log_det = sum(-s, dims=2)
    return x, log_det
end

#------------------------------------------------------------------------------------------
mutable struct SlowMAF <: FlowOp
    dim::Int64 
    layers::Array{Function}
    order::Array{Int64}
    p::PyObject
end

function SlowMAF(dim::Int64, parity::Bool, nns::Array)
    local order
    @assert length(nns)==dim - 1
    if parity
        order = Array(1:dim)
    else 
        order = reverse(Array(1:dim))
    end
    p = Variable(zeros(2))
    SlowMAF(dim, nns, order, p)
end

function forward(fo::SlowMAF, x::Union{Array{<:Real}, PyObject})
    z = tf.zeros_like(x)
    log_det = zeros(size(x,1))
    for i = 1:fo.dim
        if i==1
            st = repeat(fo.p', size(x,1), 1)
        else
            st = fo.layers[i-1](x[:,1:i-1])
            @assert size(st, 2)==2
        end
        s, t = st[:,1], st[:,2]
        z = scatter_update(z, :, fo.order[i],  x[:, i]*exp(s) + t)
        log_det += s 
    end
    return z, log_det
end

function backward(fo::SlowMAF, z::Union{Array{<:Real}, PyObject})
    x = tf.zeros_like(z)
    log_det = zeros(size(z,1))
    for i = 1:fo.dim 
        if i==1
            st = repeat(fo.p', size(x,1), 1)
        else
            st = fo.layers[i-1](x[:,1:i-1])
        end
        s, t = st[:,1], st[:,2]
        x = scatter_update(x, :, i, (z[:, fo.order[i]] - t) * exp(-s))
        log_det += -s 
    end
    return x, log_det 
end

#------------------------------------------------------------------------------------------
mutable struct MAF <: FlowOp
    dim::Int64 
    net::Function
    parity::Bool
end

function MAF(dim::Int64, parity::Bool, config::Array{Int64}; name::Union{String, Missing} = missing, 
    activation::Union{Nothing,String} = "relu", kwargs...)
    push!(config, 2)
    kwargs = jlargs(kwargs)
    if ismissing(name)
        name = "MAF_"*randstring(10)
    end
    net = x->autoregressive_network(x, config, name; activation = activation)
    MAF(dim, net, parity)
end

function forward(fo::MAF, x::Union{Array{<:Real}, PyObject})
    st = fo.net(x)
    # @info st, split(st, 1, dims=2)
    s, t = squeeze.(split(st, 2, dims=3))
    z = x .* exp(s) + t 
    if fo.parity
        z = reverse(z, dims=2)
    end
    log_det = sum(s, dims=2)
    return z, log_det
end

function backward(fo::MAF, z::Union{Array{<:Real}, PyObject})
    x = zeros_like(z)
    log_det = zeros(size(z, 1))
    if fo.parity
        z = reverse(z, dims=2)
    end
    for i = 1:fo.dim
        st = fo.net(copy(x))
        s, t = squeeze.(split(st, 2, dims=3))
        x = scatter_update(x, :, i, (z[:, i] - t[:, i]) * exp(-s[:, i]))
        log_det += -s[:,i]
    end
    return x, log_det
end

"""
    autoregressive_network(x::Union{Array{Float64}, PyObject}, config::Array{<:Integer}, 
    scope::String="default"; activation::String=nothing, kwargs...)

Creates an masked autoencoder for distribution estimation. 
"""
function autoregressive_network(x::Union{Array{Float64}, PyObject}, config::Array{<:Integer}, 
    scope::String="default"; activation::Union{Nothing,String}="tanh", kwargs...)
    local y
    x = constant(x)
    params = config[end]
    hidden_units = PyVector(config[1:end-1])
    event_shape = [size(x, 2)]
    if haskey(STORAGE, scope*"/autoregressive_network/")
        made = STORAGE[scope*"/autoregressive_network/"]
        y = made(x)
    else
        made = ADCME.tfp.bijectors.AutoregressiveNetwork(params=params, hidden_units = hidden_units,
            event_shape = event_shape, activation = activation, kwargs...)
        STORAGE[scope*"/autoregressive_network/"] = made
        y = made(x)
    end
    return y
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
    z1 = exp(s) .* x1 + t 
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
    x1 = (z1-t).*exp(-s)
    if fo.parity
        x0, x1 = x1, x0
    end
    x = [x0 x1]
    log_det = sum(-s, dims=2)
    return x, log_det
end

#------------------------------------------------------------------------------------------
mutable struct NormalizingFlow 
    flows::Array
    NormalizingFlow(flows::Array) = new(flows)
end

function forward(fo::NormalizingFlow, x::Union{PyObject, Array{<:Real,2}})
    x = constant(x)
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

function backward(fo::NormalizingFlow, z::Union{PyObject, Array{<:Real,2}})
    z = constant(z)
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
    prior::ADCMEDistribution
    flow::NormalizingFlow
end

function NormalizingFlowModel(prior::T, flows::Array{<:FlowOp}) where T<:ADCMEDistribution
    NormalizingFlowModel(prior, NormalizingFlow(flows))
end

function forward(nf::NormalizingFlowModel, x::Union{PyObject, Array{<:Real,2}})
    x = constant(x)
    zs, log_det = forward(nf.flow, x)
    prior_logprob = logpdf(nf.prior, zs[end])
    return zs, prior_logprob, log_det
end

function backward(nf::NormalizingFlowModel, z::Union{PyObject, Array{<:Real,2}})
    z = constant(z)
    xs, log_det = backward(nf.flow, z)
    return xs, log_det
end

function Base.:rand(nf::NormalizingFlowModel, num_samples::Int64)
    z = rand(nf.prior, num_samples)
    xs, _ = backward(nf.flow, z)
    return xs 
end

#------------------------------------------------------------------------------------------
(o::FlowOp)(x::Union{PyObject, Array{<:Real,2}}) = forward(o, x)
(o::NormalizingFlowModel)(x::Union{PyObject, Array{<:Real,2}}) = forward(o, x)