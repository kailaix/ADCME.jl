export AffineConstantFlow, AffineHalfFlow, SlowMAF, MAF, IAF, ActNorm,
       Invertible1x1Conv, NormalizingFlow, NormalizingFlowModel, NeuralCouplingFlow,
       autoregressive_network, FlowOp, LinearFlow, TanhFlow, AffineScalarFlow,
       InvertFlow, AffineFlow

abstract type FlowOp end 

#------------------------------------------------------------------------------------------
# Users can define their own FlowOp in the Main
forward(fo::T, x::Union{Array{<:Real}, PyObject}) where T<:FlowOp  = Main.forward(fo, x)
backward(fo::T, z::Union{Array{<:Real}, PyObject}) where T<:FlowOp = Main.backward(fo, z)

#------------------------------------------------------------------------------------------
mutable struct LinearFlow <: FlowOp
    dim::Int64
    indim::Int64 
    outdim::Int64 
    A::PyObject
    b::PyObject
end

@doc raw"""
    LinearFlow(indim::Int64, outdim::Int64, A::Union{PyObject, Array{<:Real,2}, Missing}=missing, 
        b::Union{PyObject, Array{<:Real,1}, Missing}=missing, name::Union{String, Missing}=missing)
    LinearFlow(A::Union{PyObject, Array{<:Real,2}},
        b::Union{PyObject, Array{<:Real,1}, Missing} = missing; name::Union{String, Missing}=missing)

Creates a linear flow operator, 

$$x = Az + b$$

The operator is not necessarily invertible. $A$ has the size `outdim×indim`, $b$ is a length `outdim` vector.

# Example 1: Reconstructing a Gaussian random variable using invertible A
```julia
using ADCME
using Distributions 

d = MvNormal([2.0;3.0],[2.0 1.0;1.0 2.0])
flows = [LinearFlow(2, 2)]
prior = ADCME.MultivariateNormalDiag(loc = zeros(2))
model = NormalizingFlowModel(prior, flows)
x = rand(d, 10000)'|>Array
zs, prior_logpdf, logdet = model(x)
log_pdf = prior_logpdf + logdet
loss = -sum(log_pdf)
sess = Session(); init(sess)
BFGS!(sess, loss)
run(sess, flows[1].b) # approximately [2.0 3.0]
run(sess, flows[1].A * flows[1].A') # approximate [2.0 1.0;1.0 2.0]
```

# Example 2: Reconstructing a Gaussian random variable using non-invertible A
```julia
using ADCME
using Distributions 

d = Normal()
A = reshape([2. 1.], 2, 1)
b = Variable(ones(2)) 
flows = [LinearFlow(A, b)]
prior = ADCME.MultivariateNormalDiag(loc = zeros(1))
model = NormalizingFlowModel(prior, flows)

x = rand(d, 10000, 1)
x = x * [2.0 1.0] .+ [1.0 3.0]

x = [2.0 2.0]
zs, prior_logpdf, logdet = model(x)
log_pdf = prior_logpdf + logdet
loss = -sum(log_pdf)
sess = Session(); init(sess)
BFGS!(sess, loss)
run(sess, b[1]*2 + b[2]) # approximate 6.0
```
"""
function LinearFlow(indim::Int64, outdim::Int64, A::Union{PyObject, Array{<:Real,2}, Missing}=missing, 
        b::Union{PyObject, Array{<:Real,1}, Missing}=missing; name::Union{String, Missing}=missing)
    if ismissing(name)
        name = randstring(10)
    end
    if ismissing(A)
        if indim==outdim 
            A = diagm(0=>ones(indim))
            A = get_variable(A, name=name*"_A")
        else
            A = get_variable(Float64, shape=[outdim,indim], name=name*"_A")
        end
    end

    if ismissing(b)
        b = get_variable(zeros(1,outdim), name=name*"_b")
    end

    A = constant(A)
    b = constant(b)
    b = reshape(b, (1,-1))

    if size(A)!=(outdim,indim)
        error("Expected A shape: $((outdim,indim)), but got $(size(A))")
    end
    if size(b)!=(1,outdim)
        error("Expected b shape: $((1,outdim)), but got $(size(b))")
    end

    LinearFlow(outdim, indim, outdim, A, b)
end

function LinearFlow(dim::Int64)
    LinearFlow(dim,dim)
end

function LinearFlow(A::Union{PyObject, Array{<:Real,2}},
    b::Union{PyObject, Array{<:Real}, Missing} = missing; name::Union{String, Missing}=missing)
    outdim, indim = size(A)
    if ismissing(b)
        b = zeros(outdim)
    end
    A = constant(A)
    b = constant(b)
    LinearFlow(outdim, indim, outdim, A, b)
end

function forward(fo::LinearFlow, x::Union{Array{<:Real}, PyObject})
    if fo.indim==fo.outdim
        return solve_batch(fo.A, x - fo.b), -log(abs(det(fo.A))) * ones(size(x, 1))
    else
        return solve_batch(fo.A, x - fo.b), -1/2*log(abs(det(fo.A'*fo.A))) * ones(size(x, 1))
    end
end

function backward(fo::LinearFlow, z::Union{Array{<:Real}, PyObject})
    d = nothing
    if fo.indim==fo.outdim
        d = log(abs(det(fo.A)))
    end
    z * fo.A' + fo.b, d
end


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

function forward(fo::AffineConstantFlow, x::Union{Array{<:Real}, PyObject})
    s = ismissing(fo.s) ? zeros_like(x) : fo.s
    t = ismissing(fo.t) ? zeros_like(x) : fo.t
    z = x .* exp(s) + t 
    log_det = sum(s, dims=2)
    return z, log_det
end

function backward(fo::AffineConstantFlow, z::Union{Array{<:Real}, PyObject})
    s = ismissing(fo.s) ? zeros_like(z) : fo.s
    t = ismissing(fo.t) ? zeros_like(z) : fo.t
    x = (z-t) .* exp(-s)
    log_det = sum(-s, dims=2)
    return x, log_det
end


#------------------------------------------------------------------------------------------
mutable struct ActNorm <: FlowOp
    dim::Int64
    fo::AffineConstantFlow
    initialized::PyObject 
end

function ActNorm(dim::Int64, name::Union{String, Missing}=missing)
    if ismissing(name)
        name = "ActNorm_"*randstring(10)
    end
    initialized = get_variable(1, name = name)
    ActNorm(dim, AffineConstantFlow(dim), initialized)
end

function forward(actnorm::ActNorm, x::Union{Array{<:Real}, PyObject})    
    x = constant(x)
    actnorm.fo.s, actnorm.fo.t = tf.cond(tf.equal(actnorm.initialized,constant(0)), 
        ()->(actnorm.fo.s, actnorm.fo.t), #actnorm.fo.s, actnorm.fo.t),
        ()->begin
            setzero = assign(actnorm.initialized, 0)
            x0 = copy(x)
            s0 = reshape(-log(std(x0, dims=1)), 1, :)
            t0 = reshape(mean(-x0 .* exp(s0), dims=1), 1, :)
            op = assign([actnorm.fo.s, actnorm.fo.t], [s0, t0])
            p = tf.print("ActNorm: initializing s and t...")
            op[1] = bind(op[1], setzero)
            op[1] = bind(op[1], p)
            op[1], op[2]
        end)    
    forward(actnorm.fo, x)
end

function backward(actnorm::ActNorm, z::Union{Array{<:Real}, PyObject})
    backward(actnorm.fo, z)
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
mutable struct IAF <: FlowOp
    maf::MAF
end

function IAF(dim::Int64, parity::Bool, config::Array{Int64}; name::Union{String, Missing} = missing, 
    activation::Union{Nothing,String} = "relu", kwargs...)
    MAF(dim, parity, config; name=name, activation=activation, kwargs...)
end

function forward(fo::IAF, x::Union{Array{<:Real}, PyObject})
    backward(fo.maf, x)
end

function backward(fo::IAF, z::Union{Array{<:Real}, PyObject})
    forward(fo.maf, z)
end



#------------------------------------------------------------------------------------------
mutable struct Invertible1x1Conv <: FlowOp
    dim::Int64
    P::Array{Float64}
    L::PyObject
    S::PyObject
    U::PyObject
end

function Invertible1x1Conv(dim::Int64, name::Union{Missing,String}=missing)
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
    L = tril(fo.L, -1) + diagm(0=>ones(fo.dim))
    U = triu(fo.U, 1)
    W = fo.P * L * (U + diagm(fo.S))
    return W
end

function forward(fo::Invertible1x1Conv, x::Union{Array{<:Real}, PyObject})
    W = _assemble_W(fo)
    z = x*W 
    log_det = sum(log(abs(fo.S)))
    return z, log_det
end

function backward(fo::Invertible1x1Conv, z::Union{Array{<:Real}, PyObject})
    W = _assemble_W(fo)
    Winv = inv(W)
    x = z*Winv
    log_det = -sum(log(abs(fo.S)))
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
        s_cond = x->constant(zeros(size(x,1), (dim+1)÷2))
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
mutable struct NeuralCouplingFlow <: FlowOp
    dim::Int64 
    K::Int64
    B::Int64 
    f1::Function
    f2::Function 
end

function NeuralCouplingFlow(dim::Int64, f1::Function, f2::Function, K::Int64=8, B::Int64=3)
    @assert mod(dim,2)==0
    NeuralCouplingFlow(dim, K, B, f1, f2)
end

function forward(fo::NeuralCouplingFlow, x::Union{Array{<:Real}, PyObject})
    x = constant(x)
    dim, K, B, f1, f2 = fo.dim, fo.K, fo.B, fo.f1, fo.f2
    RQS = tfp.bijectors.RationalQuadraticSpline
    
    log_det = constant(zeros(size(x,1)))
    lower, upper = x[:,1:dim÷2], x[:,dim÷2+1:end]

    f1_out = f1(lower)
    @assert size(f1_out,2)==(3K-1)*size(lower,2)
    out = reshape(f1_out, (-1, size(lower,2), 3K-1))
    W, H, D = split(out, [K, K, K-1], dims=3)
    
    W, H = softmax(W, dims=3), softmax(H, dims=3)
    W, H = 2B*W, 2B*H
    D = softplus(D)
   
    rqs = RQS(W, H, D, range_min=-B)
    upper, ld = rqs.forward(upper), rqs.forward_log_det_jacobian(upper, 1)
    log_det += ld

    f2_out = f2(upper) 
    @assert size(f2_out,2)==(3K-1)*size(upper,2)
    out = reshape(f2_out, (-1, size(upper,2), 3K -1))
    W, H, D = split(out, [K, K, K-1], dims=3)
    W, H = softmax(W, dims=3), softmax(H, dims=3)
    W, H = 2B*W, 2B*H 
    D = softplus(D)

    rqs = RQS(W, H, D, range_min=-B)
    lower, ld = rqs.forward(lower), rqs.forward_log_det_jacobian(lower, 1)
    log_det += ld

    return [lower upper], log_det
end

function backward(fo::NeuralCouplingFlow, x::Union{Array{<:Real}, PyObject})
    x = constant(x)
    dim, K, B, f1, f2 = fo.dim, fo.K, fo.B, fo.f1, fo.f2
    RQS = tfp.bijectors.RationalQuadraticSpline

    log_det = constant(zeros(size(x,1)))
    lower, upper = x[:,1:dim÷2], x[:,dim÷2+1:end]

    f2_out = f2(upper)
    @assert size(f2_out,2)==(3K-1)*size(upper,2)
    out = reshape(f2_out, (-1, size(upper,2), 3K-1))
    W, H, D = split(out, [K, K, K-1], dims=3)
    W, H = softmax(W, dims=3), softmax(H, dims=3)
    W, H = 2B*W, 2B*H 
    D = softplus(D)
    
    rqs = RQS(W, H, D, range_min=-B)
    lower, ld = rqs.inverse(lower), rqs.inverse_log_det_jacobian(lower, 1)
    log_det += ld

    f1_out = f1(lower) 
    @assert size(f1_out,2)==(3K-1)*size(lower,2)
    out = reshape(f1_out, (-1, size(lower,2), 3K -1))
    W, H, D = split(out, [K, K, K-1], dims=3)
    W, H = softmax(W, dims=3), softmax(H, dims=3)
    W, H = 2B*W, 2B*H 
    D = softplus(D)
    rqs = RQS(W, H, D, range_min=-B)
    upper, ld = rqs.inverse(upper), rqs.inverse_log_det_jacobian(upper, 1)
    log_det += ld

    return [lower upper], log_det
end


#------------------------------------------------------------------------------------------
mutable struct TanhFlow <: FlowOp
    dim::Int64
    o::PyObject
end

function TanhFlow(dim::Int64)
    TanhFlow(dim, tfp.bijectors.Tanh())
end

function forward(fo::TanhFlow, x::Union{Array{<:Real}, PyObject})
    fo.o.forward(x), fo.o.forward_log_det_jacobian(x, 1)
end

function backward(fo::TanhFlow, z::Union{Array{<:Real}, PyObject})
    fo.o.inverse(z), fo.o.inverse_log_det_jacobian(z, 1)
end

#------------------------------------------------------------------------------------------
mutable struct AffineScalarFlow <: FlowOp
    dim::Int64
    o::PyObject
end

function AffineScalarFlow(dim::Int64, shift::Float64 = 0.0, scale::Float64 = 1.0) 
    AffineScalarFlow(dim, tfp.bijectors.AffineScalar(shift=shift, scale=constant(scale)))
end

function forward(fo::AffineScalarFlow, x::Union{Array{<:Real}, PyObject})
    fo.o.forward(x), fo.o.forward_log_det_jacobian(x, 1)
end

function backward(fo::AffineScalarFlow, z::Union{Array{<:Real}, PyObject})
    fo.o.inverse(z), fo.o.inverse_log_det_jacobian(z, 1)
end


#------------------------------------------------------------------------------------------
mutable struct InvertFlow <: FlowOp
    dim::Int64
    o::FlowOp
end

"""
    InvertFlow(fo::FlowOp)

Creates a flow operator that is the invert of `fo`.
# Example 
```julia
inv_tanh_flow = InvertFlow(TanhFlow(2))
```
"""
function InvertFlow(fo::FlowOp)
    InvertFlow(fo.dim, fo)
end
forward(fo::InvertFlow, x::Union{Array{<:Real}, PyObject}) = backward(fo.o, x)
backward(fo::InvertFlow, z::Union{Array{<:Real}, PyObject}) = forward(fo.o, z)

#------------------------------------------------------------------------------------------
@doc raw"""
    AffineFlow(args...;kwargs...)


Creates a linear flow operator, 

$$z = Ax + b$$

See [`LinearFlow`](@ref)
"""
AffineFlow(args...;kwargs...) = InvertFlow(LinearFlow(args...;kwargs...))

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
        if isnothing(ld) || isnothing(log_det)
            log_det = nothing
        else
            log_det += ld 
        end
        push!(xs, z)
    end
    return xs, log_det
end

#------------------------------------------------------------------------------------------
mutable struct NormalizingFlowModel
    prior::ADCMEDistribution
    flow::NormalizingFlow
end

function Base.:show(io::IO, model::NormalizingFlowModel)
    typevec = typeof.(model.flow.flows)
    println("( $(string(typeof(model.prior))[7:end]) )")
    println("\t↓")
    for i = length(typevec):-1:1
        println("$(typevec[i]) (dim = $(model.flow.flows[i].dim))")
        println("\t↓")
    end
    println("(    Data    )")
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
(o::NormalizingFlowModel)(x::Union{PyObject, Array{<:Real,2}}) = forward(o, x)


#------------------------------------------------------------------------------------------
# From TensorFlow Probability
for (op1, op2) in [(:Abs, :AbsoluteValue), (:Exp, :Exp), (:Log, :Log),
        (:MaskedAutoregressiveFlow, :MaskedAutoregressiveFlow), (:Permute,:Permute),
        (:PowerTransform, :PowerTransform), (:Sigmoid,:Sigmoid), (:Softplus,:Softplus),
        (:Square, :Square)]
    @eval begin 
        export $op1
        mutable struct $op1 <: FlowOp
            dim::Int64
            fo::PyObject
        end
        function $op1(dim::Int64, args...;kwargs...)
            fo = tfp.bijectors.$op2(args...;kwargs...)
            $op1(dim, fo)
        end
        forward(o::$op1, x::Union{PyObject, Array{<:Real,2}}) = (o.fo.forward(x), o.fo.forward_log_det_jacobian(x, 1))
        backward(o::$op1, z::Union{PyObject, Array{<:Real,2}}) = (o.fo.inverse(z), o.fo.inverse_log_det_jacobian(z, 1))
    end
end