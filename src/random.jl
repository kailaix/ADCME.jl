export seed!, categorical, choice
# random variables
for op in [:uniform, :normal]
    @eval begin
        export $op
        function $op(shape...;kwargs...)
            kwargs = jlargs(kwargs)
            if !haskey(kwargs, :dtype); kwargs[:dtype] = tf.float64; end
            tf.random.$op(shape;kwargs...)
        end
        $op() = squeeze($op(1, dtype=Float64))
        Docs.getdoc(::typeof($op)) = Docs.getdoc(tf.random.$op)
    end
end

"""
categorical(n::Union{PyObject, Integer}; kwargs...)

`kwargs` has a keyword argument `logits`, a 2-D Tensor with shape `[batch_size, num_classes]`.  
Each slice `[i, :]` represents the unnormalized log-probabilities for all classes.
"""
function categorical(n::Union{PyObject, Integer}; kwargs...)
    flag = false
    kwargs = jlargs(kwargs)
    if !haskey(kwargs, :dtype); kwargs[:dtype] = tf.float64; end
    if !haskey(kwargs, :logits)
        kwargs[:logits] = tf.ones(n, dtype=kwargs[:dtype])
    end
    logits = kwargs[:logits]; delete!(kwargs, :logits)
    if length(size(logits))==1
        flag = true
        logits = convert_to_tensor(reshape(logits, 1, length(logits)))
    end
    out = tf.random.categorical(logits, n)
    if flag
        out = squeeze(out)
    end
    return out+1
end

"""
choice(inputs::Union{PyObject, Array}, n_samples::Union{PyObject, Integer};replace::Bool=false)

Choose `n_samples` samples from `inputs` with/without replacement. 
"""
function choice(inputs::Union{PyObject, Array}, n_samples::Union{PyObject, Integer};
    replace::Bool=false)
    inputs = convert_to_tensor(inputs)
    if replace
        Idx = categorical(n_samples, logits=ones(size(inputs,1)))
        tf.gather(inputs, Idx-1)
    else
        dist = uniform(size(inputs,1))
        _, indices_to_keep = tf.nn.top_k(dist, n_samples)
        indices_to_keep_sorted = tf.sort(indices_to_keep)
        tf.gather(inputs,indices_to_keep)
    end
end

for op in [:Beta, :Bernoulli,:Gamma, :TruncatedNormal, :Binomial, :Cauchy, 
        :Chi, :Chi2, :Exponential, :Gumbel, 
        :HalfCauchy, :HalfNormal, :Horseshoe, :InverseGamma,
        :InverseGaussian, :Kumaraswamy, :Pareto, :SinhArcsinh,
        :StudentT, :VonMises,
        :Poisson]
    opname = Symbol(lowercase(string(op)))
    @eval begin
        export $opname
        function $opname(shape...;kwargs...)
            if !haskey(kwargs, :dtype); T = Float64; 
            else T=kwargs[:dtype]; end
            kwargs = jlargs(kwargs)
            if haskey(kwargs, :dtype); delete!(kwargs, :dtype); end
            out = tfp.distributions.$op(;kwargs...).sample(shape...)
            cast(out, T)
        end
        $opname(;kwargs...) = squeeze($opname(1; dtype =Float64, kwargs...))
        Docs.getdoc(::typeof($opname)) = Docs.getdoc(tfp.distributions.$op)
    end
end

function seed!(k::Int64)
    tf.random.set_random_seed(k)
end
