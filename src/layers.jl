export 
dense,
dropout,
flatten,
Dense,
ELU,
Flatten,
LeakyReLU,
MaxPooling1D,
MaxPooling2D,
MaxPooling3D,
Reshape,
UpSampling1D,
UpSampling2D,
UpSampling3D,
ZeroPadding1D,
ZeroPadding2D,
ZeroPadding3D,
Conv1D,
Conv2D,
Conv3D,
Conv2DTranspose,
Conv3DTranspose,
BatchNormalization,
Dropout,
ae,
ae_num,
ae_init,
ae_to_code,
fc,
sparse_softmax_cross_entropy_with_logits

# for a keras layer, `training` is a keyword
# dropout = tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None)(dense, training=is_training)

Dense(args...;kwargs...) = tf.keras.layers.Dense(args...;kwargs...)
ELU(args...;kwargs...) = tf.keras.layers.ELU(args...;kwargs...)
Flatten(args...;kwargs...) = tf.keras.layers.Flatten(args...;kwargs...)
LeakyReLU(args...;kwargs...) = tf.keras.layers.LeakyReLU(args...;kwargs...)
MaxPooling1D(args...;kwargs...) = tf.keras.layers.MaxPooling1D(args...;kwargs...)
MaxPooling2D(args...;kwargs...) = tf.keras.layers.MaxPooling2D(args...;kwargs...)
MaxPooling3D(args...;kwargs...) = tf.keras.layers.MaxPooling3D(args...;kwargs...)
Reshape(args...;kwargs...) = tf.keras.layers.Reshape(args...;kwargs...)
UpSampling1D(args...;kwargs...) = tf.keras.layers.UpSampling1D(args...;kwargs...)
UpSampling2D(args...;kwargs...) = tf.keras.layers.UpSampling2D(args...;kwargs...)
UpSampling3D(args...;kwargs...) = tf.keras.layers.UpSampling3D(args...;kwargs...)
ZeroPadding1D(args...;kwargs...) = tf.keras.layers.ZeroPadding1D(args...;kwargs...)
ZeroPadding2D(args...;kwargs...) = tf.keras.layers.ZeroPadding2D(args...;kwargs...)
ZeroPadding3D(args...;kwargs...) = tf.keras.layers.ZeroPadding3D(args...;kwargs...)
Conv1D(args...;kwargs...) = tf.keras.layers.Conv1D(args...;kwargs...)
Conv2D(args...;kwargs...) = tf.keras.layers.Conv2D(args...;kwargs...)
Conv3D(args...;kwargs...) = tf.keras.layers.Conv3D(args...;kwargs...)
Conv2DTranspose(args...;kwargs...) = tf.keras.layers.Conv2DTranspose(args...;kwargs...)
Conv3DTranspose(args...;kwargs...) = tf.keras.layers.Conv3DTranspose(args...;kwargs...)
BatchNormalization(args...;kwargs...) = tf.keras.layers.BatchNormalization(args...;kwargs...)
Dropout(args...;kwargs...) = tf.keras.layers.Dropout(args...;kwargs...)


@doc raw"""
    fc(x::Union{Array{Float64,2},PyObject}, output_dims::Array{Int64,1}, 
    θ::Union{Array{Float64,1}, PyObject};
    activation::String = "tanh")

Creates a fully connected neural network with output dimension `o` and inputs $x\in \mathbb{R}^{m\times n}$. 

$$n \rightarrow o_1 \rightarrow o_2 \rightarrow \ldots \rightarrow o_k$$

`θ` is the weights and biases of the neural network, e.g., `θ = ae_init(output_dims)`.

`fc` outputs two tensors:

- $u\in \mathbb{R}^{m\times o_k}$, the output of the neural network 
- $\partial u\in \mathbb{R}^{m \times o_k \times n}$, the sensitivity of the neural network per sample.
"""
function fc(x::Union{Array{Float64,2},PyObject}, output_dims::Array{Int64,1}, 
    θ::Union{Array{Float64,1}, PyObject};
    activation::String = "tanh")
    if !haskey(COLIB, "extended_nn")
        install("ExtendedNN", force=true)
    end
    extended_nn_ = load_system_op(COLIB["extended_nn"]...; multiple=true)
    config = [size(x,2);output_dims]
    x_,config_,θ_ = convert_to_tensor([x,config,θ], [Float64,Int64,Float64])
    x_ = reshape(x_, (-1,))
    u, du = extended_nn_(x_,config_,θ_,activation)
    reshape(u, (size(x,1), config[end])), reshape(du, (size(x,1), config[end], size(x,2)))
end

"""
    ae(x::PyObject, output_dims::Array{Int64}, scope::String = "default";
        activation::Union{Function,String} = "tanh")

Creates a neural network with intermediate numbers of neurons `output_dims`.
"""
function ae(x::Union{Array{Float64},PyObject}, output_dims::Array{Int64}, scope::String = "default";
        activation::Union{Function,String} = "tanh")
    if isa(x, Array)
        x = constant(x)
    end
    flag = false
    if length(size(x))==1
        x = reshape(x, length(x), 1)
        flag = true
    end
    net = x
    variable_scope(scope, reuse=AUTO_REUSE) do
        for i = 1:length(output_dims)-1
            net = dense(net, output_dims[i], activation=activation)
        end
        net = dense(net, output_dims[end])
    end
    if flag && (size(net,2)==1)
        net = squeeze(net)
    end
    return net
end

"""
    ae(x::Union{Array{Float64}, PyObject}, output_dims::Array{Int64}, θ::Union{Array{Float64}, PyObject};
    activation::Union{Function,String} = "tanh")

Creates a neural network with intermediate numbers of neurons `output_dims`. The weights are given by `θ`

# Example 1: Explicitly construct weights and biases
```julia
x = constant(rand(10,2))
n = ae_num([2,20,20,20,2])
θ = Variable(randn(n)*0.001)
y = ae(x, [20,20,20,2], θ)
```

# Example 2: Implicitly construct weights and biases
```julia
θ = ae_init([10,20,20,20,2]) 
x = constant(rand(10,10))
y = ae(x, [20,20,20,2], θ)
```

See also [`ae_num`](@ref), [`ae_init`](@ref).
"""
function ae(x::Union{Array{Float64}, PyObject}, output_dims::Array{Int64}, θ::Union{Array{Float64}, PyObject};
    activation::Union{Function,String} = "tanh")
    activation=="tanh" && (activation = tanh)
    activation=="sigmoid" && (activation = sigmoid)
    activation=="selu" && (activation = selu)
    activation=="elu" && (activation = elu)
    activation=="relu" && (activation = relu)
    activation=="leaky_relu" && (activation = leaky_relu)
    x = convert_to_tensor(x)
    θ = convert_to_tensor(θ)
    flag = false
    if length(size(x))==1
        x = reshape(x, length(x), 1)
        flag = true
    end
    offset = 0
    net = x
    
    output_dims = [size(x,2); output_dims]
    for i = 1:length(output_dims)-2
        m = output_dims[i]
        n = output_dims[i+1]
        net = net * reshape(θ[offset+1:offset+m*n], (m, n))  + θ[offset+m*n+1:offset+m*n+n]
        net = activation(net)
        offset += m*n+n
    end
    m = output_dims[end-1]
    n = output_dims[end]
    net = net * reshape(θ[offset+1:offset+m*n], (m, n))  + θ[offset+m*n+1:offset+m*n+n]

    offset += m*n+n
    if offset!=length(θ)
        error("ADCME: the weights and configuration does not match. Required $offset but given $(length(θ)).")
    end
    if flag && (size(net,2)==1)
        net = squeeze(net)
    end
    return net
end

function ae(x::Union{Array{Float64}, PyObject}, 
    output_dims::Array{Int64}, 
    θ::Union{Array{Array{Float64}}, Array{PyObject}};
    activation::Union{Function,String} = "tanh")
    if isa(θ, Array{Array{Float64}})
        val = []
        for t in θ
            push!(val, θ'[:])
        end
        val = vcat(val...)
    else
        val = []
        for t in θ
            push!(val, reshape(θ, (-1,)))
        end
        vcat(val...)
    end
    ae(x, output_dims, θ, activation=activation)
end


@doc raw"""
    ae_init(output_dims::Array{Int64}; T::Type=Float64, method::String="xavier")

Return the initial weights and bias values by TensorFlow as a vector. The neural network architecture is

$$o_1 (Input layer) \rightarrow o_2 \rightarrow \cdots \rightarrow o_n (Output layer)$$

Three types of 
random initializers are provided

- `xavier` (default). It is useful for `tanh` fully connected neural network. 
```math 
W^l_i \sim \sqrt{\frac{1}{n_{l-1}}}
```
- `xavier_avg`. A variant of `xavier`
```math 
W^l_i \sim \sqrt{\frac{2}{n_l + n_{l-1}}}
```
- `he`. This is the activation aware initialization of weights and helps mitigate the problem
of vanishing/exploding gradients. 
```math 
W^l_i \sim \sqrt{\frac{2}{n_{l-1}}}
```
"""
function ae_init(output_dims::Array{Int64}; T::Type=Float64, method::String="xavier")
    N = ae_num(output_dims)
    W = zeros(T, N)
    offset = 0
    for i = 1:length(output_dims)-1
        m = output_dims[i]
        n = output_dims[i+1]
        if method=="xavier"
            W[offset+1:offset+m*n] = randn(T, m*n) * T(sqrt(1/m))
        elseif method=="xavier_normal"
            W[offset+1:offset+m*n] = randn(T, m*n) * T(sqrt(2/(n+m)))
        elseif method=="xavier_uniform"
            W[offset+1:offset+m*n] = rand(T, m*n) * T(sqrt(6/(n+m)))
        elseif method=="he"
            W[offset+1:offset+m*n] = randn(T, m*n) * T(sqrt(2/(m)))
        else
            error("Method $method not understood")
        end
        offset += m*n+n
    end
    W
end

"""
    ae_num(output_dims::Array{Int64})

Estimates the number of weights and biases for the neural network. Note the first dimension
should be the feature dimension (this is different from [`ae`](@ref) since in `ae` the feature
dimension can be inferred), and the last dimension should be the output dimension. 
"""
function ae_num(output_dims::Array{Int64})
    offset = 0
    for i = 1:length(output_dims)-2
        m = output_dims[i]
        n = output_dims[i+1]
        offset += m*n+n
    end
    m = output_dims[end-1]
    n = output_dims[end]
    offset += m*n+n
    return offset
end

function _ae_to_code(d::Dict, scope::String; activation::String)
    i = 0
    nn_code = ""
    while true
        si = i==0 ? "" : "_$i"
        Wkey = "$(scope)backslashfully_connected$(si)backslashweightscolon0"
        bkey = "$(scope)backslashfully_connected$(si)backslashbiasescolon0"
        if haskey(d, Wkey)
            if i!=0
                nn_code *= "    isa(net, Array) ? (net = $activation.(net)) : (net=$activation(net))\n"
                nn_code *= "    #-------------------------------------------------------------------\n"
            end
            nn_code *= "    W$i = aedict$scope[\"$Wkey\"]\n    b$i = aedict$scope[\"$bkey\"];\n"
            nn_code *= "    isa(net, Array) ? (net = net * W$i .+ b$i') : (net = net *W$i + b$i)\n"
            i += 1
        else
            break
        end
    end
    nn_code = """  global nn$scope\n  function nn$scope(net)
$(nn_code)    return net\n  end """
    nn_code
end

"""
    ae_to_code(file::String, scope::String; activation::String = "tanh")

Return the code string from the feed-forward neural network data in `file`. Usually we can immediately evaluate 
the code string into Julia session by 
```julia
eval(Meta.parse(s))
```
If `activation` is not specified, `tanh` is the default. 
"""
function ae_to_code(file::String, scope::String; activation::String = "tanh")
    d = matread(file)
    s = "let aedict$scope = matread(\"$file\")\n"*_ae_to_code(d, scope; activation = activation)*"\nend\n"
    return s
end

# tensorflow layers from contrib 
for (op, tfop) in [(:avg_pool2d, :avg_pool2d), (:avg_pool3d, :avg_pool3d),
        (:dropout, :dropout), (:flatten, :flatten), (:max_pool2d, :max_pool2d), (:max_pool3d, :max_pool3d)]
    @eval begin 
        export $op 
        $op(args...; kwargs...) = tf.contrib.layers.$tfop(args...; kwargs...)
    end
end

for (op, tfop) in [(:conv1d, :conv1d), (:conv2d, :conv2d), (:conv2d_in_plane, :conv2d_in_plane),
    (:conv2d_transpose, :conv2d_transpose), (:conv3d, :conv3d), (:conv3d_transpose, :conv3d_transpose)]
    @eval begin 
        export $op 
        function $op(args...;activation = nothing, bias=false,  kwargs...)
            if isa(activation, String)
                if lowercase(activation)=="relu"
                    activation = relu 
                elseif lowercase(activation)=="sigmoid"
                    activation = sigmoid
                elseif lowercase(activation) in ["leakyrelu" , "leaky_relu"]
                    activation = leaky_relu
                else
                    error("Activation function $activation not understood")
                end
            end
            if bias
                biases_initializer = tf.zeros_initializer()
            else
                biases_initializer = nothing
            end
            tf.contrib.layers.$tfop(args...; activation_fn = activation, biases_initializer=biases_initializer, kwargs...)
        end
    end
end


export dense, bn
function dense(inputs, units, args...; activation = nothing, kwargs...) 
    string2fn = Dict(
        "relu" => relu,
        "tanh" => tanh,
        "sigmoid" => sigmoid,
        "leakyrelu" => leaky_relu,
        "leaky_relu" => leaky_relu,
        "relu6" => relu6,
        "softmax" => softmax,
        "softplus" => softplus,
        "selu" => selu,
        "elu" => elu
    )
    if isa(activation, String)
        if haskey(string2fn, lowercase(activation))
            activation = string2fn[lowercase(activation)] 
        else
            error("Activation function $activation not understood")
        end
    end
    tf.contrib.layers.fully_connected(inputs, units,  args...; activation_fn=activation, kwargs...)
end

"""
    bn(args...;center = true, scale=true, kwargs...)

`bn` accepts a keyword parameter `is_training`. 
# Example
```julia
bn(inputs, name="batch_norm", is_training=true)
```

!!! note
    `bn` should be used with `control_dependency`
    ```julia
    update_ops = get_collection(UPDATE_OPS)
    control_dependencies(update_ops) do 
        global train_step = AdamOptimizer().minimize(loss)
    end 
    ```
"""
function bn(args...;center = true, scale=true, kwargs...)
    @warn  """
`bn` should be used with `control_dependency`
```julia
update_ops = get_collection(UPDATE_OPS)
control_dependencies(update_ops) do 
    global train_step = AdamOptimizer().minimize(loss)
end 
```
""" maxlog=1
    kwargs = Dict{Any, Any}(kwargs)
    if :is_training in keys(kwargs)
        kwargs[:training] = kwargs[:is_training]
        delete!(kwargs, :is_training)
    end
    if :scope in keys(kwargs)
        kwargs[:name] = kwargs[:scope]
        delete!(kwargs, :scope)
    end
    tf.layers.batch_normalization(args...;center = center, scale=scale, kwargs...)
end
sparse_softmax_cross_entropy_with_logits(args...;kwargs...) = tf.nn.sparse_softmax_cross_entropy_with_logits(args...;kwargs...)

export group_conv2d
function group_conv2d(inputs::PyObject, filters::Int64, args...;  groups = 1, scope=scope, kwargs...)
    if groups==1
        return conv2d(inputs, filters, args...;scope=scope, kwargs...)
    else
        dims = size(inputs)
        if mod(dims[end], groups)!=0 || mod(filters, groups)!=0 
            error("channels and outputs must be the multiples of `groups`")
        end
        n = div(filters, groups)
        in_ = Array{PyObject}(undef, groups)
        out_ = Array{PyObject}(undef, groups)
        for i = 1:groups 
py"""
temp = $inputs[:,:,:,$((i-1)*n):$(i*n)]
"""
            in_[i] = py"temp"
            out_[i] = conv2d(in_[i], n, args...;scope=scope*"_group$i", kwargs...)
        end
        out = concat(out_, dims=4)
        return out
    end
end

export separable_conv2d
function separable_conv2d(inputs, num_outputs, args...; activation=false, bias=false, kwargs...)
    if isa(activation, String)
        if lowercase(activation)=="relu"
            activation = relu 
        elseif lowercase(activation)=="tanh"
            activation = tanh
        elseif lowercase(activation)=="sigmoid"
            activation = sigmoid
        elseif lowercase(activation) in ["leakyrelu" , "leaky_relu"]
            activation = leaky_relu
        else
            error("Activation function $activation not understood")
        end
    end
    if bias
        biases_initializer = tf.zeros_initializer()
    else
        biases_initializer = nothing
    end
    tf.contrib.layers.separable_conv2d(inputs, num_outputs, args...;activation_fn = activation, 
                        biases_initializer=biases_initializer, kwargs...)
end

export depthwise_conv2d
function depthwise_conv2d(input, num_outputs, args...;
    kernel_size = 3, channel_multiplier = 1, stride = 1, padding="SAME", bias=false, scope="default", reuse=AUTO_REUSE, kwargs...)
    local res, strides
    if isa(kernel_size, Int64)
        kernel_size = (3,3)
    end
    if isa(stride, Int64)
        strides = (1, stride, stride, 1)
    end
    variable_scope(scope, reuse=reuse) do
        filter = get_variable("dconv2d",
                shape=[kernel_size[1], kernel_size[2], size(input,4), channel_multiplier],
                initializer=tf.contrib.layers.xavier_initializer())
        res = tf.nn.depthwise_conv2d(input, filter, strides, padding, args...; kwargs...)
        if bias
            res = tf.contrib.layers.bias_add(res, scope=scope, reuse=AUTO_REUSE)
        end
    end
    return res
end
