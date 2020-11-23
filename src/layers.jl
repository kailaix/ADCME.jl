export 
dense,
flatten,
ae,
ae_num,
ae_init,
fc,
fc_num,
fc_init,
fc_to_code,
ae_to_code,
fcx,
bn,
sparse_softmax_cross_entropy_with_logits,
Resnet1D

#----------------------------------------------------
# activation functions

_string2fn = Dict(
    "relu" => relu,
    "tanh" => tanh,
    "sigmoid" => sigmoid,
    "leakyrelu" => leaky_relu,
    "leaky_relu" => leaky_relu,
    "relu6" => relu6,
    "softmax" => softmax,
    "softplus" => softplus,
    "selu" => selu,
    "elu" => elu,
    "concat_elu"=>concat_elu,
    "concat_relu"=>concat_relu, 
    "hard_sigmoid"=>hard_sigmoid,
    "swish"=>swish, 
    "hard_swish"=>hard_swish, 
    "concat_hard_swish"=>concat_hard_swish,
    "sin"=>sin,
    "fourier"=>fourier
)

function get_activation_function(a::Union{Function, String, Nothing})
    if isnothing(a)
        return x->x 
    end
    if isa(a, String)
        if haskey(_string2fn, lowercase(a))
            return _string2fn[lowercase(a)]
        else
            error("Activation function $a is not supported, available activation functions:\n$(collect(keys(_string2fn))).")
        end
    else
        return a 
    end
end


@doc raw"""
    fcx(x::Union{Array{Float64,2},PyObject}, output_dims::Array{Int64,1}, 
    θ::Union{Array{Float64,1}, PyObject};
    activation::String = "tanh")

Creates a fully connected neural network with output dimension `o` and inputs $x\in \mathbb{R}^{m\times n}$. 

$$x \rightarrow o_1 \rightarrow o_2 \rightarrow \ldots \rightarrow o_k$$

`θ` is the weights and biases of the neural network, e.g., `θ = ae_init(output_dims)`.

`fcx` outputs two tensors:

- the output of the neural network: $u\in \mathbb{R}^{m\times o_k}$.

- the sensitivity of the neural network per sample: $\frac{\partial u}{\partial x}\in \mathbb{R}^{m \times o_k \times n}$

"""
function fcx(x::Union{Array{Float64,2},PyObject}, output_dims::Array{Int64,1}, 
    θ::Union{Array{Float64,1}, PyObject};
    activation::String = "tanh")
    pth = joinpath(@__DIR__, "../deps/Plugin/ExtendedNN/build/ExtendedNn")
    pth = get_library(pth)
    require_file(pth) do 
        install_adept()
        change_directory(splitdir(pth)[1])
        require_cmakecache() do 
            cmake()
        end
        make()
    end
    extended_nn_ = load_op_and_grad(pth, "extended_nn"; multiple=true)
    config = [size(x,2);output_dims]
    x_,config_,θ_ = convert_to_tensor([x,config,θ], [Float64,Int64,Float64])
    x_ = reshape(x_, (-1,))
    u, du = extended_nn_(x_,config_,θ_,activation)
    reshape(u, (size(x,1), config[end])), reshape(du, (size(x,1), config[end], size(x,2)))
end

"""
    ae(x::PyObject, output_dims::Array{Int64}, scope::String = "default";
        activation::Union{Function,String} = "tanh")

Alias: `fc`, `ae`

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
    activation::Union{Function,String, Nothing} = "tanh")

Alias: `fc`, `ae`

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
    activation::Union{Function,String, Nothing} = "tanh")
    activation = get_activation_function(activation)
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

"""
    ae(x::Union{Array{Float64}, PyObject}, 
        output_dims::Array{Int64}, 
        θ::Union{Array{Array{Float64}}, Array{PyObject}};
        activation::Union{Function,String} = "tanh")

Alias: `fc`, `ae`

Constructs a neural network with given weights and biases `θ`

# Example
```julia
x = constant(rand(10,30))
θ = ae_init([30, 20, 20, 5])
y = ae(x, [20, 20, 5], θ) # 10×5
```
"""
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
    fc_init(output_dims::Array{Int64})

Return the initial weights and bias values by TensorFlow as a vector. The neural network architecture is

```math
o_1 (\text{Input layer}) \rightarrow o_2 \rightarrow \ldots \rightarrow o_n (\text{Output layer})
```

Three types of 
random initializers are provided

- `xavier` (default). It is useful for `tanh` fully connected neural network. 

```
W^l_i \sim \sqrt{\frac{1}{n_{l-1}}}
```


- `xavier_avg`. A variant of `xavier`

```math
W^l_i \sim \sqrt{\frac{2}{n_l + n_{l-1}}}
```

- `he`. This is the activation aware initialization of weights and helps mitigate the problem
of vanishing/exploding gradients. 

$$W^l_i \sim \sqrt{\frac{2}{n_{l-1}}}$$

# Example
```julia
x = constant(rand(10,30))
θ = ae_init([30, 20, 20, 5])
y = ae(x, [20, 20, 5], θ) # 10×5
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
    fc_num(output_dims::Array{Int64})

Estimates the number of weights and biases for the neural network. Note the first dimension
should be the feature dimension (this is different from [`ae`](@ref) since in `ae` the feature
dimension can be inferred), and the last dimension should be the output dimension. 

# Example
```julia
x = constant(rand(10,30))
θ = ae_init([30, 20, 20, 5])
@assert ae_num([30, 20, 20, 5])==length(θ)
y = ae(x, [20, 20, 5], θ)
```
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
function ae_to_code(file::String, scope::String = "default"; activation::String = "tanh")
    d = matread(file)
    s = "let aedict$scope = matread(\"$file\")\n"*_ae_to_code(d, scope; activation = activation)*"\nend\n"
    return s
end

fc_to_code = ae_to_code

# tensorflow layers from contrib 
for (op, tfop) in [(:avg_pool2d, :avg_pool2d), (:avg_pool3d, :avg_pool3d),
        (:flatten, :flatten), (:max_pool2d, :max_pool2d), (:max_pool3d, :max_pool3d)]
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
            activation = get_activation_function(activation)
            if bias
                biases_initializer = tf.zeros_initializer()
            else
                biases_initializer = nothing
            end
            tf.contrib.layers.$tfop(args...; activation_fn = activation, biases_initializer=biases_initializer, kwargs...)
        end
    end
end

"""
    dense(inputs::Union{PyObject, Array{<:Real}}, units::Int64, args...; 
        activation::Union{String, Function} = nothing, kwargs...)

Creates a fully connected layer with the activation function specified by `activation`
"""
function dense(inputs::Union{PyObject, Array{<:Real}}, units::Int64, args...; activation::Union{String, Function, Nothing} = nothing, kwargs...) 
    inputs = constant(inputs)
    activation = get_activation_function(activation)
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
function separable_conv2d(inputs, num_outputs, args...; activation=nothing, bias=false, kwargs...)
    activation = get_activation_function(activation)
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

"""
$(@doc ae)
"""
fc = ae
"""
$(@doc ae_num)
"""
fc_num = ae_num
"""
$(@doc ae_init)
"""
fc_init = ae_init


#------------------------------------------------------------------------------------------
export dropout

"""
    dropout(x::Union{PyObject, Real, Array{<:Real}}, 
    rate::Union{Real, PyObject}, training::Union{PyObject,Bool} = true; kwargs...)

Randomly drops out entries in `x` with a rate of `rate`. 
"""
function dropout(x::Union{PyObject, Real, Array{<:Real}}, 
    rate::Union{Real, PyObject}, training::Union{PyObject,Bool, Nothing} = nothing ; kwargs...)
    x = constant(x)
    if isnothing(training)
        training = options.training.training 
    else
        training = constant(training)
    end
    tf.keras.layers.Dropout(rate, kwargs...)(x, training)
end


export BatchNormalization
mutable struct BatchNormalization
    dims::Int64
    o::PyObject
end

""" 
    BatchNormalization(dims::Int64=2; kwargs...)

Creates a batch normalization layer. 
# Example
```julia
b = BatchNormalization(2)
x = rand(10,2)
training = placeholder(true)
y = b(x, training)
run(sess, y)
```
"""
function BatchNormalization(dims::Int64=-1; kwargs...)
    local o
    if dims>=1
        o = tf.keras.layers.BatchNormalization(dims-1, kwargs...)
    else
        o = tf.keras.layers.BatchNormalization(dims, kwargs...)
    end
    BatchNormalization(dims, o)
end

function Base.:show(io::IO, b::BatchNormalization)
    print("<BatchNormalization normalization_dim=$(b.dims)>")
end

function (o::BatchNormalization)(x, training=ADCME.options.training.training) 
    flag = false 
    if get_dtype(x)==Float64
        x = cast(x, Float32)
        flag = true 
    end
    out = o.o(x, training)
    if flag 
        out = cast(Float64, out)
    end
    return out 
end

export Dense, Conv1D, Conv2D, Conv3D, Conv2DTranspose

mutable struct Dense 
    hidden_dim::Int64
    activation::Union{String, Function}
    o::PyObject
end

"""
    Dense(units::Int64, activation::Union{String, Function, Nothing} = nothing,
        args...;kwargs...)

Creates a callable dense neural network.
"""
function Dense(units::Int64, activation::Union{String, Function, Nothing} = nothing,
    args...;kwargs...)
    activation = get_activation_function(activation)
    o = tf.keras.layers.Dense(units, activation, args...;kwargs...)
    Dense(units, activation, o)
end

function Base.:show(io::IO, o::Dense)
    print("<Fully connected neural network with $(o.hidden_dim) hidden units and activation function \"$(o.activation)\">")
end

(o::Dense)(x) = o.o(x)

mutable struct Conv1D 
    filters
    kernel_size
    strides
    activation
    o::PyObject 
end
"""
    Conv1D(filters, kernel_size, strides, activation, args...;kwargs...)

```julia
c = Conv1D(32, 3, 1, "relu")
x = rand(100, 6, 128) # 128-length vectors with 6 timesteps ("channels")
y = c(x) # shape=(100, 4, 32)
```
"""
function Conv1D(filters, kernel_size, strides=1, activation=nothing, args...;kwargs...)
    activation = get_activation_function(activation)
    o = tf.keras.layers.Conv1D(filters, kernel_size, strides,args...; activation = activation, kwargs...)
    Conv1D(filters, kernel_size, strides, activation, o)
end
function Base.:show(io::IO, o::Conv1D)
    print("<Conv1D filters=$(o.filters) kernel_size=$(o.kernel_size) strides=$(o.strides) activation=$(o.activation)>")
end
function (o::Conv1D)(x::Union{PyObject, Array{<:Real,3}})
    x = constant(x)
    @assert length(size(x))==3
    o.o(x)
end


mutable struct Conv2D 
    filters
    kernel_size
    strides
    activation
    o::PyObject 
end
"""
    Conv2D(filters, kernel_size, strides, activation, args...;kwargs...)

The arrangement is (samples, rows, cols, channels) (data_format='channels_last')
```julia
Conv2D(32, 3, 1, "relu")
```
"""
function Conv2D(filters, kernel_size, strides=1, activation=nothing, args...;kwargs...)
    activation = get_activation_function(activation)
    o = tf.keras.layers.Conv2D(filters, kernel_size, strides,args...; activation = activation, kwargs...)
    Conv2D(filters, kernel_size, strides, activation, o)
end
function Base.:show(io::IO, o::Conv2D)
    print("<Conv2D filters=$(o.filters) kernel_size=$(o.kernel_size) strides=$(o.strides) activation=$(o.activation)>")
end
function (o::Conv2D)(x::Union{PyObject, Array{<:Real,4}})
    x = constant(x)
    @assert length(size(x))==4
    o.o(x)
end

mutable struct Conv2DTranspose
    filters
    kernel_size
    strides
    activation
    o::PyObject 
end

function Conv2DTranspose(filters, kernel_size, strides=1, activation=nothing, args...;kwargs...)
    activation = get_activation_function(activation)
    o = tf.keras.layers.Conv2DTranspose(
        filters, kernel_size, strides=strides; activation = activation, kwargs...
    )
    Conv2DTranspose(filters, kernel_size, strides, activation, o)
end

function (o::Conv2DTranspose)(x::Union{PyObject, Array{<:Real,4}})
    x = constant(x)
    @assert length(size(x))==4
    o.o(x)
end

mutable struct Conv3D 
    filters
    kernel_size
    strides
    activation
    o::PyObject 
end
"""
    Conv3D(filters, kernel_size, strides, activation, args...;kwargs...)

The arrangement is (samples, rows, cols, channels) (data_format='channels_last')
```julia
c = Conv3D(32, 3, 1, "relu")
x = constant(rand(100, 10, 10, 10, 16))
y = c(x)
# shape=(100, 8, 8, 8, 32)
```
"""
function Conv3D(filters, kernel_size, strides=1, activation=nothing, args...;kwargs...)
    activation = get_activation_function(activation)
    o = tf.keras.layers.Conv3D(filters, kernel_size, strides,args...; activation = activation, kwargs...)
    Conv3D(filters, kernel_size, strides, activation, o)
end
function Base.:show(io::IO, o::Conv3D)
    print("<Conv3D filters=$(o.filters) kernel_size=$(o.kernel_size) strides=$(o.strides) activation=$(o.activation)>")
end
function (o::Conv3D)(x::Union{PyObject, Array{<:Real,5}})
    x = constant(x)
    @assert length(size(x))==5
    o.o(x)
end

#------------------------------------------------------------------------------------------
# resnet, adapted from https://github.com/bayesiains/nsf/blob/master/nn/resnet.py
mutable struct ResnetBlock
    use_batch_norm::Bool 
    activation::Union{String,Function}
    linear_layers::Array{Dense}
    bn_layers::Array{BatchNormalization}
    dropout_probability::Float64
end

function ResnetBlock(features::Int64; dropout_probability::Float64=0., use_batch_norm::Bool,
     activation::String="relu")
    activation = get_activation_function(activation)
    bn_layers = []
    if use_batch_norm
        bn_layers = [
            BatchNormalization(),
            BatchNormalization()
        ]
    end
    linear_layers = [
        Dense(features, nothing)
        Dense(features, nothing)
    ]
    ResnetBlock(use_batch_norm, activation, linear_layers, bn_layers, dropout_probability)
end

function (res::ResnetBlock)(input)
    x = input
    if res.use_batch_norm
        x = res.bn_layers[1](x)
    end
    x = res.activation(x)
    x = res.linear_layers[1](x)
    if res.use_batch_norm
        x = res.bn_layers[2](x)
    end
    x = res.activation(x)
    x = dropout(x, res.dropout_probability, options.training.training)
    x = res.linear_layers[2](x)
    return x + input
end

mutable struct Resnet1D
    initial_layer::Dense
    blocks::Array{ResnetBlock}
    final_layer::Dense
end

"""
    Resnet1D(out_features::Int64, hidden_features::Int64;
        num_blocks::Int64=2, activation::Union{String, Function, Nothing} = "relu", 
        dropout_probability::Float64 = 0.0, use_batch_norm::Bool = false, name::Union{String, Missing} = missing)

Creates a 1D residual network. If `name` is not missing, `Resnet1D` does not create a new entity. 
# Example 
```julia
resnet = Resnet1D(20)
x = rand(1000,10)
y = resnet(x)
```

# Example: Digit recognition
```
using MLDatasets
using ADCME

# load data 
train_x, train_y = MNIST.traindata()
train_x = reshape(Float64.(train_x), :, size(train_x,3))'|>Array
test_x, test_y = MNIST.testdata()
test_x = reshape(Float64.(test_x), :, size(test_x,3))'|>Array

# construct loss function 
ADCME.options.training.training = placeholder(true)
x = placeholder(rand(64, 784))
l = placeholder(rand(Int64, 64))
resnet = Resnet1D(10, num_blocks=10)
y = resnet(x)
loss = mean(sparse_softmax_cross_entropy_with_logits(labels=l, logits=y))

# train the neural network 
opt = AdamOptimizer().minimize(loss)
sess = Session(); init(sess)
for i = 1:10000
    idx = rand(1:60000, 64)
    _, loss_ = run(sess, [opt, loss], feed_dict=Dict(l=>train_y[idx], x=>train_x[idx,:]))
    @info i, loss_
end

# test 
for i = 1:10
    idx = rand(1:10000,100)
    y0 = resnet(test_x[idx,:])
    y0 = run(sess, y0, ADCME.options.training.training=>false)
    pred = [x[2]-1 for x in argmax(y0, dims=2)]
    @info "Accuracy = ", sum(pred .== test_y[idx])/100
end
```
![](https://github.com/ADCMEMarket/ADCMEImages/tree/master/ADCME/assets/resnet.png?raw=true)
"""
function Resnet1D(out_features::Int64, hidden_features::Int64 = 20;
     num_blocks::Int64=2, activation::Union{String, Function, Nothing} = "relu", 
     dropout_probability::Float64 = 0.0, use_batch_norm::Bool = false,
     name::Union{String, Missing} = missing)
     if haskey(ADCME.STORAGE, name)
        @info "Reusing $name..."
        return ADCME.STORAGE[name]
     end
     initial_layer = Dense(hidden_features)
     blocks = ResnetBlock[]
     for i = 1:num_blocks
        push!(blocks, ResnetBlock(
            hidden_features,
            dropout_probability = dropout_probability, 
            use_batch_norm = use_batch_norm
        ))
     end
     final_layer = Dense(out_features)
     if ismissing(name)
        name = "Resnet1D_"*randstring(10)
     end
     res = Resnet1D(initial_layer, blocks, final_layer)
     ADCME.STORAGE[name] = res 
     return res 
end

function (res::Resnet1D)(x)
    x = res.initial_layer(x)
    for b in res.blocks
        x = b(x)
    end
    x = res.final_layer(x)
    x
end

function Base.:show(io::IO, res::Resnet1D)
    println("( Input )")
    println("\t↓")
    show(io, res.initial_layer)
    println("\n")
    for i = 1:length(res.blocks)
        show(io, res.blocks[i])
        println("\n")
        println("\t↓")
    end
    show(io, res.final_layer)
    println("\n( Output )")
end