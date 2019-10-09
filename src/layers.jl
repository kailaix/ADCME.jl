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
num_ae,
ae_init,
ae_to_code,
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


"""
    ae(x::PyObject, output_dims::Array{Int64}, scope::String = "default")

Creates a neural network with intermediate numbers of neurons `output_dims`.
"""
function ae(x::Union{Array{Float64},PyObject}, output_dims::Array{Int64}, scope::String = "default")
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
            net = dense(net, output_dims[i], activation="tanh")
        end
        net = dense(net, output_dims[end])
    end
    if flag && (size(net,2)==1)
        net = squeeze(net)
    end
    return net
end

"""
    ae(x::Union{Array{Float64}, PyObject}, output_dims::Array{Int64}, θ::Union{Array{Float64}, PyObject})

Creates a neural network with intermediate numbers of neurons `output_dims`. The weights are given by `θ`
"""
function ae(x::Union{Array{Float64}, PyObject}, output_dims::Array{Int64}, θ::Union{Array{Float64}, PyObject})
    if isa(x, Array)
        x = constant(x)
    end
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
        net = net * reshape(θ[offset+1:offset+m*n], m, n)  + θ[offset+m*n+1:offset+m*n+n]
        net = tanh(net)
        offset += m*n+n
    end
    m = output_dims[end-1]
    n = output_dims[end]
    net = net * reshape(θ[offset+1:offset+m*n], m, n)  + θ[offset+m*n+1:offset+m*n+n]
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
    ae_init(sess::PyObject, x::Union{Array{Float64}, PyObject}, output_dims::Array{Int64})

Return the initial weights and bias values by TensorFlow as a vector.
"""
function ae_init(sess::PyObject, x::Union{Array{Float64}, PyObject}, output_dims::Array{Int64})
    @warn "This function will destroy the current session"
    reset_default_graph()
    y = ae(x, output_dims, "internal")
    vs = get_collection()
    sess = Session()
    init(sess)
    vs = run(sess, vs)
    vals = Array{Float64}[]
    for v in vs
        push!(vals, v[:])
    end
    vcat(vals...)
end

"""
    num_ae(output_dims::Array{Int64})

Estimates the number of weights for the neural network.
"""
function num_ae(output_dims::Array{Int64})
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

function ae_to_code(d::Dict, scope::String)
    i = 0
    nn_code = ""
    while true
        si = i==0 ? "" : "_$i"
        Wkey = "$(scope)backslashfully_connected$(si)backslashweightscolon0"
        bkey = "$(scope)backslashfully_connected$(si)backslashbiasescolon0"
        if haskey(d, Wkey)
            if i!=0
                nn_code *= "\tisa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))\n"
            end
            nn_code *= "\tW$i = aedict$scope[\"$Wkey\"]; b$i = aedict$scope[\"$bkey\"];\n"
            nn_code *= "\tisa(net, Array) ? (net = net * W$i .+ b$i') : (net = net *W$i + b$i)\n"
            i += 1
        else
            break
        end
    end
    nn_code = """function nn$scope(net)
$(nn_code)\treturn net\nend """
    nn_code
end

function ae_to_code(file::String, scope::String)
    d = matread(file)
    s = "aedict$scope = matread(\"$file\");"*ae_to_code(d, scope)
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
    tf.contrib.layers.fully_connected(inputs, units,  args...; activation_fn=activation, kwargs...)
end

"""
example:
bn(inputs, name="batch_norm", is_training=true)
"""
function bn(args...;center = true, scale=true, kwargs...)
    @warn  """
`bn` should be used with `control_dependency`
Example
=======
update_ops = get_collection(UPDATE_OPS)
control_dependencies(update_ops) do 
    global train_step = AdamOptimizer().minimize(loss)
end
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