# Neural Networks 

A neural network can be viewed as a computational graph where each operator in the computational graph is composed of linear transformation or simple explicit nonlinear mapping (called _activation functions_). There are essential components of the neural network 

1. Input: the input to the neural network, which is a real or complex valued vector; the input is often called _features_ in machine learning. To leverage dense linear algebra, features are usually aggregated into a matrix and fed to the neural network. 
2. Output: the output of the neural network is also a real or complex valued vectors. The vector can be tranformed to categorical values (labels) based on the specific application. 

The common activations functions include ReLU (Rectified linear unit), tanh, leaky ReLU, SELU, ELU, etc. In general, for inverse modeling in scientific computing, tanh usually outperms the others due to its smoothness and boundedness, and forms a solid choice at the first try. 

A common limitation of the neural network is overfitting. The neural network contains plenty of free parameters, which makes the neural network "memorize" the training data easily. Therefore, you may see very a small training error, but have large test errors. Regularization methods have been proposed to alleviate this problem; to name a few, restricting network sizes, imposing weight regulization (Lasso or Ridge), using Dropout and batch normalization, etc. 

## Constructing a Neural Network 

ADCME provides a very simple way to specify a fully connected neural network, [`fc`](@ref) (short for _autoencoder_)

```julia
x = constant(rand(10,2)) # input
config = [20,20,20,3] # hidden layers
θ = fc_init([2;config]) # getting an initial weight-and-biases vector. 

y1 = fc(x, config)
y2 = fc(x, config, θ)
```

!!! note
    When you construct a neural network using `fc(x, config)` syntax, ADCME will construct the weights and biases automatically for you and label the parameters (the default is `default`). In some cases, you may have multiple neural networks, and you can label the neural network manually using 

    ```julia
    fc(x1, config1, "label1")
    fc(x2, config2, "label2")
    ...
    ```

In scientific computing, sometimes we not only want to evaluate the neural network output, but also the sensitivity. Specifically, if 

$$y = NN_{\theta}(x)$$

We also want to compute $\nabla_x NN_{\theta}(x)$. ADCME provides a function [`fcx`](@ref) (short for _fully-connected_)

```julia
y3, dy3 = fcx(x, config, θ)
```

Here `dy3` will be a $10\times 3 \times 2$ tensor, where `dy3[i,:,:]` is the Jacobian matrix of the $i$-th output with respect to the $i$-th input (Note the $i$-th output is independent of $j$-th input, whenever $i\neq j$).

## Prediction

After training a neural network, we can use the trained neural network for prediction. Here is an example

```julia
using ADCME
x_train = rand(10,2)
x_test = rand(20,2)
y = fc(x_train, [20,20,10])
y_obs = rand(10,10)
loss = sum((y-y_obs)^2)
sess = Session(); init(sess)
BFGS!(sess, loss)
# prediction
run(sess, fc(x_test, [20,20,10]))
```

Note that the second `fc` does not create a new neural network, but instead searches for a neural network with the label `default` because the default label is `default`. If you constructed a neural network with label `mylabel`: `fc(x_train, [20,20,10], "mylabel")`, you can predict using 

```julia
run(sess, fc(x_test, [20,20,10], "mylabel"))
```

## Save the Neural Network 

To save the trained neural network in the Session `sess`, we can use

```julia
ADCME.save(sess, "filename.mat")
```

This will create a `.mat` file that contains all the **labeled** weights and biases. If there are other variables besides neural network parameters, these variables will also be saved. 

To load the weights and biases to the current session, create a neural network with the same label and run

```julia
ADCME.load(sess, "filename.mat")
```

## Convert Neural Network to Codes

Sometimes we may also want to convert a fully-connected neural network to pure Julia codes. This can be done via [`fc_to_code`](@ref). 


After saving the neural network to a mat file via `ADCME.save`, we can call

```julia
ae_to_code("filename.mat", "mylabel")
```

If the second argument is missing, the default is `default`. For example,

```
julia> ae_to_code("filename.mat", "default")|>println
let aedictdefault = matread("filename.mat")
  global nndefault
  function nndefault(net)
    W0 = aedictdefault["defaultbackslashfully_connectedbackslashweightscolon0"]
    b0 = aedictdefault["defaultbackslashfully_connectedbackslashbiasescolon0"];
    isa(net, Array) ? (net = net * W0 .+ b0') : (net = net *W0 + b0)
    isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
    #-------------------------------------------------------------------
    W1 = aedictdefault["defaultbackslashfully_connected_1backslashweightscolon0"]
    b1 = aedictdefault["defaultbackslashfully_connected_1backslashbiasescolon0"];
    isa(net, Array) ? (net = net * W1 .+ b1') : (net = net *W1 + b1)
    isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
    #-------------------------------------------------------------------
    W2 = aedictdefault["defaultbackslashfully_connected_2backslashweightscolon0"]
    b2 = aedictdefault["defaultbackslashfully_connected_2backslashbiasescolon0"];
    isa(net, Array) ? (net = net * W2 .+ b2') : (net = net *W2 + b2)
    return net
  end
end
```

## Advance: Use Neural Network Implementations from Python Script/Modules

If you have a Python implementation of a neural network architecture and want to use that architecture, we do not need to reimplement it in ADCME. Instead, we can use the `PyCall.jl` package and import the functionalities. For example, if you have a Python package `nnpy` and it has a function `magic_neural_network`. We can use the following code to call `magic_neural_network`

```julia
using PyCall
using ADCME

nnpy = pyimport("nnpy")

x = constant(rand(100,2))
y = nnpy.magic_neural_network(x)
```

Because all the runtime computation are conducted in C++, there is no harm to performance using this mechanism.  