# Neural Networks 

A neural network can be viewed as a computational graph where each operator in the computational graph is composed of linear transformation or simple explicit nonlinear mapping (called _activation functions_). There are essential components of the neural network 

1. Input: the input to the neural network, which is a real or complex valued vector; the input is often called _features_ in machine learning. To leverage dense linear algebra, features are usually aggregated into a matrix and fed to the neural network. 
2. Output: the output of the neural network is also a real or complex valued vectors. The vector can be tranformed to categorical values (labels) based on the specific application. 

The common activations functions include ReLU (Rectified linear unit), tanh, leaky ReLU, SELU, ELU, etc. In general, for inverse modeling in scientific computing, tanh usually outperms the others due to its smoothness and boundedness, and forms a solid choice at the first try. 

A common limitation of the neural network is overfitting. The neural network contains plenty of free parameters, which makes the neural network "memorize" the training data easily. Therefore, you may see very a small training error, but have large test errors. Regularization methods have been proposed to alleviate this problem; to name a few, restricting network sizes, imposing weight regulization (Lasso or Ridge), using Dropout and batch normalization, etc. 

ADCME provides a very simple way to specify a fully connected neural network, [`ae`](@ref) (short for _autoencoder_)

```julia
x = constant(rand(10,2)) # input
config = [20,20,20,3] # hidden layers
θ = ae_init([2;config]) # getting an initial weight-and-biases vector. 

y1 = ae(x, config)
y2 = ae(x, config, θ)
```

In scientific computing, sometimes we not only want to evaluate the neural network output, but also the sensitivity. Specifically, if 

$$y = NN_{\theta}(x)$$

We also want to compute $\nabla_x NN_{\theta}(x)$. ADCME provides a function [`fc`](@ref) (short for _fully-connected_)

```julia
y3, dy3 = fc(x, config, θ)
```

Here `dy3` will be a $10\times 3 \times 2$ tensor, where `dy3[i,:,:]` is the Jacobian matrix of the $i$-th output with respect to the $i$-th input (Note the $i$-th output is independent of $j$-th input, whenever $i\neq j$).  