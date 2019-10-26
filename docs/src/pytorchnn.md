# Neural Network in PyTorch C++

In this section, we describe how we can implement a neural network in C++ with PyTorch APIs. This is useful when we want to create a custom operator in ADCME and a neural network is embedded in the operator (we cannot simply "pass" the neural network to the C++ backend). 

We first need to download [LibTorch](https://pytorch.org/) source. Uncompress the library to your working directory. I have created a simple wrapper for some utility functions in ADCME. To use the wrapper, simply add [la.h](https://github.com/kailaix/ADCME.jl/blob/master/examples/custom_op/headers/la.h) to your include directories.

To create a neural network, the following self-explained C++ code can be used
```c++
struct Net : torch::nn::Module {
  Net() {
    fc1 = register_module("fc1", torch::nn::Linear(3, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::tanh(fc1->forward(x));
    x = torch::tanh(fc2->forward(x));
    x = fc3->forward(x);
    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
```
!!! info 
    To create a linear layer with double precison, run `fc1->to(torch::kDouble)` after construction. 

**Create a Neural Network**
```c++
auto nn = std::make_shared<Net>();
```

**Evaluate an input**
```c++
auto in = torch::rand({8,3},optf.requires_grad(true));
auto out = nn->forward(in);
```
Here we required gradients with respect to the input `in` and put `optf.requires_grad(true)` in the argument.

**Compute Gradients**

To compute gradients, we need to call `backward` of a **scalar** to populate the gradient entries. For example, assume our neural network model is
``y = f_\theta(x)``
and we want to compute $\frac{\partial y}{\partial x}$. In our case, `x` is a $8\times 3$ matrix (8 instances of data, each with 3 features). Each output is 10 dimensional. For each input $x_i\in\mathbb{R}^3$ and each output feature $y_j\in\mathbb{R}$, we want to compute  
``\frac{\partial y_j}{\partial x_i}\in \mathbb{R}^3``
For efficiency, we can compute the gradients of all batches simultaneously, i.e., for all $i$
```c++
auto t = out.sum(0);
t[0].sum().backward();
in.grad().fill_(0.0);
```
where we compute 8 vectors of $\mathbb{R}^3$, i.e., `in.grad()` is a $8\times 3$ matrix (the same size as `in`). 

**Access Neural Network Weights and Biases**

The neural network weights and biases can be assessed with 
```c++
std::cout << nn->fc1->bias << std::endl;
std::cout << nn->fc1->weights << std::endl;
```
We can also manually set the weight values
```c++
nn->fc1->bias.set_data(torch::ones({64}));
```
The grads can also be computed
```c++
std::cout <<  nn->fc1->weight.grad() << std::endl;
```

**Compile**

To compile the script, in `CMakeLists.txt`, we have
```txt
cmake_minimum_required(VERSION 3.5)
project(TorchExample)

set(CMAKE_PREFIX_PATH libtorch)
find_package(Torch REQUIRED)

include_directories(<path/to/la.h>)
add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 11)
```

**Full Script**
```c++
#include "la.h"

struct Net : torch::nn::Module {
  Net() {
    fc1 = register_module("fc1", torch::nn::Linear(3, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::tanh(fc1->forward(x));
    x = torch::tanh(fc2->forward(x));
    x = fc3->forward(x);
    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};



int main(){

    auto nn = std::make_shared<Net>();

    auto in = torch::rand({8,3},optf.requires_grad(true));
    auto out = nn->forward(in);
    
    auto t = out.sum(0);
    t[0].sum().backward();
    in.grad().fill_(0.0);
    std::cout << out << std::endl;

    
    std::cout << nn->fc1->bias << std::endl;
    nn->fc1->bias.set_data(torch::ones({64}));
    std::cout << nn->fc1->bias << std::endl;

    std::cout << nn->fc1->weight << std::endl;
    std::cout <<  nn->fc1->weight.grad() << std::endl;
    
    return 1;
}
```
