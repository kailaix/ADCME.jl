# Neural Network in C++

In this section, we describe how we can implement a neural network in C++. This is useful when we want to create a custom operator in ADCME and a neural network is embedded in the operator (we cannot simply "pass" the neural network to the C++ backend). 

## PyTorch 

The first method is by using PyTorch C++ APIs.

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
!!! note 
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


## Adept-2

The second approach is to use a third-party automatic differentiation library. Here we use [Adept-2](https://github.com/rjhogan/Adept-2). The idea is that we code a neural network in `C++` using array operations. 

To start with, download and compile Adept-2 with the following command
```bash
if [ ! -d "Adept-2" ]; then
  git clone https://github.com/rjhogan/Adept-2
fi
cd Adept-2
autoreconf -i
./configure
make -j
make check
make install
```

All the libraries should be available in `Adept-2/adept/.libs`. The following script shows a simple neural network implementation
```c++
#include "adept.h"
#include "adept_arrays.h"
#include <iostream>
using namespace adept;
using namespace std;

int main()
{
  Stack stack;
  Array<2, double, true> X(100,3), W1(3,20), W2(20,20), W3(20,4);
  Array<1, double, true> b1(20), b2(20), b3(4);
  double V[400];
  for(int i=0;i<300;i++) X[i] = 0.01*i;
  for(int i=0;i<60;i++) W1[i] = 0.01*i;
  for(int i=0;i<400;i++) W2[i] = 0.01*i;
  for(int i=0;i<80;i++) W3[i] = 0.01*i;
  for(int i=0;i<20;i++) b1[i] = 0.01*i;
  for(int i=0;i<20;i++) b2[i] = 0.01*i;
  for(int i=0;i<4;i++) b3[i] = 0.01*i;

  stack.new_recording();
  auto x = X**W1;
  Array<2, double, true> y1(x.size(0),x.size(1));
  for(int i=0;i<x.size(0);i++) 
    for(int j=0;j<x.size(1);j++)
        y1(i,j) = tanh(x(i,j)+b1(j));
    
  auto w = y1**W2;
  Array<2, double, true> y2(w.size(0),w.size(1));
  for(int i=0;i<w.size(0);i++) 
    for(int j=0;j<w.size(1);j++)
        y2(i,j) = tanh(w(i,j)+b2(j));

  auto z = y2**W3;
  Array<2, double, true> y3(z.size(0),z.size(1));
  for(int i=0;i<z.size(0);i++) 
    for(int j=0;j<z.size(1);j++)
        y3(i,j) = z(i,j)+b3(j);
  
  auto out = sum(y3, 0);
  
  out[0].set_gradient(1.0);
  stack.compute_adjoint();
  auto g1 = X.get_gradient();
  cout << g1 << endl;

  auto g2 = W3.get_gradient();
  cout << g2 << endl;

  out[1].set_gradient(1.0);
  stack.compute_adjoint();
  auto g1_ = X.get_gradient();
  cout << g1_ << endl;

  auto g2_ = W3.get_gradient();
  cout << g2_ << endl;


  y3(0,0).set_gradient(1.0);
  stack.compute_adjoint();
  auto g1__ = X.get_gradient();
  cout << g1__ << endl;
  auto g2__ = W3.get_gradient();
  cout << g2__ << endl;
}
```

The following codes might be useful 
```c++
typedef Array<2, double, true> Array2D;
typedef Array<1, double, true> Array1D;

void setv(Array1D& v,  const double *val){
    int n = v.size(0);
    for(int i=0;i<n;i++) v(i).set_value(val[i]);
}

void setv(Array2D& v, const double *val){
    int k = 0;
    int m = v.size(0), n = v.size(1);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++)
            v(i,j).set_value(val[k++]);
    }
}

void getg(Array1D& v, double *val){
    int n = v.size(0);
    auto gv = v.get_gradient();
    for(int i=0;i<n;i++) val[i] = value(gv(i));
}

void getg(Array2D& v, double *val){
    int k = 0;
    int m = v.size(0), n = v.size(1);
    auto gv = v.get_gradient();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++)
            val[k++] = value(gv(i, j));
    }
}
```

