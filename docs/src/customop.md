# Custom Operators

## Basic Usage
Custom operators are ways to add missing features in ADCME. Typically users do not have to worry about custom operators. However, in the following situation custom opreators might be very useful

- Direct implementation in ADCME is inefficient (bottleneck). 
- There are legacy codes users want to reuse, such as GPU-accelerated codes. 
- Special acceleration techniques such as checkpointing scheme. 

In the following, we present an example of implementing the sparse solver custom operator for $Ax=b$.

**Input**: row vector `ii`, column vector`jj` and value vector `vv` for the sparse coefficient matrix; row vector `kk` and value vector `ff`, matrix dimension $d$

**Output**: solution vector $u$


**Step 1: Create and modify the template file**

The following command helps create the wrapper

```julia
customop()
```

There will be a `custom_op.txt` in the current directory. Modify the template file 

```txt
MySparseSolver
int32 ii(?)
int32 jj(?)
double vv(?)
int32 kk(?)
double ff(?)
int32 d()
double u(?) -> output
```

The first line is the name of the operator. It should always be in the camel case. 

The 2nd to the 7th lines specify the input arguments, the signature is `type`+`variable name`+`shape`. For the shape, `()` corresponds to a scalar, `(?)` to a vector and `(?,?)` to a matrix. 

The last line is the output, denoted by ` -> output`. Note there must be a space before and after `->`. 

The following types are accepted: `int32`, `int64`, `double`, `float`, `string`, `bool`. The name of the arguments must all be in *lower cases*. 


**Step 2: Implement core codes**

Run `customop()` again and there will be `CMakeLists.txt`, `gradtest.jl`, `MySparseSolver.cpp` appearing in the current directory. `MySparseSolver.cpp` is the main wrapper for the codes and `gradtest.jl` is used for testing the operator and its gradients. `CMakeLists.txt` is the file for compilation. 

Create a new file `MySparseSolver.h` and implement both the forward simulation and backward simulation (gradients)

```cpp
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <vector>
#include <iostream>
using namespace std;
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

SpMat A;

void forward(double *u, const int *ii, const int *jj, const double *vv, int nv, const int *kk, const double *ff,int nf,  int d){
    vector<T> triplets;
    Eigen::VectorXd rhs(d); rhs.setZero();
    for(int i=0;i<nv;i++){
      triplets.push_back(T(ii[i]-1,jj[i]-1,vv[i]));
    }
    for(int i=0;i<nf;i++){
      rhs[kk[i]-1] += ff[i];
    }
    A.resize(d, d);
    A.setFromTriplets(triplets.begin(), triplets.end());
    auto C = Eigen::MatrixXd(A);
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    auto x = solver.solve(rhs);
    for(int i=0;i<d;i++) u[i] = x[i];
}

void backward(double *grad_vv, const double *grad_u, const int *ii, const int *jj, const double *u, int nv, int d){
    Eigen::VectorXd g(d);
    for(int i=0;i<d;i++) g[i] = grad_u[i];
    auto B = A.transpose();
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(B);
    solver.factorize(B);
    auto x = solver.solve(g);
    // cout << x << endl;
    for(int i=0;i<nv;i++) grad_vv[i] = 0.0;
    for(int i=0;i<nv;i++){
      grad_vv[i] -= x[ii[i]-1]*u[jj[i]-1];
    }
}
```

In this implementation we have used `Eigen` library for solving sparse matrix. Other choices are also possible, such as algebraic multigrid methods. Note here for convenience we have created a global variable `SpMat A;`. This is not recommend if you want to run the code concurrently. 

**Step 3: Compile**

It is recommended that you use the `cmake`, `make` and `gcc` provided by `ADCME`. 
| Variable      | Description                           |
| ------------- | ------------------------------------- |
| `ADCME.CXX`   | C++ Compiler                          |
| `ADCME.CC`    | C Compiler                            |
| `ADCME.TFLIB` | `libtensorflow_framework.so` location |
| `ADCME.CMAKE` | Cmake binary location                 |
| `ADCME.MAKE`  | Make binary location                  |

A simple way is to set the environment by
```bash
export CC=<CC>
export CXX=<CXX>
alias cmake=<CMAKE>
alias make=<MAKE>
```
The values such as `<CC>` are obtained from the last table. Run the following command

```bash
mkdir build
cd build
cmake ..
make -j
```

Based on your operation system, you will create `libMySparseSolver.{so,dylib,dll}`. This will be the dynamic library to link in `TensorFlow`. 

**Step 4: Test**

Finally, you could use `gradtest.jl` to test the operator and its gradients (specify appropriate data in `gradtest.jl` first). If you implement the gradients correctly, you will be able to obtain first order convergence for finite difference and second order convergence for automatic differentiation. 

![custom_op](asset/custom_op.png)

If the process fails, it is most probability the GCC compiler is not compatible with which was used to compile `libtensorflow_framework.{so,dylib}`. In the Linux system, you can check the compiler using 
```bash
readelf -p .comment libtensorflow_framework.so
```
Compatibility issues are frustrating. We hope you can submit an issue to ADCME developers; we are happy to resolve the compatibility issue and improve the robustness of ADCME.


## Best Practice and Caveats


### Allocating Memories

Whenever memory is needed, one should allocate memory by TensorFlow context. 
```cpp
Tensor* tmp_var = nullptr;
TensorShae tmp_shape({10,10});
OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, tmp_shape, &tmp_var));
```

There are three methods to allocate Tensors when an Op kernel executes ([details](https://github.com/tensorflow/tensorflow/blob/584876113e6248639d18d4e16c77b47cb1b251c1/tensorflow/core/framework/op_kernel.h#L753-L801))
- `allocate_persistent`: if the memory is used between Op invocations.
- `allocate_temp`: if the memory is used only within `Compute`.
- `allocate_output`: if the memory will be used as output
