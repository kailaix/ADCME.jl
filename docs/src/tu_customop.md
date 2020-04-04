# Advanced: Custom Operators

!!! note
    As a reminder, there are many built-in custom operators in `deps/CustomOps` and they are good resources for understanding custom operators. The following is a step-by-step instruction on how custom operators are implemented. 

## CPU Operators
Custom operators are ways to add missing features in ADCME. Typically users do not have to worry about custom operators. However, in the following situation custom opreators might be very useful

- Direct implementation in ADCME is inefficient (bottleneck). 
- There are legacy codes users want to reuse, such as GPU-accelerated codes. 
- Special acceleration techniques such as checkpointing scheme. 

In the following, we present an example of implementing a sparse solver for $Au=b$ as a custom operator.

**Input**: row vector `ii`, column vector`jj` and value vector `vv` for the sparse coefficient matrix $A$; row vector `kk` and value vector `ff` for the right hand side $b$; the coefficient matrix dimension is $d\times d$

**Output**: solution vector $u\in \mathbb{R}^d$


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

The 2nd to the 7th lines specify the input arguments, the signature is `type`+`variable name`+`shape`. For the shape, `()` corresponds to a scalar, `(?)` to a vector and `(?,?)` to a matrix. The variable names must be in *lower cases*. Additionally, the supported types are: `int32`, `int64`, `float`, `double`, `bool` and `string`. 

The last line is the output, denoted by ` -> output` (do not forget the whitespace before and after `->`).  

!!! note
    If there are non-real type outputs, the corresponding top gradients input to the gradient kernel should be removed. 


**Step 2: Implement the kernels**

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

!!! note
    In this implementation we have used `Eigen` library for solving sparse matrix. Other choices are also possible, such as algebraic multigrid methods. Note here for convenience we have created a global variable `SpMat A;`. This is not recommend if you want to run the code concurrently, since the variable `A` must be overwritten by another concurrent thread. 

**Step 3: Compile**

It is recommended that you use the `cmake`, `make` and `gcc` provided by `ADCME`. The binary locations can be found via

| Variable      | Description                           |
| ------------- | ------------------------------------- |
| `ADCME.CXX`   | C++ Compiler                          |
| `ADCME.CC`    | C Compiler                            |
| `ADCME.TFLIB` | `libtensorflow_framework.so` location |
| `ADCME.CMAKE` | Cmake binary location                 |
| `ADCME.MAKE`  | Make binary location                  |

ADCME will properly handle the environment variable for you. So we always recommend you to compile custom operators using ADCME functions:

First `cd` into your custom operator director (where `CMakeLists.txt` is located), create a directory `build` if it doesn't exist, `cd` into `build`, and do 

```julia-repl
julia> using ADCME
julia> ADCME.cmake()
```
Based on your operation system, you will create `libMySparseSolver.{so,dylib,dll}`. This will be the dynamic library to link in `TensorFlow`. 

**Step 4: Test**

Finally, you could use `gradtest.jl` to test the operator and its gradients (specify appropriate data in `gradtest.jl` first). If you implement the gradients correctly, you will be able to obtain first order convergence for finite difference and second order convergence for automatic differentiation. Note you need to modify this file first, e.g., creating data and modifying the function `scalar_function`. 

![custom_op](assets/custom_op.png)

!!! info 
    If the process fails, it is most probable the GCC compiler is not compatible with which was used to compile `libtensorflow_framework.{so,dylib}`. ADCME downloads a  GCC compiler via Conda for you. However, if you follow the above steps but encounter some problems, we are happy to resolve the compatibility issue and improve the robustness of ADCME. Submitting an issue is welcome.


## GPU Operators


### Dependencies
To create a GPU custom operator, you must have an NVCC compiler and a CUDA toolkit installed on your system. To install NVCC, see [the installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). To check you have successfully installed NVCC, type
```bash
which nvcc
```
It should gives you the location of `nvcc` compiler. 

For quick installation of other dependencies, you can try
```julia
ENV["GPU"] = 1
Pkg.build("ADCME")
```

#### Manual Installation

In case 

- To install CUDA toolkit (if you do not have one), you can install via conda
```julia
using Conda
Conda.add("cudatoolkit", channel="anaconda")
```

- The next step is to cp the CUDA include file to tensorflow include directory. This could be done with 
```julia
using ADCME
gpus = joinpath(splitdir(tf.__file__)[1], "include/third_party/gpus")
if !isdir(gpus)
  mkdir(gpus)
end
gpus = joinpath(gpus, "cuda")
if !isdir(gpus)
  mkdir(gpus)
end
incpath = joinpath(splitdir(strip(read(`which nvcc`, String)))[1], "../include/")
if !isdir(joinpath(gpus, "include"))
    mv(incpath, gpus)
end
```

- Finally, add the CUDA library path to `LD_LIBRARY_PATH`. This can be done by adding the following line to `.bashrc`
```bash
export LD_LIBRARY_PATH=<path>:$LD_LIBRARY_PATH
```
where `<path>` is 
```julia
joinpath(Conda.ROOTENV, "pkgs/cudatoolkit-10.1.168-0/lib/")
```

### File Organization
There should be three files in your source directories
- `MyOp.cpp`: driver file
- `MyOp.cu`: GPU implementation
- `MyOp.h`: CPU implementation

The first two files have been generated for you by `customop()`. The following are two important notes on the implementation.

- In `MyOp.cu`, the implementation usually has the structure

```c++
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;

    __global__ void forward_(const int nthreads, double *out, const double *y, const double *H0, int n){
      for(int i : CudaGridRangeX(nthreads)) {
          // do something here
      }
    }

    void forwardGPU(double *out, const double *y, const double *H0, int n, const GPUDevice& d){
      // forward_<<<(n+255)/256, 256>>>(out, y, H0, n);
      GpuLaunchConfig config = GetGpuLaunchConfig(n, d);
      TF_CHECK_OK(GpuLaunchKernel(
          forward_, config.block_count, config.thread_per_block, 0,
          d.stream(), config.virtual_thread_count, out, y, H0, n));
      }
}
```

- In `MyOp.cpp`, the device information (`const GPUDevice& d` above) is obtained with 
```c++
context->eigen_device<GPUDevice>()
```

!!! info 
    `for(int i : CudaGridRangeX(nthreads))` is interpreted as 
    ```c++
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x)
    ```
    and the kernel launch semantic is equivalent to 
    ```c++
    forward_<<<config.block_count, config.thread_per_block, 0,
                                d.stream()>>>(config.virtual_thread_count,
                                              			out, y, H0, n);
    ```

## Miscellany

### Mutable Inputs
Sometimes we want to modify tensors in place. In this case we can use mutable inputs. Mutable inputs must be [`Variable`](@ref) and it must be forwarded to one of the output. We consider implement a `my_assign` operator, with signature

```
my_assign(u::PyObject, v::PyObject)::PyObject
```

Here `u` is a `Variable` and we copy the data from `v` to `u`. In the `MyAssign.cpp` file, we modify the input and output specifications to 
```c++
.Input("u : Ref(double)")
.Input("v : double")
.Output("w : Ref(double)")
```

In addition, the input tensor is obtained through
```c++
Tensor u = context->mutable_input(0, true);
```
The second argument `lock_held` specifies whether the input mutex is acquired (false) before the operation. Note the output must be a `Tensor` instead of a reference. 

To forward the input, use
```c++
context->forward_ref_input_to_ref_output(0,0);
```

We use the following code snippet to test the program
```julia
my_assign = load_op("./build/libMyAssign","my_assign")
u = Variable([0.1,0.2,0.3])
v = constant(Array{Float64}(1:3))
u2 = u^2
w = my_assign(u,v)
sess = tf.Session()
init(sess)
@show run(sess, u)
@show run(sess, u2)
@show run(sess, w)
@show run(sess, u2)
```

The output is 
```
[0.1,0.2,0.3]
[0.1,0.04,0.09]
[1.0,2.0,3.0]
[1.0,4.0,9.0]
```
We can see that the tensors depending on `u` are also aware of the assign operator. The complete programs can be downloaded here: [CMakeLists.txt](https://kailaix.github.io/ADCME.jl/dev/codes/mutable/CMakeLists.txt), [MyAssign.cpp](https://kailaix.github.io/ADCME.jl/dev/codes/mutable/MyAssign.cpp), [gradtest.jl](https://kailaix.github.io/ADCME.jl/dev/codes/mutable/gradtest.jl).

### Third-party Plugins

ADCME also allows third-party custom operators hosted on Github. To build your own custom operators, implement your own custom operators in a Github repository. The root directory of the repository should have the following files

* `formula.txt`, which tells how ADCME should interact with the custom operator. It is a Julia Pair, which has the format

  ```
  signature => (source_directory, library_name, signature, has_gradient)
  ```

  For example

  ```julia
  "ot_network"=>("OTNetwork", "libOTNetwork", "ot_network", true)
  ```

* `CMakeLists.txt`, which is used for compiling the library. 

Users are free to arrange other source files or other third-party libraries. 

Upon using those libraries in ADCME, users first download those libraries to `deps` directory via

```julia
install("https://github.com/ADCMEMarket/OTNetwork")
```

The official plugins are hosted on `https://github.com/ADCMEMarket`. To get access to the custom operators in ADCME, use

```julia
op = load_system_op("OTNetwork")
```

1. https://on-demand.gputechconf.com/ai-conference-2019/T1-3_Minseok%20Lee_Adding%20custom%20CUDA%20C++%20Operations%20in%20Tensorflow%20for%20boosting%20BERT%20Inference.pdf)



## Troubleshooting

Here are some common errors you might encounter during custom operator compilation:

**Q: The cmake output for the Julia path is empty.**

```text
Julia=
```

**A:** Check whether `which julia` outputs the Julia location you are using. 

**Q: The cmake output for Python path, Eigen path, etc., is empty.**

```text
Python path=
EIGEN_INC=
TF_INC=
TF_ABI=
TF_LIB_FILE=
```

**A:** Update ADCME to the latest version and check whether or not the ADCME compiler string is empty

```julia
using ADCME
ADCME.__STR__
```

**Q: Julia package precompilation errors that seem not linked to ADCME.**

**A:** Remove the corresponding packages using `using Pkg; Pkg.rm(XXX)` and reinstall those packages. 

**Q: Precompilation error linked to ADCME**

```text
ERROR: LoadError: ADCME is not properly built; run `Pkg.build("ADCME")` to fix the problem.
```

**A:** Build ADCME using `Pkg.build("ADCME")`. Exit Julia and open Julia again. Check whether `deps.jl` exists in the `deps` directory of your Julia package (optional).

