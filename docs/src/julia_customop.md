# Julia Custom Operators

!!! warning
    Currently, embedding Julia suffers from multithreading issues: calling Julia from a non-Julia thread is not supported in ADCME. When TensorFlow kernel codes are executed concurrently, it is difficult to invoke the Julia functions. See [issue](https://github.com/kailaix/ADCME.jl/issues/8).

In scientific and engineering applications, the operators provided by `TensorFlow` are not sufficient for high performance computing. In addition, constraining oneself to `TensorFlow` environment sacrifices the powerful scientific computing ecosystems provided by other languages such as `Julia` and `Python`. For example, one might want to code a finite volume method for a sophisticated fluid dynamics problem; it is hard to have the flexible syntax to achieve this goal, obtain performance boost from existing fast solvers such as AMG, and benefit from many other third-party packages within `TensorFlow`. This motivates us to find a way to "plugin" custom operators to `TensorFlow`.



We have already introduced how to incooperate `C++` custom operators.  For many researchers, they usually prototype the solvers in a high level language such as MATLAB, Julia or Python. To enjoy the parallelism and automatic differentiation feature of `TensorFlow`, they need to port them into `C/C++`. However, this is also cumbersome sometimes, espeically the original solvers depend on many packages in the high-level language. 



We solve this problem by incorporating `Julia` functions directly into `TensorFlow`. That is, for any `Julia` functions, we can immediately convert it to a `TensorFlow` operator. At runtime, when this operator is executed, the corresponding `Julia` function is executed. That implies we have the `Julia` speed. Most importantly, the function is perfectly compitable with the native `Julia` environment; third-party packages, global variables, nested functions, etc. all work smoothly. Since `Julia` has the ability to call other languages in a quite elegant and simple manner, such as `C/C++`, `Python`, `R`, `Java`, this means it is possible to incoporate packages/codes from any supported languages into `TensorFlow` ecosystem. We need to point out that in `TensorFlow`, `tf.numpy_function` can be used to convert a `Python` function to a `TensorFlow` operator. However, in the runtime, the speed for this operator falls back to `Python` (or `numpy` operation for related parts). This is a drawback. 



The key for implementing the mechanism is embedding `Julia` in `C++`. Still we need to create a `C++` dynamic library for `TensorFlow`. However, the library is only an interface for invoking `Julia` code. At runtime, `jl_get_function` is called to search for the related function in the main module. `C++` arrays, which include all the relavant data, are passed to this function through `jl_call`. It requires routine convertion from `C++` arrays to `Julia` array interfaces `jl_array_t*`. However, those bookkeeping tasks are programatic and possibly will be automated in the future. Afterwards,`Julia` returns the result to `C++` and thereafter the data are passed to the next operator. 



There are two caveats in the implementation. The first is that due to GIL of Python, we must take care of the thread lock while interfacing with `Julia`. This was done by putting a guard around th e`Julia` interface

```c
PyGILState_STATE py_threadstate;
py_threadstate = PyGILState_Ensure();
// code here 
PyGILState_Release(py_threadstate);
```

The second is the memory mangement of `Julia` arrays. This was done by defining gabage collection markers explicitly

```julia
jl_value_t **args;
JL_GC_PUSHARGS(args, 6); // args can now hold 2 `jl_value_t*` objects
args[0] = ...
args[1] = ...
# do something
JL_GC_POP();
```



This technique is remarkable and puts together one of the best langages in scientific computing and that in machine learning. The work that can be built on `ADCME` is enormous and significantly reduce the development time. 



## Example

Here we present a simple example. Suppose we want to compute the Jacobian of a two layer neural network $\frac{\partial y}{\partial x}$

$$y = W_2\tanh(W_1x+b_1)+b_2$$

where $x, b_1, b_2, y\in \mathbb{R}^{10}$, $W_1, W_2\in \mathbb{R}^{100}$. In `TensorFlow`, this can be done by computing the gradients $\frac{\partial y_i}{\partial x}$ for each $i$. In `Julia`, we can use `ForwardDiff` to do it automatically. 

```julia
function twolayer(J, x, w1, w2, b1, b2)
    f = x -> begin
        w1 = reshape(w1, 10, 10)
        w2 = reshape(w2, 10, 10)
        z = w2*tanh.(w1*x+b1)+b2
    end
    J[:] = ForwardDiff.jacobian(f, x)[:]
end
```

To make a custom operator, we first generate a wrapper

```julia
using ADCME
mkdir("TwoLayer")
cd("TwoLayer")
customop()
```

We modify `custom_op.txt`

```
TwoLayer
double x(?)
double w1(?)
double b1(?)
double w2(?)
double b2(?)
double y(?) -> output
```

and run 

```julia
customop()
```

Three files are generated`CMakeLists.txt`, `TwoLayer.cpp` and `gradtest.jl`. Now create a new file `TwoLayer.h`

```c++
#include "julia.h"
#include "Python.h"

void forward(double *y, const double *x, const double *w1, const double *w2, const double *b1, const double *b2, int n){
    PyGILState_STATE py_threadstate;
    py_threadstate = PyGILState_Ensure();
    jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
    jl_value_t **args;
    JL_GC_PUSHARGS(args, 6); // args can now hold 2 `jl_value_t*` objects
    args[0] = (jl_value_t*)jl_ptr_to_array_1d(array_type, y, n*n, 0);
    args[1] = (jl_value_t*)jl_ptr_to_array_1d(array_type, const_cast<double*>(x), n, 0);
    args[2] = (jl_value_t*)jl_ptr_to_array_1d(array_type, const_cast<double*>(w1), n*n, 0);
    args[3] = (jl_value_t*)jl_ptr_to_array_1d(array_type, const_cast<double*>(w2), n*n, 0);
    args[4] = (jl_value_t*)jl_ptr_to_array_1d(array_type, const_cast<double*>(b1), n, 0);
    args[5] = (jl_value_t*)jl_ptr_to_array_1d(array_type, const_cast<double*>(b2), n, 0);
    auto fun = jl_get_function(jl_main_module, "twolayer");
  	if (fun==NULL) jl_errorf("Function not found in Main module.");
    else jl_call(fun, args, 6);
    JL_GC_POP();
    if (jl_exception_occurred())
        printf("%s \n", jl_typeof_str(jl_exception_occurred()));
    PyGILState_Release(py_threadstate);
}
```

Most of the codes have been explanined except `jl_ptr_to_array_1d`. This function generates a `Julia` array wrapper from `C++` arrays. The last argument `0` indicates that `Julia` is not responsible for gabage collection. `TwoLayer.cpp` should also be modified according to [https://github.com/kailaix/ADCME.jl/blob/master/examples/twolayer_jacobian/TwoLayer.cpp](https://github.com/kailaix/ADCME.jl/blob/master/examples/twolayer_jacobian/TwoLayer.cpp).



Finally, we can test in `gradtest.jl` 

```julia
two_layer = load_op("build/libTwoLayer", "two_layer")


w1 = rand(100)
w2 = rand(100)
b1 = rand(10)
b2 = rand(10)
x = rand(10)
J = rand(100)
twolayer(J, x, w1, w2, b1, b2)

y = two_layer(constant(x), constant(w1), constant(b1), constant(w2), constant(b2))
sess = Session(); init(sess)
J0 = run(sess, y)
@show norm(J-J0)
```



## Embedded in Modules

If the custom operator is intended to be used in a precompiled module, we can load the dynamic library at initialization

```julia
global my_op 
function __init__()
	global my_op = load_op("$(@__DIR__)/path/to/libMyOp", "my_op")
end
```

The corresponding `Julia` function called by `my_op` must be exported in the module (such that it is in the Main module when invoked). One such example is given in [MyModule](https://github.com/kailaix/ADCME.jl/blob/master/examples/JuliaOpModule.jl)

## Quick Reference for Implementing C++ Custom Operators in ADCME

1. Set output shape
```
c->set_output(0, c->Vector(n));
c->set_output(0, c->Matrix(m, n));
c->set_output(0, c->Scalar());
```

2. Names
`.Input` and `.Ouput` : names must be in lower case, no `_`, only letters.

3. TensorFlow Input/Output to TensorFlow Tensors
```
grad.vec<double>();
grad.scalar<double>();
grad.matrix<double>();
grad.flat<double>();
```
Obtain flat arrays
```
grad.flat<double>().data()
```

4. Scalars
Allocate scalars using TensorShape()

5. Allocate Shapes
Although you can use -1 for shape reference, you must allocate exact shapes in `Compute`