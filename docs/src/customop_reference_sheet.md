## Quick Reference for Implementing Julia Custom Operator in ADCMAE

1. Header files

```c++
  #include "julia.h"
  #include "Python.h"
```

2. For Python GIL workround
```c++
PyGILState_STATE py_threadstate;
py_threadstate = PyGILState_Ensure();
...
PyGILState_Release(py_threadstate);
```

3. Get function from Julia main module 
```julia
jl_get_function(jl_main_module, "myfun");
```

4. C++ to Julia
```julia
jl_value_t *a = jl_box_float64(3.0);
jl_value_t *b = jl_box_float32(3.0f);
jl_value_t *c = jl_box_int32(3);
```

5. Julia to C++
```julia
double ret_unboxed = jl_unbox_float64(ret);
float  ret_unboxed = jl_unbox_float32(ret);
int32  ret_unboxed = jl_unbox_int32(ret);
```

6. C++ Arrays to Julia Arrays
```julia
jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
jl_array_t* x          = jl_alloc_array_1d(array_type, 10);
```
or for existing arrays
```julia
double *existingArray = (double*)malloc(sizeof(double)*10);
jl_array_t *x = jl_ptr_to_array_1d(array_type, existingArray, 10, 0);
```

7. Julia Arrays to C++ Arrays
```julia
double *xData = (double*)jl_array_data(x);
```

8. Call Julia Functions
```julia
jl_array_t *y = (jl_array_t*)jl_call1(func, (jl_value_t*)x);
jl_value_t *jl_call(jl_function_t *f, jl_value_t **args, int32_t nargs)
```

9. Gabage collection
```julia
jl_value_t **args;
JL_GC_PUSHARGS(args, 2); // args can now hold 2 `jl_value_t*` objects
args[0] = some_value;
args[1] = some_other_value;
// Do something with args (e.g. call jl_... functions)
JL_GC_POP();
```

Reference:
[Embedding Julia](https://docs.julialang.org/en/v1/manual/embedding/index.html)


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