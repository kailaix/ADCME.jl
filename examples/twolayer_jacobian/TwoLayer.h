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
    jl_call(fun, args, 6);
    JL_GC_POP();
    PyGILState_Release(py_threadstate);
}