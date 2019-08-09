#include "uv.h"
#include "julia.h"

extern "C" void init_jl_runtime();
extern "C" void exit_jl_runtime(int retcode);

jl_value_t* darray(const void * a, int size){
    double *val = (double*) malloc(size*sizeof(double));
    double *a_ = (double *)a;
    for(int i=0;i<size;i++){
        val[i] = a_[i];
    }
    jl_value_t* array_type_1d = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
    return (jl_value_t*)jl_ptr_to_array_1d(array_type_1d, val, size, 1);
}

jl_value_t* darray(int n){
    jl_value_t* array_type_1d = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
    return (jl_value_t*)jl_alloc_array_1d(array_type_1d, n);
}

jl_value_t* darray(int m, int n){
    jl_value_t* array_type_2d = jl_apply_array_type((jl_value_t*)jl_float64_type, 2);
    return  (jl_value_t*)jl_alloc_array_2d(array_type_2d, m, n);
}

double * dpointer(jl_value_t* arr){
    return (double *)jl_array_data(arr);
}

void CHECK_ERROR(){
    jl_value_t *e = jl_exception_occurred();
    if(e)
    { 
        jl_value_t* exception = jl_exception_occurred();
        jl_value_t* sprint_fun = jl_get_function(jl_main_module, "sprint");
        jl_value_t* showerror_fun = jl_get_function(jl_main_module, "showerror");
        
        JL_GC_PUSH3(&exception, &sprint_fun, &showerror_fun);
        const char* returned_exception = jl_string_ptr(jl_call2(sprint_fun, showerror_fun, exception));
        printf("Error occurred in Julia: %s\n", returned_exception); 
        JL_GC_POP();
    }
}


/* Useful functions

1. jl_value_t **args; JL_GC_PUSHARGS(args, 7);
2. JL_GC_POP();

*/