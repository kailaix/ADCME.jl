#include <thread>
void forward(double *y, const double *x, int n){
    std::cout << "is initialized = " << jl_is_initialized() << std::endl;
    jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
    jl_value_t **args;
    JL_GC_PUSHARGS(args, 2); // args can now hold 2 `jl_value_t*` objects
    args[0] = (jl_value_t*)jl_ptr_to_array_1d(array_type, y, n, 0);
    args[1] = (jl_value_t*)jl_ptr_to_array_1d(array_type, const_cast<double*>(x), n, 0);
    auto fun = jl_get_function(jl_main_module, "myfun");
    if (fun==NULL){
        jl_errorf("Function not found in Main module.");
    }
    else{
        jl_call(fun, args, 2);
    }
    JL_GC_POP();
    if (jl_exception_occurred())
        printf("%s \n", jl_typeof_str(jl_exception_occurred()));
}

extern "C" void get_id(){
    std::cout << "my thread = " << std::this_thread::get_id() << std::endl;
}