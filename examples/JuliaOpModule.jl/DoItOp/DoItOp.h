#include <mutex>
#include <thread>
#include <iostream>

std::mutex mu;
void foo()
{
    auto id = std::this_thread::get_id();
 
    mu.lock();
    std::cout << "thread " << id << " sleeping...\n";
    mu.unlock();
 
}

void printjulia(){
    jl_eval_string("println(\"Evaluating Julia script\");");
}

int v;
const double *global_x;
double *global_y;

extern "C" void Cfunction(){
    printf("From Cfunction\n");
    printf("%d, %f\n", v, global_x[0]);
    // jl_eval_string("println(\"Cfunction!\")");
}

void forward(double *y, const double *x, int n){
    PyGILState_STATE py_threadstate;
    py_threadstate = PyGILState_Ensure();
    foo();
    // printf("My id = %d\n\n", jl_threadid());
    v = 10;
    printf("\n*************\n wake up julia \n*************\n");
    global_x = x;
    global_y = y;
    // uv_async_send(uv_async_cond);


    // jl_eval_string("println(1);");
    // jl_value_t **args;
    // JL_GC_PUSHARGS(args, 2); // args can now hold 2 `jl_value_t*` objects
    // args[0] = (jl_value_t*)jl_ptr_to_array_1d(array_type, y, n, 0);
    // args[1] = (jl_value_t*)jl_ptr_to_array_1d(array_type, const_cast<double*>(x), n, 0);
    // auto fun = jl_get_function(jl_main_module, "DoIt!");
    // if (fun==NULL){
    //     jl_errorf("Function not found in Main module.");
    // }
    // else{
    //     jl_call(fun, args, 2);
    // }
    // JL_GC_POP();
    // if (jl_exception_occurred())
    //     printf("%s \n", jl_typeof_str(jl_exception_occurred()));
    PyGILState_Release(py_threadstate);
}