#include "stdio.h"
#include "init_julia.h"

extern "C" void forward(jl_array_t*x,jl_array_t*b1,jl_array_t*w1,jl_array_t*b2,jl_array_t*w2,jl_array_t*y);
extern "C" void backward(jl_array_t*x,jl_array_t*b1,jl_array_t*w1,jl_array_t*b2,jl_array_t*w2,jl_array_t*y);

int main(int argc, char *argv[])
{
    static const int n = 3;
    double b1[3] = {1.0,1.0,1.0};
    double b2[3] = {1.0,1.0,1.0};
    double x[3] = {0.0,0.0,0.0};
    double w1[9] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    double w2[9] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    init_jl_runtime();
    jl_value_t **args;
    JL_GC_PUSHARGS(args, 7);

    args[0] = darray( x, n);
    args[1] = darray( b1, n);
    args[3] = darray( b2, n);
    args[5]  = darray(n);
    args[6] = darray(n, n);
    args[2] = darray(n, n);
    args[4] = darray(n, n);
    
    double *p1 = dpointer(args[2]);
    double *p2 = dpointer(args[4]);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++){
            p1[j*n+i] = w1[i*n+j];
            p2[j*n+i] = w2[i*n+j];
        }

    forward((jl_array_t*)args[0],(jl_array_t*) args[1], (jl_array_t*)args[2],(jl_array_t*)args[3],(jl_array_t*)args[4],
        (jl_array_t*)args[5]);
    
    backward((jl_array_t*)args[0],(jl_array_t*) args[1], (jl_array_t*)args[2],(jl_array_t*)args[3],(jl_array_t*)args[4],
            (jl_array_t*)args[6]);
    
    CHECK_ERROR();
    double *out = dpointer(args[5]);
    for(int i=0;i<n;i++){
        printf("%d: %f\n", i, out[i]);
    }
    double *out2 = dpointer(args[6]);
    for(int i=0;i<n*n;i++){
        printf("%d: %f\n", i, out2[i]);
    }
    JL_GC_POP();
    exit_jl_runtime(0);
    printf("Success!\n");
    return 0;

}
