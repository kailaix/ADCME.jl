#include "cuda.h"


__global__ void return_double_(int n, double *b, const double*a){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n) b[i] = 2*a[i];
}

void return_double(int n, double *b, const double*a){
    return_double_<<<(n+255)/256, 256>>>(n, b, a);
}