#include "cuda.h"

#include <cmath> 
#define THREADS_PER_BLOCK 256

__global__ void ComputeSinKernel(double *y, const double *x, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<n){
        double xi = x[i];
        y[i] = xi - pow(xi, 3)/6.0 + pow(xi, 5)/120.0 - pow(xi, 7)/5040.0; 
    }
}

void ComputeSinGPU(double *y, const double *x, int n){
    printf("I am running on GPU device!\n");
    ComputeSinKernel<<< (n - 1)/THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(y, x, n);
}
