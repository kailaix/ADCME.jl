#include "cuda.h"

__global__ void multiply_by_two(double *y, const double *x, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        y[i] = 2*x[i];
    }
}


void multiply_by_two_forward(double *y, const double *x, int n){
    multiply_by_two<<<  (n-1)/64 + 1, 64 >>>(y, x, n);
}

void multiply_by_two_backward(double *grad_x, const double *grad_y, int n){
    multiply_by_two<<< (n-1)/64 + 1, 64 >>>(grad_x, grad_y, n);
}