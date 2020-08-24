void TriSolve_forward(double *X, const double *A, 
    const double *b, const double *C, const double *d,
    int n){
    double *B = new double[n];
    double *D = new double[n];
    memcpy(B, b, sizeof(double)*n);
    memcpy(D, d, sizeof(double)*n);
#pragma omp parallel for
    for (int i=1;i<n;i++){
        double w = A[i-1]/B[i-1];
        B[i] = B[i] - w * C[i-1];
        D[i] = D[i] - w * D[i-1];
    }
    X[n-1] = D[n-1]/B[n-1];
    for (int i = n-2; i>-1; i--){
        X[i] = (D[i]-C[i]*X[i+1])/B[i];
    }
    delete [] B;
    delete [] D;
}

void TriSolve_backward(
    double *grad_A, double *grad_B, double *grad_C, double *grad_D,
    const double *grad_X,
    const double *X, const double *A, 
    const double *B, const double *C, const double *D,
    int n){
    TriSolve_forward(grad_D, C, B, A, grad_X, n);    
#pragma omp parallel for
    for(int i = 0; i<n; i++){
        if(i>0) grad_A[i-1] = -grad_D[i] * X[i-1];
        grad_B[i] = -grad_D[i] * X[i];
        if(i<n-1) grad_C[i] = -grad_D[i] * X[i+1];
    }
}