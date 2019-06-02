void forward(double* u, const double* v, const double* w, int N){
    for(int i=0; i<N; i++){
        u[i] = 0.0;
        for(int k=0;k<N;k++){
            u[i] += w[ i*N+k ] * v[k] * v[k];
        }
        u[i] = pow(u[i], 1.0/3.0);
    }
}

void backward(const double* du, const double* v, const double*w,
             double*dv, double*dw, int N, double* pBuffer){
    double *u = pBuffer + N;
    forward(u, v, w, N);
    // step 1: compute D = dJ/du*(dF/du)^{-1}
    for(int i=0;i<N;i++){
        pBuffer[i] = du[i]/(3.0*u[i]*u[i]);
    }
    // step 2: compute D*dF/dv
    for(int i=0;i<N;i++){
        dv[i] = 0.0;
        for(int j=0;j<N;j++){
            dv[i] += 2*w[j*N+i]*v[i]*pBuffer[j];
        }
    }
    // step 3: compute D*dF/dw
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            dw[i*N+ j] = pBuffer[i]*v[j]*v[j];
        }
    }
}

