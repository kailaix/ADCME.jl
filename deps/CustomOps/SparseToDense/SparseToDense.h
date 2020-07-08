void forward(double *A, const int64*ij, const double *vv, int N, int m, int n){
  for(int i=0;i<N;i++){
    int ii = ij[2*i], jj = ij[2*i+1];
    A[ii*n+jj] += vv[i];
  }
}

void backward(
  double *grad_vv,
  const double *grad_A, 
  const int64*ij,
  int N, int m, int n){
    for(int i=0;i<N;i++){
       int ii = ij[2*i], jj = ij[2*i+1];
       grad_vv[i] = grad_A[ii*n+jj];
    }
}