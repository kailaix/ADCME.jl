#include <algorithm>
void forward_triu(double *v, const double*u, int m, int n, int num, int batch_size){
  for(int b=0;b<batch_size;b++){
    for(int i=0;i<std::min(m, n-num);i++){
      for(int j=std::max(0,i+num);j<n;j++){
        v[i*n+j] = u[i*n+j];
      }
    }
    v += m*n;
    u += m*n;
  }
}

void backward_triu(
  double * grad_u,
  const double *grad_v,
  const double *v, const double*u, int m, int n, int num, int batch_size
){
  for(int b=0;b<batch_size;b++){
    for(int i=0;i<std::min(m, n-num);i++){
      for(int j=std::max(0,i+num);j<n;j++){
        grad_u[i*n+j] = grad_v[i*n+j];
      }
    }
    grad_u += m*n;
    grad_v += m*n;
  }
}


void forward_tril(double *v, const double*u, int m, int n, int num, int batch_size){
  num = -num;
  for(int b=0;b<batch_size;b++){
    for(int i=std::max(0,num);i<m;i++){
      for(int j=0;j<std::min(n, i-num+1);j++){
        v[i*n+j] = u[i*n+j];
      }
    }
    v += m*n;
    u += m*n;
  }
}

void backward_tril(
  double * grad_u,
  const double *grad_v,
  const double *v, const double*u, int m, int n, int num, int batch_size
){
  num = -num;
  for(int b=0;b<batch_size;b++){
    for(int i=std::max(0,num);i<m;i++){
      for(int j=0;j<std::min(n, i-num+1);j++){
        grad_u[i*n+j] = grad_v[i*n+j];
      }
    }
    grad_u += m*n;
    grad_v += m*n;
  }
}