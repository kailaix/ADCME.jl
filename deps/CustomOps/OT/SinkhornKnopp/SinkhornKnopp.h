#include "../src/common.h"


void forward(double *l, double *p, const double *a, const double *b, const double *m, int dim_a, int dim_b,
        double reg, int max_iter, double tol, int method){
    auto A = Eigen::Map<const VectorXd>(a, dim_a);
    auto B = Eigen::Map<const VectorXd>(b, dim_b);
    auto M = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(m, dim_a, dim_b);
    MatrixXd Q;
    if (method==0){
        Q = sinkhorn_knopp(A, B, M, reg, max_iter, tol);
    }else if(method==1){
        Q = greenkhorn(A, B, M, reg, max_iter, tol);
    }else{
      throw "Method not implemented";
    }
      
    int k = 0;
    *l = 0.0;
    for(int i=0;i<dim_a;i++){
      for(int j=0;j<dim_b;j++){
        *l += Q(i, j) * M(i, j);
        p[k++] = Q(i, j);
      }
    }

}

void backward(double *dm, const double *p, int dim_a, int dim_b, int method){
    for(int i=0;i<dim_a*dim_b;i++) dm[i] = p[i];
}