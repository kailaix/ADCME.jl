#include "../lru_cache.h"

void forward(double *out, const double *rhs, int d, int o){
  Eigen::Map<const Eigen::VectorXd> r(rhs, d);
  auto retrieve = cache1.retrieve(o);
  if (!retrieve.second){
      VLOG(WARNING) << "*** Increase Cache size to accommodate more matrices ***";
      return;
  }
  Eigen::SparseLU<SpMat>* solver = retrieve.first;
  Eigen::VectorXd x = solver->solve(r);
  for(int i = 0; i < d; i++) out[i] = x[i];
}

void backward(double *grad_rhs, double *grad_vv, const double *grad_out, const double *out, 
      const int64 *ii, const int64 *jj, int N, int d, int o){
  auto retrieve = cache2.retrieve(o);
  if (!retrieve.second){
      VLOG(WARNING) << "*** Increase Cache size to accommodate more matrices ***";
      return;
  }
  Eigen::SparseLU<SpMat>* solvert = retrieve.first;

  Eigen::Map<const Eigen::VectorXd> RHS(grad_out, d);
  Eigen::VectorXd g = solvert->solve(RHS);
  for(int i=0;i<N;i++) grad_vv[i] = 0.0;
  for(int i=0;i<d;i++) grad_rhs[i] = g[i];
  for(int i=0;i<N;i++){
    grad_vv[i] = -g[ii[i]-1]*out[jj[i]-1];
  }
  // std::cout << Eigen::MatrixXd(g) << std::endl;

}