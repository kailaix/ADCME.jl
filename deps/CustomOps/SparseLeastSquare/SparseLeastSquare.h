#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <vector>
#include <iostream>
typedef Eigen::Map<const Eigen::MatrixXd> MapTypeConst;   // a read-only map
using namespace std;
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

void forward(double *u, const int *ii, const int *jj, const double *vv, int nv, const double *ff,int m,  int n){
    printf("DEBUG: Matrix Size (%d, %d)\n", m, n);
    vector<T> triplets;
    Eigen::VectorXd rhs(m); rhs.setZero();
    for(int i=0;i<m;i++) rhs[i] = ff[i];
    for(int i=0;i<nv;i++){
      // printf("DEBUG: %d %d %f\n", ii[i]-1,jj[i]-1,vv[i]);
      triplets.push_back(T(ii[i]-1,jj[i]-1,vv[i]));
    }
    SpMat A;
    A.resize(m, n);
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::LeastSquaresConjugateGradient<SpMat> solver;
    solver.compute(A);
    Eigen::VectorXd x = solver.solve(rhs);

    for(int i=0;i<n;i++) u[i] = x[i];
}

void backward(double *grad_vv, double *grad_f, const double *grad_u, const int *ii, const int *jj, const double *vv, const double *u, const double*f,  int nv, int m, int n){
    vector<T> triplets;
    for(int i=0;i<nv;i++){
      triplets.push_back(T(ii[i]-1,jj[i]-1,vv[i]));
    }
    SpMat A;
    A.resize(m, n);
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::VectorXd uvec(n);
    for(int i=0;i<n;i++) uvec[i] = u[i];
    Eigen::VectorXd t = A*uvec;
    Eigen::VectorXd g(n);
    for(int i=0;i<n;i++) g[i] = grad_u[i];
    SpMat At = A.transpose();
    Eigen::MatrixXd M = At*A;
    Eigen::VectorXd x = M.fullPivLu().solve(g);
    Eigen::VectorXd gf = A*x;
    for(int i=0;i<m;i++) grad_f[i] = gf[i];

    // processing gradient w.r.t. vv
    for(int i=0;i<nv;i++) grad_vv[i] = 0.0;
    for(int i=0;i<nv;i++){
      grad_vv[i] += -t[ii[i]-1]*x[jj[i]-1];
      grad_vv[i] += f[ii[i]-1]*x[jj[i]-1];
      // iterate over i-th row
      for (SpMat::InnerIterator it(At,ii[i]-1); it; ++it)
      {
        grad_vv[i] += - u[jj[i]-1] * x[it.row()] * it.value();
      }
    }
}
