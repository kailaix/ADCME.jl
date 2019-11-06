#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <vector>
#include <iostream>
using namespace std;
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

void forward(double *u, const int64 *ii, const int64 *jj, const double *vv, int64 nv, const int64 *kk, const double *ff,int64 nf,  int64 d){
    vector<T> triplets;
    Eigen::VectorXd rhs(d); rhs.setZero();
    for(int64 i=0;i<nv;i++){
      triplets.push_back(T(ii[i]-1,jj[i]-1,vv[i]));
    }
    for(int64 i=0;i<nf;i++){
      rhs[kk[i]-1] += ff[i];
    }
    SpMat A;
    A.resize(d, d);
    A.setFromTriplets(triplets.begin(), triplets.end());
    auto C = Eigen::MatrixXd(A);
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    auto x = solver.solve(rhs);
    for(int64 i=0;i<d;i++) u[i] = x[i];
}

void backward(double *grad_ff, double *grad_vv, const double *grad_u, 
    const int64 *ii, const int64 *jj, const double *vv, const double *u, 
      int64 nv, int64 d, const int64 *kk, int64 nf){
    Eigen::VectorXd g(d);
    for(int64 i=0;i<d;i++) g[i] = grad_u[i];

    vector<T> triplets;
    Eigen::VectorXd rhs(d); rhs.setZero();
    for(int64 i=0;i<nv;i++){
      triplets.push_back(T(ii[i]-1,jj[i]-1,vv[i]));
    }
    SpMat A;
    A.resize(d, d);
    A.setFromTriplets(triplets.begin(), triplets.end());
    auto B = A.transpose();
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(B);
    solver.factorize(B);
    auto x = solver.solve(g);
    // cout << x << endl;
    for(int64 i=0;i<nv;i++) grad_vv[i] = 0.0;
    for(int64 i=0;i<nv;i++){
      grad_vv[i] -= x[ii[i]-1]*u[jj[i]-1];
    }

    for(int64 i=0;i<nf;i++) grad_ff[i] = 0.0;
    for(int64 i=0;i<nf;i++) grad_ff[i] += x[kk[i]-1];

}

