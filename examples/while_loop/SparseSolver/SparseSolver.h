#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <vector>
#include <iostream>
using namespace std;
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;


SpMat A;

void forward(double *u, const int *ii, const int *jj, const double *vv, int nv, const int *kk, const double *ff,int nf,  int d){
    vector<T> triplets;
    Eigen::VectorXd rhs(d); rhs.setZero();
    for(int i=0;i<nv;i++){
      triplets.push_back(T(ii[i]-1,jj[i]-1,vv[i]));
    }
    for(int i=0;i<nf;i++){
      rhs[kk[i]-1] += ff[i];
    }
    A.resize(d, d);
    A.setFromTriplets(triplets.begin(), triplets.end());
    auto C = Eigen::MatrixXd(A);
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    auto x = solver.solve(rhs);
    for(int i=0;i<d;i++) u[i] = x[i];
}

void backward(double *grad_vv, const double *grad_u, const int *ii, const int *jj, const double *u, int nv, int d){
    Eigen::VectorXd g(d);
    for(int i=0;i<d;i++) g[i] = grad_u[i];
    auto B = A.transpose();
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(B);
    solver.factorize(B);
    auto x = solver.solve(g);
    // cout << x << endl;
    for(int i=0;i<nv;i++) grad_vv[i] = 0.0;
    for(int i=0;i<nv;i++){
      grad_vv[i] -= x[ii[i]-1]*u[jj[i]-1];
    }
}

