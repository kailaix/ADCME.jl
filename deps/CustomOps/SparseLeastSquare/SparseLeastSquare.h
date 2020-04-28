#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseQR>
#include <eigen3/Eigen/SparseLU>


typedef Eigen::Map<const Eigen::MatrixXd> MapTypeConst;   // a read-only map
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

void forward(double *u, const int *ii, const int *jj, const double *vv, int nv, 
    const double *ff,int m,  int n, int nbatch){
    std::vector<T> triplets;
    for(int i=0;i<nv;i++){
      triplets.push_back(T(ii[i]-1,jj[i]-1,vv[i]));
    }
    SpMat A;
    A.resize(m, n);
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SparseQR<SpMat,Eigen::COLAMDOrdering<int>> solver;
    solver.compute(A);

    for(int i=0;i<nbatch;i++){
      Eigen::Map<const Eigen::VectorXd> rhs(ff, m);
      Eigen::VectorXd x = solver.solve(rhs);
      for(int i=0;i<n;i++) u[i] = x[i];
      ff += m;
      u += n;
    }
    
}

void backward(double *grad_vv, double *grad_f, const double *grad_u, const int *ii, const int *jj, 
        const double *vv, const double *u, const double*f,  int nv, int m, int n, int nbatch){
    std::vector<T> triplets;
    for(int i=0;i<nv;i++){
      triplets.push_back(T(ii[i]-1,jj[i]-1,vv[i]));
    }
    SpMat A;
    A.resize(m, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    
    SpMat At = A.transpose();
    SpMat M = At*A;
    // std::cout << M << std::endl;
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(M);
    solver.factorize(M);

    for(int l = 0; l < nbatch; l++){
        Eigen::Map<const Eigen::VectorXd> uvec(u, n);
        Eigen::Map<const Eigen::VectorXd> g(grad_u, n);

        Eigen::VectorXd t = A*uvec;

        Eigen::VectorXd x = solver.solve(g);
        Eigen::VectorXd gf = A*x;
        // std::cout << uvec << std::endl << std::endl;
        // std::cout << g << std::endl << std::endl;
        // std::cout << gf << std::endl << std::endl;
        for(int i=0;i<m;i++) grad_f[i] = gf[i];
        for(int i=0;i<nv;i++){
          grad_vv[i] += -t[ii[i]-1]*x[jj[i]-1];
          grad_vv[i] += f[ii[i]-1]*x[jj[i]-1];
          for (SpMat::InnerIterator it(At,ii[i]-1); it; ++it)
          {
            grad_vv[i] += - u[jj[i]-1] * x[it.row()] * it.value();
          }
        }
        grad_u += n;
        u += n;
        grad_f += m;
        f += m;
    }
    


}
