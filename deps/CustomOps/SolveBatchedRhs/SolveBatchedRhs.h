#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/QR>

using Eigen::VectorXd;
using Eigen::MatrixXd;

void forward(double *sol, const double*a, const double*rhs, int m, int n, int nbatch){
  MatrixXd A(m, n);
  for (int i=0;i<m;i++){
    for(int j=0;j<n;j++){
      A(i, j) = a[i*n+j];
    }
  }
  auto solver = A.fullPivHouseholderQr();
  for(int i=0;i<nbatch;i++){
    VectorXd f = Eigen::Map<const VectorXd>(rhs, m);
    VectorXd s = solver.solve(f);
    for(int i=0;i<n;i++) sol[i] = s[i];
    rhs += m;
    sol += n; 
  }
}

void backward(
  double * grad_a, double * grad_rhs, 
  const double *grad_sol, 
  const double *sol, const double*a, const double*rhs, int m, int n, int nbatch){
  MatrixXd A(m, n);
  for (int i=0;i<m;i++){
    for(int j=0;j<n;j++){
      A(i, j) = a[i*n+j];
    }
  }
  MatrixXd At = A.transpose();
  
  if (m==n){
    auto solver = At.fullPivHouseholderQr();
    for(int i=0;i<nbatch;i++){
      VectorXd g = Eigen::Map<const VectorXd>(grad_sol, n);
      VectorXd x = solver.solve(g);
      for(int c=0;c<n;c++){
        grad_rhs[c] = x[c];
        for(int r=0;r<m;r++){
          grad_a[r*n+c] -= sol[c] * x[r];
        }
      }
      grad_sol += n; 
      sol += n;
      grad_rhs += m;
    }
  }

  else {
    MatrixXd M = At*A;
    auto solver = M.fullPivHouseholderQr();
    for(int i=0;i<nbatch;i++){
      VectorXd g = Eigen::Map<const VectorXd>(grad_sol, n);
      VectorXd x = solver.solve(g);
      x = A*x;
      for(int r =0;r<m;r++) grad_rhs[r] = x[r];
      for(int c=0;c<n;c++){
        for(int r=0;r<m;r++){
          grad_a[r*n+c] -= sol[c] * x[r];
        }
      }
      grad_sol += n; 
      sol += n;
      grad_rhs += m;
    }
  }
  
}