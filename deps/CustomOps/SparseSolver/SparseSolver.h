#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <vector>
#include <iostream>
// #include <chrono> s
// using namespace std::chrono; 

using namespace std;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

// Eigen::SparseLU<SpMat>
template<typename SOLVER>
bool forward(double *u, const int64 *ii, const int64 *jj, const double *vv, int64 nv, 
                   const double *f,  int64 d){
    // std::cout << "*************" << std::endl;
    // auto start0 = high_resolution_clock::now();
    vector<T> triplets;
    Eigen::Map<const Eigen::VectorXd> rhs(f, d); 
    for(int64 i=0;i<nv;i++){
      // printf("%d %d --> %f\n", ii[i],jj[i],vv[i]);
      triplets.push_back(T(ii[i]-1,jj[i]-1,vv[i]));
    }
    SpMat A(d, d);
    A.setFromTriplets(triplets.begin(), triplets.end());
    // auto start1 = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(start1 - start0); 
    // std::cout << duration.count() << std::endl;
    
    SOLVER solver;

    // auto start2 = high_resolution_clock::now();
    // auto duration1 = duration_cast<microseconds>(start2 - start1); 
    // std::cout << duration1.count() << std::endl;

    solver.analyzePattern(A);
    auto info = solver.info();
    if (!info){
      return false;
    }
    solver.factorize(A);
    Eigen::VectorXd x = solver.solve(rhs);

    // auto start3 = high_resolution_clock::now();
    // auto duration2 = duration_cast<microseconds>(start3 - start2); 
    // std::cout << duration2.count() << std::endl;


    for(int i=0;i<(int)d;i++) u[i] = x[i]; // very slow

    // auto start4 = high_resolution_clock::now();
    // auto duration3 = duration_cast<microseconds>(start4 - start3); 
    // std::cout << duration3.count() << std::endl;
    return true;
}

template<typename SOLVER>
void backward(double *grad_f, double *grad_vv, const double *grad_u, 
    const double *u, const int64 *ii, const int64 *jj, const double *vv, int64 nv, 
                   const double *f,  int64 d){

    Eigen::Map<const Eigen::VectorXd> g(grad_u, d);

    vector<T> triplets;
    for(int64 i=0;i<nv;i++){
      triplets.push_back(T(jj[i]-1,ii[i]-1,vv[i]));
    }
    SpMat B(d, d);
    B.setFromTriplets(triplets.begin(), triplets.end());
    
    SOLVER solver;
    solver.analyzePattern(B);
    solver.factorize(B);
    Eigen::VectorXd x = solver.solve(g);
    // cout << x << endl;
    for(int64 i=0;i<nv;i++) grad_vv[i] = 0.0;
    for(int64 i=0;i<nv;i++){
      grad_vv[i] -= x[ii[i]-1]*u[jj[i]-1];
    }
    for(int64 i=0;i<d;i++) grad_f[i] = x[i];
}

