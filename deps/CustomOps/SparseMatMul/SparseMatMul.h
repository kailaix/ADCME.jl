#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <vector>
#include <iostream>
using namespace std;
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;


// WARNING: BASE=0
void forward(const int64 *ii1, const int64 *jj1, const double *vv1, int n1,
            const int64 *ii2, const int64 *jj2, const double *vv2, int n2,
            int64 m, int64 n, int64 k, SpMat* C){
    vector<T> triplets1;
    for(int64 i=0;i<n1;i++){
      triplets1.push_back(T(ii1[i],jj1[i],vv1[i]));  
    }
    SpMat A(m, n);
    A.setFromTriplets(triplets1.begin(), triplets1.end());

    vector<T> triplets2;
    for(int64 i=0;i<n2;i++){
      triplets2.push_back(T(ii2[i],jj2[i],vv2[i]));  
    }
    SpMat B(n, k);
    B.setFromTriplets(triplets2.begin(), triplets2.end());
    
    // std::cout << "A" << Eigen::MatrixXd(A) << std::endl;

    // std::cout << "B" << Eigen::MatrixXd(B) << std::endl;
    *C = A*B;
}

void forward_diag_sparse(const int64 *ii1, const int64 *jj1, const double *vv1, int n1,
            const int64 *ii2, const int64 *jj2, const double *vv2, int n2,
            int64 m, int64 n, int64 k, SpMat* C){
    vector<T> triplets;
    for(int64 i=0;i<n2;i++){
      triplets.push_back(T(ii2[i],jj2[i],vv2[i]*vv1[ii2[i]]));  
    }
    C->setFromTriplets(triplets.begin(), triplets.end());
}

void forward_sparse_diag(const int64 *ii1, const int64 *jj1, const double *vv1, int n1,
            const int64 *ii2, const int64 *jj2, const double *vv2, int n2,
            int64 m, int64 n, int64 k, SpMat* C){
    vector<T> triplets;
    for(int64 i=0;i<n1;i++){
      triplets.push_back(T(ii1[i],jj1[i],vv1[i]*vv2[jj1[i]]));  
    }
    C->setFromTriplets(triplets.begin(), triplets.end());
}

// void backward(const int64 *ii1, const int64 *jj1, const double *vv1, int n1,
//             const int64 *ii2, const int64 *jj2, const double *vv2, int n2,
//             int64 m, int64 n, int64 k, SpMat* C){
//     vector<T> triplets1;
//     for(int64 i=0;i<n1;i++){
//       triplets1.push_back(T(ii1[i],jj1[i],vv1[i]));  
//     }
//     SpMat A(m, n);
//     A.setFromTriplets(triplets1.begin(), triplets1.end());

//     vector<T> triplets2;
//     for(int64 i=0;i<n2;i++){
//       triplets2.push_back(T(ii2[i],jj2[i],vv2[i]));  
//     }
//     SpMat B(m, n);
//     B.setFromTriplets(triplets1.begin(), triplets1.end());
    
//     *C = A*B;
// }
