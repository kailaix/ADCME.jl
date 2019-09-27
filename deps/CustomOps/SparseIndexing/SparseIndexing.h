#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <vector>
#include <iostream>
using namespace std;
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;


// WARNING: BASE=0
void forward(const int64 *ii1, const int64 *jj1, const double *vv1, int64 n1, int64 m, int64 n, 
            const int64 *ii, int64 nz_ii, const int64 *jj, int64 nz_jj, SpMat* C){
    std::map<int64, int64> ii_map;
    std::map<int64, int64> jj_map;
    std::vector<int64> ii_vec;
    for(int64 i=0;i<nz_ii;i++){
      ii_map[ii[i]] = i;
      ii_vec.push_back(ii[i]);
    }
    for(int64 i=0;i<nz_jj;i++) jj_map[jj[i]] = i;
    
    vector<T> triplets1;
    for(int64 i=0;i<n1;i++){
      triplets1.push_back(T(ii1[i]-1,jj1[i]-1,vv1[i]));  
    }
    SpMat A(m, n);
    A.setFromTriplets(triplets1.begin(), triplets1.end());

    vector<T> triplets2;
    int64 p = 0;
    for (int64 k = 0; k < nz_jj; ++k){
      int64 col = jj[k];
      vector<int64> rows;
      std::map<int64, double> val_map;
      vector<int64> v(nz_ii);
      // printf("DEBUG: %d %d\n", k, col);
      for (SpMat::InnerIterator it(A, col-1); it; ++it){
          rows.push_back(it.row()+1);
          val_map[it.row()+1] = it.value(); 
      }
      // assumption: ii, rows are properly sorted
      std::sort(ii_vec.begin(),ii_vec.end());
      auto it = std::set_intersection(ii_vec.begin(),ii_vec.end(), rows.begin(), rows.end(), v.begin());
      v.resize(it-v.begin());
      for (it=v.begin(); it!=v.end(); ++it){
        // printf("DEBUG: %d (%d), %d (%d) --> %f\n", ii_map[*it], *it, jj_map[col], col, val_map[*it]);
        triplets2.push_back(T(ii_map[*it], jj_map[col], val_map[*it]));
      }
    }
    C->setFromTriplets(triplets2.begin(), triplets2.end());
    
}
