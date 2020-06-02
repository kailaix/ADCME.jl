#include "../lru_cache.h"


void forward(int64 *o, const int64 *ii, const int64 *jj, const double *vv, int N, int d, int n){
  if (cache1.max_size()<n) cache1.resize(n);
  if (cache2.max_size()<n) cache2.resize(n);

  std::vector<T> triplets;
  for (int i = 0;i<N;i++) triplets.push_back(T(ii[i]-1, jj[i]-1, vv[i]));
  SpMat A(d, d);
  A.setFromTriplets(triplets.begin(), triplets.end());
  SpMat B = A.transpose();

  auto solver = new Eigen::SparseLU<SpMat>;
  auto solvert = new Eigen::SparseLU<SpMat>;;
  solver->analyzePattern(A);
  solver->factorize(A);
  // printf("A factorized!\n");


  solvert->analyzePattern(B);
  solvert->factorize(B);

  int id = cache1.get_new_id();

  cache1.put(id, solver);
  cache2.put(id, solvert);
  // printf("B factorized!\n");

  char info[1024];
  sprintf(info, "Factorization: current matrix id= %d, maximum cache size = %d\n", id, cache1.max_size());
  VLOG(INFO) << info; 
  *o = id;
}