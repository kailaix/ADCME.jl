#include "omp.h"
#include <cmath> 

void ComputeSin(double *y, const double *x, int n){

  int N = omp_get_max_threads();
  printf("There are %d OpenMP threads\n", N);

  #pragma omp parallel for
  for (int i = 0; i < n; i++){
    int nt = omp_get_thread_num();
    printf("%d is computing...\n", nt);
    double xi = x[i];
    y[i] = xi - pow(xi, 3)/6.0 + pow(xi, 5)/120.0 - pow(xi, 7)/5040.0;
  }

}