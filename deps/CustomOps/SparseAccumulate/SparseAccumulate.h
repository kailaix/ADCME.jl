#include <random>
#include <iostream>
#include <time.h>       /* time */
#include <map>

using std::map;

class SparseAccum {
private:
  double tol; // tolerance
  int handle; // unique handle 
  std::vector<std::vector<int>> jj; // cols
  std::vector<std::vector<double>> vv; // values
  int n; // number of rows
  std::mutex mu; // concurrency 
public:
  explicit SparseAccum(int handle_);
  
  void push_back(const int*cols, const double*vals, int row, int N);
  int get_n(); // return the number of nonzero elements
  void copy_to(int*rows, int*cols, double*vals, int num); // copy all the memory to rows, cols and vals
  void initialize(int nrows, double tol_); // initialize the SparseAccum  
  void print();
};

int get_unique_id();

int create_sparse_assembler(map<int, SparseAccum*>& sa, int h, int nrow, double tol);
int destroy_sparse_assembler(map<int, SparseAccum*>& sa, int h);
int initialize_sparse_assembler(map<int, SparseAccum*>& sa, int h, int nrow, double tol);
int accumulate_sparse_assembler(map<int, SparseAccum*>& sa, int h, int row, const int* cols, const double *vals, int N);
int copy_sparse_assemlber(map<int, SparseAccum*>& sa, int h, int*rows, int*cols, double*vals);


