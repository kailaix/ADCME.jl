#include "SparseAccumulate.h"

int get_unique_id(){
  srand (time(NULL));
  return rand();
}

SparseAccum::SparseAccum(int handle_): n(0), handle(handle_){
}

void SparseAccum::push_back(const int*cols, const double*vals, int row, int N){
    // std::lock_guard<std::mutex> lck(mu);
    // printf("++++%d %d\n", row,N);
    for(int i=0;i<N;i++){
      // printf("push %d: %d, %d, %f\n", i, row, cols[i], vals[i]);
      if(std::abs(vals[i])>tol){
        if (row > n) {
          char info[1024];
          sprintf(info, "row_id out of the bounds %d > %d\n", row, n);  
          VLOG(FATAL) << info;
          return;
        }
        jj[row-1].push_back(cols[i]);
        vv[row-1].push_back(vals[i]);
      }
    }
}


int SparseAccum::get_n(){ 
    int k = 0;
    for(int i=0;i<jj.size();i++) 
      k+=jj[i].size();
    return k;
}

void SparseAccum::copy_to(int*rows, int*cols, double*vals, int num){
    // std::lock_guard<std::mutex> lck(mu);
    int j = 0;
    for(int i=0;i<n;i++){
      for(int k=0;k<jj[i].size();k++){
          rows[j] = i+1;
          cols[j] = jj[i][k];
          vals[j] = vv[i][k];
          j++;
      }
    }
}

void SparseAccum::initialize(int nrows, double tol_){
   jj.clear(); vv.clear();
   tol = tol_;
   n = nrows;
   jj.resize(n);
   vv.resize(n);
}

void SparseAccum::print(){
}

int create_sparse_assembler(map<int, SparseAccum*>& sa, int h, int nrow, double tol){
  if (sa.count(h)==0){
    auto s = new SparseAccum(h);
    sa[h] = s;
    sa[h]->initialize(nrow, tol);
    char info[1024];
    sprintf(info, "Create a new sparse assembler [Handle ID = %d] with %d rows and tolerance %g.\n", h, nrow, tol);
    VLOG(INFO) << info; 
    VLOG(INFO) << "Current sparse assembler:";
    for(map<int, SparseAccum*>::iterator iter = sa.begin(); iter != sa.end(); ++iter)
    {
      int k =  iter->first;
      sprintf(info, " %d |", k);
      VLOG(INFO) << info;
    }
  }
  return h;
}

int destroy_sparse_assembler(map<int, SparseAccum*>& sa, int h){
  if (sa.count(h)>0){
    VLOG(INFO) << "destroy_sparse_assembler";
    delete sa[h];
    sa.erase(h);
    return 0;
  }
  return -1;
}

int initialize_sparse_assembler(map<int, SparseAccum*>& sa, int h, int nrow, double tol){
  // printf("initialize_sparse_assembler, %d %d\n", nrow, tol);
  if (sa.count(h)==0){
    return -1;
  }
  sa[h]->initialize(nrow, tol);
  return 0;
}

int accumulate_sparse_assembler(map<int, SparseAccum*>& sa, int h, int row, const int* cols, const double *vals, int N){
  if (sa.count(h)==0){
    return -1;
  }
  sa[h]->push_back(cols, vals, row, N);
  return 0;
}

int copy_sparse_assemlber(map<int, SparseAccum*>& sa, int h, int*rows, int*cols, double*vals){
  // printf("copy_sparse_assemlber\n");
  if (sa.count(h)==0){
    return -1;
  }
  // sa[h]->print();
  int num = sa[h]->get_n();
  // printf("%d --> Copy %d\n", h, num);
  sa[h]->copy_to(rows, cols, vals, num);
  destroy_sparse_assembler(sa, h);
  return 0;
}