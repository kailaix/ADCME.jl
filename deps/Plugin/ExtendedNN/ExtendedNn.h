#include "adept.h"
#include "adept_arrays.h"
#include <iostream>
using namespace adept;
using namespace std;

int get_num_theta(const int64* config, int m){
  int N = 0;
  for(int i=0;i<m-1;i++){
    N += config[i]*config[i+1] + config[i+1];
  }
  return N;
};


adouble relu(adouble x){
  return fmax(x, 0.0);
};

std::mutex mu;  

#include "ExtendedNnTanh.h"
#include "ExtendedNnReLU.h"
