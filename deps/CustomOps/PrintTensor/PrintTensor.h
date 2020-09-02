#include <string>
using std::string;
void print_matrix_forward(double *out, const double *in, string &info, int m, int n){
  memcpy(out, in, sizeof(double)*m*n);
  string s;
  char str[1024];

  s += string("====================================================\n");
  sprintf(str, "Matrix Size = %d x %d Info: %s\n", m, n, info.c_str());
  s += string(str);
  s += string("----------------------------------------------------\n");
  
  for(int i = 0; i<m;i++){
    for(int j = 0; j<n; j++){
      sprintf(str, "%00.04g ", out[i*n+j]);
      s += string(str);
    }
    s += string("\n");
  }
  s += string("====================================================\n");
  printf("%s", s.c_str());
}

void print_matrix_backward(double *grad_in, const double *grad_out, int m, int n){
  memcpy(grad_in, grad_out, sizeof(double)*m*n);
}