void forward(int64 *ii, int64 *jj, double * uu, const int64 *indexes,  const double *ks, 
              int64 n, int64 edof){
    int k = 0;
    for(int i = 0; i< n; i++){
      for (int p=0;p<edof;p++){
        for(int q=0;q<edof;q++){
            ii[k] = indexes[edof*i+p];
            jj[k] = indexes[edof*i+q];
            uu[k] = ks[edof*edof*i+p+q*edof];  
            k += 1;
        }
      }
    }
}

void backward(double *grad_ks, const double *grad_uu, const double * uu, const int64 *indexes,  const double *ks, 
              int64 n, int64 edof){
    int k = 0;
    for(int i = 0; i< n; i++){
      for (int p=0;p<edof;p++){
        for(int q=0;q<edof;q++){
            grad_ks[edof*edof*i+p+q*edof] = grad_uu[k]; 
            k += 1;
        }
      }
    }
}