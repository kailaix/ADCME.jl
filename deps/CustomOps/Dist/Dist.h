void forward(double *m, const double *x, int nx, const double *y, int ny, int order, int d){
    for(int i=0;i<nx;i++){
        for(int j=0;j<ny;j++){
            m[i*ny+j] = 0;
            for(int k=0;k<d;k++){
                if(order==1) m[i*ny+j] += abs(x[i*d+k]-y[j*d+k]);
                else m[i*ny+j] += pow(abs(x[i*d+k]-y[j*d+k]), order);
            }
            if (order!=1) m[i*ny+j] = pow(m[i*ny+j], 1.0/order);
        }
    }
}

inline double dabsxdx(double x){
    return x > 0 ? 1.0 : -1.0;
}

inline double dabsydy(double x){
    return x > 0 ? -1.0 : 1.0;
}

void backward(double *grad_x, double *grad_y, const double * grad_m,
        const double *m, const double *x, int nx, const double *y, int ny, int order, int d){
    for(int i=0;i<nx*d;i++) grad_x[i] = 0.0;
    for(int i=0;i<ny*d;i++) grad_y[i] = 0.0;

    for(int i=0;i<nx;i++){
        for(int j=0;j<ny;j++){
            for(int k=0;k<d;k++){
                // mij = pow( |x1-y1|^o + |x2-y2|^o + ... + |xd-yd|^o  ,1/o)
                if(order==1){                    
                    grad_x[i*d+k] += dabsxdx(x[i*d+k]-y[j*d+k])*grad_m[i*ny+j];
                    grad_y[j*d+k] += dabsydy(x[i*d+k]-y[j*d+k])*grad_m[i*ny+j];
                }else{
                    double cm = pow(m[i*ny+j], order*(1.0/order-1.0)) \
                                * pow(abs(x[i*d+k]-y[j*d+k]),order-1)*grad_m[i*ny+j];
                    grad_x[i*d+k] += dabsxdx(x[i*d+k]-y[j*d+k])*cm;
                    grad_y[j*d+k] += dabsydy(x[i*d+k]-y[j*d+k])*cm;
                }
            }
        }
    }
}