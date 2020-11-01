#include <algorithm>


void InterpDimOneForward(double *yi, const double *x,const double *Y, const double *xi, int ni, int Ni, OpKernelContext* context){
  for (int i = 0; i < ni; i++){
    if (xi[i] < x[0] || xi[i] > x[Ni-1]){
      char str[1024];
      sprintf(str, "%d-th interpolation point %f is not in the range (%f, %f)", i+1, xi[i], x[0], x[Ni-1]);
      OP_REQUIRES_OK(context, 
        Status(error::Code::INTERNAL, str));
      return;
    }

    if (fabs(xi[i] - x[0])<1e-16){
      yi[i] = Y[0];
      continue;
    }
    if (fabs(xi[i] - x[Ni-1])<1e-16){
      yi[i] = Y[Ni-1];
      continue;
    }

    const double *xp = std::upper_bound(x, x+Ni, xi[i]);
    int c = xp - x;
    double x2 = x[c], x1 = x[c-1];
    double y2 = Y[c], y1 = Y[c-1];
    yi[i] = (y2-y1)/(x2-x1) * (xi[i]-x1) + y1;
  }
}


void InterpDimOneBackward(
  double *grad_Y, 
  const double *grad_yi, 
  const double *yi, const double *x,const double *Y, const double *xi, int ni, int Ni){
  for (int i = 0; i < ni; i++){
    if (fabs(xi[i] - x[0])<1e-16){
      grad_Y[0] += grad_yi[i];
      continue;
    }
    if (fabs(xi[i] - x[Ni-1])<1e-16){
      grad_Y[Ni-1] += grad_yi[i];
      continue;
    }

    const double *xp = std::upper_bound(x, x+Ni, xi[i]);
    int c = xp - x;
    double x2 = x[c], x1 = x[c-1];
    double y2 = Y[c], y1 = Y[c-1];
    grad_Y[c] += (xi[i]-x1)/(x2-x1) * grad_yi[i];
    grad_Y[c-1] += (x2-xi[i])/(x2-x1)* grad_yi[i];
  }
}