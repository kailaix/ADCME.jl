#include "Common.h"

namespace RBF{
  typedef double (*RBF_BASIS)(double, double);
  typedef void (*RBF_DBASIS)(double, double, double&, double&);
  void forward(double *z, const double *x, const double *y, 
          const double *eps, const double *xc, const double *yc,
          const double *c, const double *d, int deg, int N, int nc, int kind){
    RBF_BASIS basis;
    switch (kind)
    {
        case 0:
          basis = gaussian;
          break;
        case 1:
          basis = multiquadric;
          break;
        case 2:
          basis = inversequadratic;
          break;
        case 3:
          basis = inversemultiquadric;
          break;
        default:
          break;
    }

    for(int i = 0; i < N; i++){
        z[i] = 0.0;
        for (int j = 0; j < nc; j++){
          double dist = sqrt((x[i] - xc[j]) * (x[i] - xc[j]) + (y[i] - yc[j]) * (y[i] - yc[j]));
          z[i] += c[j] * basis(dist, eps[j]);
        }
        if (deg == 1) z[i] += d[0];
        if (deg == 3) z[i] += d[0] + d[1] * x[i] + d[2] * y[i];
    }
  }



  void backward(
          double *grad_eps, double *grad_xc, double *grad_yc,
          double *grad_c, double *grad_d,
          const double *grad_z, 
          const double *z, const double *x, const double *y, 
          const double *eps, const double *xc, const double *yc,
          const double *c, const double *d, int deg, int N, int nc, int kind){
    RBF_BASIS basis;
    RBF_DBASIS dbasis;
    switch (kind)
    {
        case 0:
          basis = gaussian;
          dbasis = grad_gaussian;
          break;
        case 1:
          basis = multiquadric;
          dbasis = grad_multiquadric;
          break;
        case 2:
          basis = inversequadratic;
          dbasis = grad_inversequadratic;
          break;
        case 3:
          basis = inversemultiquadric;
          dbasis = grad_inversemultiquadric;
          break;
        default:
          break;
    }

    double deps, ddist;
    for(int i = 0; i < N; i++){
        for (int j = 0; j < nc; j++){
          double dist = sqrt((x[i] - xc[j]) * (x[i] - xc[j]) + (y[i] - yc[j]) * (y[i] - yc[j]));
          dbasis(dist, eps[j], deps, ddist);
          grad_c[j] += grad_z[i] * basis(dist, eps[j]);
          grad_eps[j] += grad_z[i] * c[j] * deps;
          grad_xc[j] += grad_z[i] * c[j] * ddist * (xc[j]-x[i])/dist;
          grad_yc[j] += grad_z[i] * c[j] * ddist * (yc[j]-y[i])/dist;
        }
        if (deg == 1) grad_d[0] += grad_z[i];
        if (deg == 3) {
          grad_d[0] += grad_z[i];
          grad_d[1] += grad_z[i] * x[i];
          grad_d[2] += grad_z[i] * y[i];
        }
    }
  }

}

