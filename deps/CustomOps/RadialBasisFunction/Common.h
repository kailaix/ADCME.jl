#include <cmath> 
namespace RBF{
    

    double gaussian(double r, double eps);
    double multiquadric(double r, double eps);
    double inversequadratic(double r, double eps);
    double inversemultiquadric(double r, double eps);
    void grad_gaussian(double r, double eps, double &deps, double &dr);
    void grad_multiquadric(double r, double eps, double &deps, double &dr);
    void grad_inversequadratic(double r, double eps, double &deps, double &dr);
    void grad_inversemultiquadric(double r, double eps, double &deps, double &dr);


    void forward(double *z, const double *x, const double *y, 
          const double *eps, const double *xc, const double *yc,
          const double *c, const double *d, int deg, int N, int nc, int kind);


    void backward(
          double *grad_eps, double *grad_xc, double *grad_yc,
          double *grad_c, double *grad_d,
          const double *grad_z, 
          const double *z, const double *x, const double *y, 
          const double *eps, const double *xc, const double *yc,
          const double *c, const double *d, int deg, int N, int nc, int kind);
}