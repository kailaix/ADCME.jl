#include <cmath> 
namespace RBF3D{
    

    double gaussian(double r, double eps);
    double multiquadric(double r, double eps);
    double inverse(double r, double eps);
    double inversemultiquadric(double r, double eps);
    void grad_gaussian(double r, double eps, double &deps, double &dr);
    void grad_multiquadric(double r, double eps, double &deps, double &dr);
    void grad_inversequadratic(double r, double eps, double &deps, double &dr);
    void grad_inversemultiquadric(double r, double eps, double &deps, double &dr);


    void forward(
        double *out_, const double *x_, const double *y_, const double *z_,
        const double *eps_, const double *xc_, const double *yc_, const double *zc_, 
        const double *c_, int nc, const double *d, int nd, int nxyz, int kind);


    void backward(
        double *grad_xc, double *grad_yc, double *grad_zc, 
        double *grad_c, double *grad_d, double *grad_eps,
        const double *grad_out, 
        const double *out_, const double *x, const double *y, const double *z,
        const double *eps_, const double *xc_, const double *yc_, const double *zc_, 
        const double *c_, int nc, const double *d, int nd, int nxyz, int kind);
}