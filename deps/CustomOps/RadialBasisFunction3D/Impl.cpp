
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include <iostream>

using Eigen::ArrayXd;
using Eigen::VectorXd;

namespace RBF3D{

    ArrayXd gaussian(const ArrayXd r, const ArrayXd eps){
        return exp(-(eps*r)*(eps*r));
    }
    ArrayXd multiquadric(const ArrayXd r, const ArrayXd eps){
        return sqrt(1+(eps*r)*(eps*r));
    }
    ArrayXd inversequadratic(const ArrayXd r, const ArrayXd eps){
        return 1./(1+(eps*r)*(eps*r));
    }
    ArrayXd inversemultiquadric(const ArrayXd r, const ArrayXd eps){
        return 1/sqrt(1+(eps*r)*(eps*r));
    }

    ArrayXd func(const ArrayXd r, const ArrayXd eps, int kind){
        if (kind==0)
            return gaussian(r, eps);
        else if (kind==1)
            return multiquadric(r, eps);
        else if (kind==2)
            return inversequadratic(r, eps);
        else 
            return inversemultiquadric(r, eps);
    }

    void grad_gaussian(const ArrayXd r, const ArrayXd eps, ArrayXd &deps, ArrayXd &dr){
        deps = exp(-(eps*r)*(eps*r)) * (-2 * r * r * eps);
        dr = exp(-(eps*r)*(eps*r)) * (-2 * r * eps * eps);
    }

    void grad_multiquadric(const ArrayXd r, const ArrayXd eps, ArrayXd &deps, ArrayXd &dr){
        ArrayXd J = sqrt(1+(eps*r)*(eps*r));
        deps = eps * r * r / J;
        dr = eps * eps * r / J;
    };
    void grad_inversequadratic(const ArrayXd r, const ArrayXd eps, ArrayXd &deps, ArrayXd &dr){
        ArrayXd J = inversequadratic(r, eps);
        J = - J * J ;
        deps = 2 * eps * r * r * J;
        dr = 2 * eps * eps * r * J;
    };
    void grad_inversemultiquadric(const ArrayXd r, const ArrayXd eps, ArrayXd &deps, ArrayXd &dr){
        ArrayXd J = inversequadratic(r, eps);
        J = - pow(J, 1.5);
        deps = eps * r * r * J;
        dr = eps * eps * r * J;
    }

    void grad(const ArrayXd r, const ArrayXd eps, ArrayXd &deps, ArrayXd &dr, int kind){
        if (kind==0)
            return grad_gaussian(r, eps, deps, dr);
        else if (kind==1)
            return grad_multiquadric(r, eps, deps, dr);
        else if (kind==2)
            return grad_inversequadratic(r, eps, deps, dr);
        else 
            return grad_inversemultiquadric(r, eps, deps, dr);
    }
}


namespace RBF3D{
    void forward(
        double *out_, const double *x_, const double *y_, const double *z_,
        const double *eps_, const double *xc_, const double *yc_, const double *zc_, 
        const double *c_, int nc, const double *d, int nd, int nxyz, int kind){
        
        Eigen::Map<const ArrayXd> xc(xc_, nc);
        Eigen::Map<const ArrayXd> yc(yc_, nc);
        Eigen::Map<const ArrayXd> zc(zc_, nc);
        Eigen::Map<const ArrayXd> eps(eps_, nc);
        Eigen::Map<const ArrayXd> c(c_, nc);

        for (int i = 0; i < nxyz; i++){
            ArrayXd r = sqrt((x_[i]-xc).square() + (y_[i]-yc).square() + (z_[i]-zc).square());
            ArrayXd f = func(r, eps, kind);
            double s = (f * c).sum();
            if (nd == 1){
                s += d[0];
            }else if (nd==4){
                s += d[0] + d[1] * x_[i] + d[2] * y_[i] + d[3] * z_[i];
            }else if (nd==0){

            }else{
                throw "Invalid specification for linear term\n";
            }
            out_[i] = s;
        }
    

    }


    void backward(
        double *grad_xc, double *grad_yc, double *grad_zc, 
        double *grad_c, double *grad_d, double *grad_eps,
        const double *grad_out, 
        const double *out_, const double *x_, const double *y_, const double *z_,
        const double *eps_, const double *xc_, const double *yc_, const double *zc_, 
        const double *c_, int nc, const double *d, int nd, int nxyz, int kind){
        
        Eigen::Map<const ArrayXd> xc(xc_, nc);
        Eigen::Map<const ArrayXd> yc(yc_, nc);
        Eigen::Map<const ArrayXd> zc(zc_, nc);
        Eigen::Map<const ArrayXd> eps(eps_, nc);
        Eigen::Map<const ArrayXd> c(c_, nc);
        Eigen::Map<const ArrayXd> out(out_, nxyz);

        ArrayXd deps(nc), dr(nc), dc(nc);
        Eigen::Map<ArrayXd> gc(grad_c, nc);
        Eigen::Map<ArrayXd> geps(grad_eps, nc);
        Eigen::Map<ArrayXd> gxc(grad_xc, nc);
        Eigen::Map<ArrayXd> gyc(grad_yc, nc);
        Eigen::Map<ArrayXd> gzc(grad_zc, nc);

        for (int i = 0; i < nxyz; i++){
            ArrayXd r = sqrt((x_[i]-xc).square() + (y_[i]-yc).square() + (z_[i]-zc).square());

            grad(r, eps, deps, dr, kind);
            ArrayXd basis = func(r, eps, kind);

            gc += basis * grad_out[i];
            geps += grad_out[i] * c * deps;
            gxc += grad_out[i] * c * dr * (xc-x_[i])/r;
            gyc += grad_out[i] * c * dr * (yc-y_[i])/r;
            gzc += grad_out[i] * c * dr * (zc-z_[i])/r;

            if (nd == 1){
                grad_d[0] += grad_out[i];
            }else if (nd==4){
                grad_d[0] += grad_out[i];
                grad_d[1] += grad_out[i] * x_[i];
                grad_d[2] += grad_out[i] * y_[i];
                grad_d[3] += grad_out[i] * z_[i];
            }
        }
    }
}
