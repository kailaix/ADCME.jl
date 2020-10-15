#include "Common.h"
#include <cmath>

namespace RBF{

double gaussian(double r, double eps){
  return exp(-(eps*r)*(eps*r));
}
double multiquadric(double r, double eps){
  return sqrt(1+(eps*r)*(eps*r));
}
double inversequadratic(double r, double eps){
  return 1./(1+(eps*r)*(eps*r));
}
double inversemultiquadric(double r, double eps){
  return 1/sqrt(1+(eps*r)*(eps*r));
}

void grad_gaussian(double r, double eps, double &deps, double &dr){
  deps = exp(-(eps*r)*(eps*r)) * (-2 * r * r * eps);
  dr = exp(-(eps*r)*(eps*r)) * (-2 * r * eps * eps);
}

void grad_multiquadric(double r, double eps, double &deps, double &dr){
  double J = sqrt(1+(eps*r)*(eps*r));
  deps = eps * r * r / J;
  dr = eps * eps * r / J;
};
void grad_inversequadratic(double r, double eps, double &deps, double &dr){
  double J = inversequadratic(r, eps);
  J = - J * J ;
  deps = 2 * eps * r * r * J;
  dr = 2 * eps * eps * r * J;
};
void grad_inversemultiquadric(double r, double eps, double &deps, double &dr){
    double J = inversequadratic(r, eps);
    J = - pow(J, 1.5);
    deps = eps * r * r * J;
    dr = eps * eps * r * J;
  }
}