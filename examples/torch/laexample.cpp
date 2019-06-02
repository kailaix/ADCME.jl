#include "la.h"
#include<cmath>

#include<iostream>
using namespace std;


// solving A*y = b if we know how to apply A
void example1(){
    int n = 100;
    torch::Tensor A = torch::rand(torch::IntArrayRef{n,n}, optd);
    auto B = torch::mm(A, A.transpose(1,0)) + torch::eye(n, optd);
    auto fn = [n, &B](const Eigen::VectorXd& x){
        // return x;
        Eigen::VectorXd z(n), y(n);
        for(int i=0;i<n;i++) z[i] = x[i];
        auto u = torch::from_blob(z.data(), torch::IntArrayRef{n}, optd);
        auto v = torch::mv(B, u);
        auto vptr = (double *)v.data_ptr();  // do it step by step!!!
        for(int i=0;i<n;i++) 
            y[i]=vptr[i];
        return y;
    };
    Eigen::VectorXd b = Eigen::VectorXd::Ones(n);
    int m = 10;
    int max_iter = 500;
    double tol = 1e-12;
    auto sol = gmres(fn, b, NoPreconditioner, m, max_iter, tol);
    cout << "example 1, error = " << (b-fn(sol)).norm()/b.norm() << endl;
}

// Given y = F(x), F: x->x.^4, compute g'*F_x 
void example2(){
    int n = 10;
    auto F = [](const torch::Tensor &x){return torch::pow(x, 4);};
    auto x = torch::ones(n,optd.requires_grad(true));
    auto g = torch::arange(0,n, optd);
    auto l = torch::dot(g, F(x));
    l.backward();
    // cout << x.grad() << endl;
    VectorXd v(10); v<<  0, 4, 8,12,16,20,24,28,32,36;
    auto u = dtorch2eigen(x.grad());
    cout << "example 2, error = " << (u-v).norm() << endl;
}

// given F(x, y) = x^2 - y^3/(1+x) = 0, and g = dl/dy, compute dl/dx = -g'*inv(F_y)*F_x
void example3(){
    int n = 5;
    // Known information: typically we are given g, x, and y
    const VectorXd g = VectorXd::Ones(n);
    VectorXd x = VectorXd(n); 
    x<<1.0,2.0,3.0,4.0,5.0;
    VectorXd y(n);
    for(int i=0;i<n;i++) y[i] = pow(x[i]*x[i]*(x[i]+1.0), 1.0/3.0);
    auto Fn = [](const torch::Tensor&x, const torch::Tensor&y){return torch::pow(x, 2) - torch::pow(y, 3)/(1.0 + x);};

    // first convert everything to pytorch 
    auto tx = deigen2torch(x, true); 
    auto ty = deigen2torch(y, true); 

    // form the apply operator: Fy * v
    auto fn = [n, &Fn, &tx, &ty](const Eigen::VectorXd& f){
        auto F = Fn(tx,ty);
        auto f_ = f;
        auto tf = deigen2torch(f_);
        auto l = torch::dot(F, tf);
        l.backward();
        auto fg = ty.grad();
        auto out = dtorch2eigen(fg);
        ty.grad().fill_(0.0);   // this is very important
        return out;
    };

    // solve inv(Fy)'*g
    int m = 5;
    int max_iter = 500;
    double tol = 1e-12;
    auto val = gmres(fn, g, NoPreconditioner, m, max_iter, tol); 
    auto tval = deigen2torch(val);

    tx.grad().fill_(0.0);
    auto F = Fn(tx,ty);
    auto l = torch::dot(tval, F);
    l.backward();
    auto res = -tx.grad(); //
    auto eres = dtorch2eigen(res);
    
    VectorXd exact_sol(5); exact_sol << 1.0499342082457277, 1.0175237711585172,1.008922214940025 ,1.0054028553482124,1.0036219820057994;
    double loss = 0.0;
    for(int i=0;i<5;i++){
        loss += fabs(eres[i] - exact_sol[i]);
    }
    cout << "example 3, error = " << loss << endl;
}



int main(){
    example1();
    example2();
    example3();
    return 0;
}