#include <functional>
#include <cmath>
#include <torch/torch.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;

auto optd = torch::TensorOptions().dtype(torch::kDouble).layout(torch::kStrided).requires_grad(false);
auto optf = torch::TensorOptions().dtype(torch::kFloat).layout(torch::kStrided).requires_grad(false);
static std::function<Eigen::VectorXd(const Eigen::VectorXd&)> NoPreconditioner = [](const Eigen::VectorXd& x){return x;};

VectorXd dtorch2eigen(const torch::Tensor &v){
    int n = v.size(0);
    auto v_ptr = (double *)v.data_ptr();
    VectorXd u(n);
    for(int i=0;i<n;i++){
        u[i] = v_ptr[i];
    }
    return u;
}

torch::Tensor deigen2torch( VectorXd &v, bool grad=false){
    int n = v.rows();
    auto tv = torch::from_blob(v.data(), torch::IntArrayRef{n}, optd.requires_grad(grad));
    return tv;
}

torch::Tensor darray2torch( double *v, int n, bool grad=false){
    auto tv = torch::from_blob(v, torch::IntArrayRef{n}, optd.requires_grad(grad));
    return tv;
}

/** GMRES **/
void rotmat(double a, double b, double &c, double &s){
    if ( fabs(b) < 1e-10 ){
        c = 1.0;
        s = 0.0;
    }
    else if (fabs(b)>fabs(a)){
        double temp = a/b;
        s = 1.0/sqrt(1.0 + temp*temp);
        c = temp * s;
    }
    else{
        double temp = b/a;
        c = 1.0/sqrt(1.0+temp*temp);
        s = temp * c;
    }
}

template <typename Operation, typename Preconditioner>
VectorXd gmres(Operation & A, const VectorXd &b, Preconditioner &M, int m, int max_it, double tol){
    int iter = 0;
    int n = b.size();
    VectorXd x = VectorXd::Zero(n); // initial guess
    double bnrm2 = b.norm();
    if (bnrm2 == 0.0) bnrm2 = 1.0;
    VectorXd r = b-A(x);
    double error = r.norm()/bnrm2;
    if (error<tol) return x;
    
    MatrixXd V = MatrixXd::Zero(n, m+1), H = MatrixXd::Zero(m+1, m);
    VectorXd cs = VectorXd::Zero(m), sn = VectorXd::Zero(m), e1 = VectorXd::Zero(m+1), s, w, y;
    e1(0) = 1.0;
    double temp;

    for(int iter = 0; iter < max_it; iter++){
        r = b-A(x);
        V.col(0) = r/r.norm(); 
        s = r.norm()*e1;
        for(int i=0;i<m;i++){
            w = A(V.col(i));
            for(int k=0;k<=i;k++){
                H(k, i) = w.dot(V.col(k));
                w -= H(k,i)*V.col(k);
            }
            H(i+1,i) = w.norm();
            V.col(i+1) = w/H(i+1,i);
            for(int k=0;k<=i-1;k++){
                temp = cs(k)*H(k,i) + sn(k)*H(k+1,i);
                H(k+1,i) = -sn(k)*H(k,i) + cs(k)*H(k+1,i);
                H(k,i) = temp;
            }
            rotmat(H(i,i), H(i+1,i), cs(i), sn(i));
            temp = cs(i)*s(i);
            s(i+1) = -sn(i)*s(i);
            s(i) = temp;
            // std::cout << s << std::endl; exit(1);
            H(i,i) = cs(i)*H(i,i) + sn(i)*H(i+1,i);
            H(i+1,i) = 0.0;
            error = fabs(s(i+1))/bnrm2;
            if(error <= tol){
                y = H.topLeftCorner(i+1, i+1).colPivHouseholderQr().solve(s.topRows(i+1));
                x += V.leftCols(i+1)*y;
                break;
            }
        }
        if(error <= tol) break;
        y = H.topLeftCorner(m,m).colPivHouseholderQr().solve(s.topRows(m));
        x += V.leftCols(m)*y;
        r = b-A(x);
        error = r.norm()/bnrm2;
        if(error <= tol) break;
    }
    return x;
}