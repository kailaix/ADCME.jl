#include "common.h"

int main(){
    MatrixXd M(2,2);
    M << 1.0,2.0,
        3.0,4.0;
    // MatrixXd M = MatrixXd::Random(2,2);
    // M = (M.array()*M.array()).matrix();
    VectorXd a(2), b(2);
    a << 0.2,0.8;
    b << 0.7,0.3;
    auto Q = sinkhorn_knopp(a, b, M, 1.0);
    std::cout << Q << std::endl;
}

// 0.14 0.06
// 0.56 0.24