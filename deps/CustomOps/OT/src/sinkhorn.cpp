#include "common.h"
using std::exp;
MatrixXd sinkhorn_knopp(const VectorXd& a, const VectorXd&  b, 
                const MatrixXd& M, double reg, int numItermax,
                double stopThr){
    
    int dim_a = M.rows(), dim_b = M.cols();
    VectorXd u = VectorXd::Ones(dim_a)/double(dim_a);
    VectorXd v = VectorXd::Ones(dim_b)/double(dim_b);

    MatrixXd K = (-M/reg).array().exp();
    MatrixXd tmp;
    MatrixXd Kp = K.array().colwise() / a.array();

    int cpt = 0;
    double err = 1.0;
    while(err > stopThr && cpt < numItermax){
        // printf("Iter %d, error = %f\n", cpt, err);
        
        auto uprev = u;
        auto vprev = v;

        // std::cout << "u" << std::endl << u << std::endl;
        // std::cout << "v" << std::endl << v << std::endl;
        // std::cout << "K" << std::endl << K << std::endl;
        VectorXd KtU = K.transpose() * u;
        v = (b.array() / KtU.array()).matrix();
        u = (1.0 / (Kp * v).array()).matrix();
        
        // std::cout << "KtU" << std::endl << KtU << std::endl;

        if (    KtU.array().cwiseEqual(0.0).any() ||
                u.array().isNaN().any() || u.array().isInf().any() ||
                v.array().isNaN().any() || v.array().isInf().any()){
            printf("Numerical Error");
            u = uprev;
            v = vprev;
            break;
        }

        if (cpt%10==0){
            tmp = (u.asDiagonal() * K * v.asDiagonal()).colwise().sum();
            // std::cout << tmp << std::endl;
            err = (tmp-b).matrix().norm();
        }
        cpt += 1;
    }

    if (cpt==numItermax){
        printf("SinkHorn did not converge!!!\n");
    }

    // std::cout << u.asDiagonal() * K * v.asDiagonal() << std::endl;
    return u.asDiagonal() * K * v.asDiagonal();
}



MatrixXd greenkhorn(const VectorXd& a, const VectorXd&  b, 
                const MatrixXd& M, double reg, int numItermax,
                double stopThr){
    int dim_a = a.size(), dim_b = b.size();
    MatrixXd K = (-M/reg).array().exp().matrix();
    VectorXd u = VectorXd::Ones(dim_a)/double(dim_a);
    VectorXd v = VectorXd::Ones(dim_b)/double(dim_b);
    MatrixXd G = (K.array().colwise()*u.array()).rowwise() * v.transpose().array();
    VectorXd viol = G.array().rowwise().sum().matrix() - a;
    VectorXd viol_2 = G.array().colwise().sum().matrix().transpose() - b;
    double stopThr_val = 1.0;
    int cpt = 0;

    for(int i=0;i<numItermax;i++){
        viol = G.array().rowwise().sum() - a.array();
        viol_2 = G.array().colwise().sum().transpose() - b.array();
        int i1; double m_viol_1 = viol.array().abs().maxCoeff(&i1);
        int i2; double m_viol_2 = viol_2.array().abs().maxCoeff(&i2);
        stopThr_val = m_viol_1 > m_viol_2 ? m_viol_1 : m_viol_2;
        if (m_viol_1 > m_viol_2){            
            u[i1] = a[i1] / (K.row(i1)*v);
            G.row(i1) = u[i1] * K.row(i1).transpose().array() * v.array();
        }else{
            v[i2] = b[i2] / (K.col(i2).transpose()*u);
            G.col(i2) = u.array() * K.col(i2).array() * v[i2];
        }
        if (stopThr_val <= stopThr) break;
        cpt += 1;
    }
    if (cpt==numItermax){
        printf("GreenHorn did not converge!!!\n");
    }
    return G;

}