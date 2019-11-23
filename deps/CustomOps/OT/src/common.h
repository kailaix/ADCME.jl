#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <vector>
#include <iostream>
using namespace std;

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

MatrixXd sinkhorn_knopp(const VectorXd& a, const VectorXd&  b, 
                const MatrixXd& M, double reg, int numItermax=1000,
                double stopThr=1e-8);

MatrixXd greenkhorn(const VectorXd& a, const VectorXd&  b, 
                const MatrixXd& M, double reg, int numItermax=1000,
                double stopThr=1e-8);