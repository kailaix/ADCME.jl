#include <set> 
#include <utility>
#include <map>
#include <vector>

using namespace std;

class SparseCompressor{
public:
    int nout; 
    int N;
    const int64 *indices;
    const double *v;
    vector<pair<int64, int64>> keys;
    vector<double> values;
    SparseCompressor(const int64 *indices, const double *v, int N);
    void forward(int64 *nindices, double *nv);
    void backward(double *grad_v, const double *grad_nv);
};

SparseCompressor::SparseCompressor(const int64 *indices, const double *v, int N) :
    N(N), v(v), indices(indices){
    for (int i = 0; i < N; i++){
        auto ij = make_pair(indices[2*i], indices[2*i+1]);
        if (keys.size()>0 && keys[keys.size()-1]==ij){
            values.back() += v[i];
        }
        else{
            keys.push_back(ij);
            values.push_back(v[i]);
        }
    }
    nout = keys.size();
}

void SparseCompressor::forward(int64 *nindices, double *nv){
    int k = 0;
    for (int k = 0; k < values.size(); k++){
        nindices[2*k] = keys[k].first;
        nindices[2*k+1] = keys[k].second;
        nv[k] = values[k];
    }
}

void SparseCompressor::backward(double *grad_v, const double *grad_nv){
    int k = 0;
    for (int i = 0; i < N; i++){
        auto ij = make_pair(indices[2*i], indices[2*i+1]);
        if (keys[k]!=ij){
            k++;
        }
        grad_v[i] = grad_nv[k];
    }
}

#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#else
    #define EXPORT
    #define IMPORT
#endif

// computes the Jacobian matrix for SparseCompressor
// compress is a linear operator 
// J is a N x nout matrix 
extern "C" EXPORT double * pcl_SparseCompressor(const int64 *indices, const double *v, int N, int *nout){
    SparseCompressor sc(indices, v, N);
    int k = 0;
    double *J = (double *)malloc(sizeof(double)*sc.nout*N);
    memset(J,  0, sizeof(double)*sc.nout*N);
    for (int i = 0; i < N; i++){
        auto ij = make_pair(indices[2*i], indices[2*i+1]);
        if (sc.keys[k]!=ij){
            k++;
        }
        J[i + k * N] = 1.0;
    }
    *nout = sc.nout;
    return J;
}