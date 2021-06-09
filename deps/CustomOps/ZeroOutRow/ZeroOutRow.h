#include <vector>
#include <set>

class ZeroOutRow{
    private:
        std::vector<int64> ii;
        std::vector<int64> jj;
        std::vector<double> vv;
        const int64 *indices;
        const double *val;
        const int64 *bd;
        int N;
        int nbd;
        std::set<int64> bdset;
    public:
        ZeroOutRow(const int64 *indices,
        const double *val, int N,
        const int64 *bd, int nbd): indices(indices), val(val), N(N), bd(bd), nbd(nbd){
            for (int i = 0; i < nbd; i++){
                bdset.insert(bd[i]-1);
            }
        };
        int forward();
        void move_data(int64 *oindices, double *oval);
        void backward(double *grad_val, const double *grad_oval);
};

int ZeroOutRow::forward(){
    for (int i = 0; i < N; i++){
        if (bdset.count(indices[2*i])) continue;
        else {
            ii.push_back(indices[2*i]);
            jj.push_back(indices[2*i+1]);
            vv.push_back(val[i]);
        }
    }
    return ii.size();
}

void ZeroOutRow::move_data(int64 *oindices, double *oval){
    for(int i = 0; i < ii.size(); i++){
        oindices[2*i] = ii[i];
        oindices[2*i+1] = jj[i];
        oval[i] = vv[i];
    }
}

void ZeroOutRow::backward(double *grad_val, const double *grad_oval){
    int k = 0;
    for (int i = 0; i < N; i++){
        if (bdset.count(indices[2*i])) continue;
        else {
            grad_val[i] = grad_oval[k];
            k++;
        }
    }
}


