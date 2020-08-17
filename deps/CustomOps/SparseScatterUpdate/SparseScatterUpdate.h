#include <map>
#include <set>


class IJV_SparseScatterUpdate{
public:
  std::vector<int64> ii,jj;
  std::vector<double> vv;
  IJV_SparseScatterUpdate() = default;
  void insert(int64 i, int64 j, double v){
    ii.push_back(i);
    jj.push_back(j);
    vv.push_back(v);
  }
  int get_size(){return ii.size();};
};

// ii, jj are not necessarily sorted, 
// however, ii, jj must not have repeated index
void forward(const int64*oii, const int64*ojj, const double *ovv, int on,
            const int64*uii, const int64*ujj, const double *uvv, int un, int m, int n,
            const int64*ii, const int64*jj, int ni, int nj, IJV_SparseScatterUpdate& ijv){
    std::set<int64> iset, jset;
    for(int i=0;i<ni;i++) iset.insert(ii[i]);
    for(int i=0;i<nj;i++) jset.insert(jj[i]);
    for (int i=0;i<on;i++){
      if(iset.count(oii[i]) && jset.count(ojj[i])) continue;
      else
        ijv.insert(oii[i], ojj[i], ovv[i]);
    }
    for (int i=0;i<un;i++){
      ijv.insert(ii[uii[i]-1], jj[ujj[i]-1], uvv[i]);
    }
}

void backward(
            double *grad_ovv, double *grad_uvv,
            const double * grad_out_vv,
            const int64*out_ii, const int64*out_jj, const double *out_vv, int out_n,
            const int64*oii, const int64*ojj, const double *ovv, int on,
            const int64*uii, const int64*ujj, const double *uvv, int un, int m, int n,
            const int64*ii, const int64*jj, int ni, int nj){

    std::set<int64> iset, jset;
    for(int i=0;i<ni;i++) iset.insert(ii[i]);
    for(int i=0;i<nj;i++) jset.insert(jj[i]);


    int k0 = 0, k1 = 0,  k2 = 0;
    for (int i=0;i<on;i++){
      if(iset.count(oii[i]) && jset.count(ojj[i])) 
        grad_ovv[k1++] = 0.0;
      else
        grad_ovv[k1++] = grad_out_vv[k0++];
    }
    for (int i=0;i<un;i++){
        grad_uvv[k2++] = grad_out_vv[k0++];
    }

}