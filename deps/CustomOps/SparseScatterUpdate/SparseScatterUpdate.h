#include <map>
#include <set>


class IJV{
public:
  std::vector<int64> ii,jj;
  std::vector<double> vv;
  IJV() = default;
  void insert(int64 i, int64 j, double v){
    ii.push_back(i);
    jj.push_back(j);
    vv.push_back(v);
  }
  int get_size(){return ii.size();};
};

void forward(const int64*oii, const int64*ojj, const double *ovv, int on,
            const int64*uii, const int64*ujj, const double *uvv, int un, int m, int n,
            const int64*ii, const int64*jj, int ni, int nj, IJV& ijv){
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

    std::map<std::pair<int64, int64>, int> imap, jmap;
    for(int i=0;i<un;i++){
      imap[ std::make_pair(ii[uii[i]-1], jj[ujj[i]-1]) ] = i;
      grad_uvv[i] = 0.0;
    }
    for(int i=0;i<on;i++){
      jmap[ std::make_pair(oii[i], ojj[i]) ] = i;
      grad_ovv[i] = 0.0;
    }
    
    for(int i=0;i<out_n;i++){
      if (imap.count(std::make_pair(out_ii[i], out_jj[i]))) {
        grad_uvv[ imap[std::make_pair(out_ii[i], out_jj[i])] ] += grad_out_vv[i];
      }
      else if(jmap.count(std::make_pair(out_ii[i], out_jj[i]))){
        grad_ovv[ jmap[std::make_pair(out_ii[i], out_jj[i])] ] += grad_out_vv[i];
      }
    }

}