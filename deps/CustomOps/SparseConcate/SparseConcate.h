#include <set>
#include <vector>
using std::set;
using std::vector;

class Forward{
  private:
    vector<int> ii,jj;
    vector<double> vv;
  public:
    Forward(const int64* ii1, const int64* jj1, const double*vv1, int N1, 
        const int64* ii2, const int64* jj2, const double*vv2, int N2, 
        int m1, int n1, int m2, int n2, bool hcat){
      // std::vector<T> triplets;            // list of non-zeros coefficients
      for(int i=0;i<N1;i++){
        ii.push_back(ii1[i]); jj.push_back(jj1[i]); vv.push_back(vv1[i]);
      }
      for(int i=0;i<N2;i++){
        if(hcat){
          ii.push_back(ii2[i]); jj.push_back(jj2[i]+n1); vv.push_back(vv2[i]);
        }
        else{
          ii.push_back(ii2[i]+m1); jj.push_back(jj2[i]); vv.push_back(vv2[i]);
        }
      }
    
        }


    void fill(OpKernelContext* context){
      int N = ii.size();
      TensorShape ii_shape({N});
      TensorShape jj_shape({N});
      TensorShape vv_shape({N});
              
      // create output tensor
      
      Tensor* ii_ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii_));
      Tensor* jj_ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj_));
      Tensor* vv_ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv_));
      
      // get the corresponding Eigen tensors for data access
      auto ii_tensor = ii_->flat<int64>().data();
      auto jj_tensor = jj_->flat<int64>().data();
      auto vv_tensor = vv_->flat<double>().data(); 
      for(int i=0;i<N;i++){
        ii_tensor[i] = ii[i];
        jj_tensor[i] = jj[i];
        vv_tensor[i] = vv[i];
      }
    }
};


void backward(double*grad_vv1,double*grad_vv2, const double*grad_vv, int N1, 
        int N2){
      
    int k = 0;
    for(int i=0;i<N1;i++){
      grad_vv1[i] = grad_vv[k++];
    }
    for(int i=0;i<N2;i++){
      grad_vv2[i] = grad_vv[k++];
    }
}