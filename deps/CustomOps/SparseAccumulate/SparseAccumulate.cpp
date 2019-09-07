#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include<mutex>
#include<cmath>
#include<string> 
#include<vector>
using std::string;
using namespace tensorflow;
// #define DEBUG
// If you want to use the PyTorch feature, uncomment the following line
// #include "la.h" 

class SparseAccum {
 public:
  explicit SparseAccum(): n(0) {
    jj.clear(); vv.clear();
    // printf("creating resources base\n");
  }
  ~SparseAccum()  {}

  
public:
  void push_back(const int32*cols, const double*vals, int32 row, int32 N){
    for(int i=0;i<N;i++){
      // printf("push %d: %d, %d, %f\n", i, row, cols[i], vals[i]);
      jj[row-1].push_back(cols[i]);
      vv[row-1].push_back(vals[i]);
    }
  }

  int32 get_n(){
    int32 num = 0;
    for (int i=0;i<n;i++){
      num += jj[i].size();
    }
    return num;
  }

  void copy(int32*rows, int32*cols, double*vals, int32 num){
    std::lock_guard<std::mutex> lck(mu);
    int32 j = 0;
    for(int i=0;i<n;i++){
      for(int k=0;k<jj[i].size();k++){
          rows[j] = i+1;
          cols[j] = jj[i][k];
          vals[j] = vv[i][k];
          j++;
      }
    }
  }

  void initialize(int32 nrows){
    std::lock_guard<std::mutex> lck(mu);
    jj.clear(); vv.clear(); n = nrows;
    jj.resize(n); vv.resize(n);
  }

 private:
  std::vector<std::vector<int32>> jj;
  std::vector<std::vector<double>> vv;
  int32 n;
  std::mutex mu;
  
};


SparseAccum r;

extern "C" void initialize_sparse_accumulate(int n){
  r.initialize((int32)n);
}

REGISTER_OP("SparseAccumulate")
.Input("rows : int32")
.Input("cols : int32")
.Input("values : double")
.Output("len : int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle rows_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &rows_shape));
        shape_inference::ShapeHandle cols_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &cols_shape));
        shape_inference::ShapeHandle values_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &values_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });
class SparseAccumulateOp : public OpKernel {
private:
  
public:
  explicit SparseAccumulateOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& rows = context->input(0);
    const Tensor& cols = context->input(1);
    const Tensor& values = context->input(2);
    
    
    const TensorShape& rows_shape = rows.shape();
    const TensorShape& cols_shape = cols.shape();
    const TensorShape& values_shape = values.shape();
    
    
    DCHECK_EQ(rows_shape.dims(), 0);
    DCHECK_EQ(cols_shape.dims(), 1);
    DCHECK_EQ(values_shape.dims(), 1);

    // extra check
        
    // create output shape
    int32 n = cols_shape.dim_size(0);
    TensorShape len_shape({});
            
    // create output tensor
    
    Tensor* len = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, len_shape, &len));
    
    // get the corresponding Eigen tensors for data access
    
    auto rows_tensor = rows.flat<int32>().data();
    auto cols_tensor = cols.flat<int32>().data();
    auto values_tensor = values.flat<double>().data();
    auto len_tensor = len->flat<int32>().data();   

    // implement your forward function here 

    // TODO:
    r.push_back(cols_tensor, values_tensor, *rows_tensor, n);
    *len_tensor = n;

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseAccumulate").Device(DEVICE_CPU), SparseAccumulateOp);




REGISTER_OP("GetSparseAccumulate")
  .Output("ii : int32")
  .Output("jj : int32")
  .Output("vv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });
class GetSparseAccumulateOp : public OpKernel {
private:
  
public:
  explicit GetSparseAccumulateOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(0, context->num_inputs());
    
    int n = r.get_n();


    // extra check
        
    // create output shape
    TensorShape ii_shape({n});
    TensorShape jj_shape({n});
    TensorShape vv_shape({n});
            
    // create output tensor
    
    Tensor* ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii));
    Tensor* jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    
    auto ii_tensor = ii->flat<int32>().data();
    auto jj_tensor = jj->flat<int32>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    r.copy(ii_tensor, jj_tensor, vv_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("GetSparseAccumulate").Device(DEVICE_CPU), GetSparseAccumulateOp);
