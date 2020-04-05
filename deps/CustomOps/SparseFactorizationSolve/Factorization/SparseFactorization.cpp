#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>


#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif
using namespace tensorflow;
#include "SparseFactorization.h"


REGISTER_OP("SparseFactorization")

.Input("ii : int64")
.Input("jj : int64")
.Input("vv : double")
.Input("d : int64")
.Input("s : int64")
.Output("o : int64")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii_shape));
        shape_inference::ShapeHandle jj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &jj_shape));
        shape_inference::ShapeHandle vv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &vv_shape));
        shape_inference::ShapeHandle d_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &d_shape));
        shape_inference::ShapeHandle s_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &s_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });



class SparseFactorizationOp : public OpKernel {
private:
  
public:
  explicit SparseFactorizationOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(5, context->num_inputs());
    
    
    const Tensor& ii = context->input(0);
    const Tensor& jj = context->input(1);
    const Tensor& vv = context->input(2);
    const Tensor& d = context->input(3);
    const Tensor& s = context->input(4);
    
    
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& d_shape = d.shape();
    const TensorShape& s_shape = s.shape();
    
    
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 0);
    DCHECK_EQ(s_shape.dims(), 0);

    // extra check
        
    // create output shape
    int N = vv_shape.dim_size(0);
    TensorShape o_shape({});
            
    // create output tensor
    
    Tensor* o = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, o_shape, &o));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto d_tensor = d.flat<int64>().data();
    auto s_tensor = s.flat<int64>().data();
    auto o_tensor = o->flat<int64>().data();   

    // implement your forward function here 

    
    // TODO:
    forward(o_tensor, ii_tensor, jj_tensor, vv_tensor, N, *d_tensor, *s_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseFactorization").Device(DEVICE_CPU), SparseFactorizationOp);
