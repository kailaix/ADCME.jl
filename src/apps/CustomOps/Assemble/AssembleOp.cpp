#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;

#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif

using namespace tensorflow;
#include "AssembleOp.h"

REGISTER_OP("AssembleOp")

.Input("index : int64")
  .Input("ks : double")
  .Input("sdof : int64")
  .Output("ii : int64")
  .Output("jj : int64")
  .Output("vv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle index_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &index_shape));
        shape_inference::ShapeHandle ks_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &ks_shape));
        shape_inference::ShapeHandle sdof_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &sdof_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("AssembleOpGrad")

.Input("grad_vv : double")
  .Input("ii : int64")
  .Input("jj : int64")
  .Input("vv : double")
  .Input("index : int64")
  .Input("ks : double")
  .Input("sdof : int64")
  .Output("grad_index : int64")
  .Output("grad_ks : double")
  .Output("grad_sdof : int64");


class AssembleOpOp : public OpKernel {
private:
  
public:
  explicit AssembleOpOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& index = context->input(0);
    const Tensor& ks = context->input(1);
    const Tensor& sdof = context->input(2);
    
    
    const TensorShape& index_shape = index.shape();
    const TensorShape& ks_shape = ks.shape();
    const TensorShape& sdof_shape = sdof.shape();
    
    
    DCHECK_EQ(index_shape.dims(), 2);
    DCHECK_EQ(ks_shape.dims(), 3);
    DCHECK_EQ(sdof_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n = index_shape.dim_size(0), edof = index_shape.dim_size(1);
    DCHECK_EQ(ks_shape.dim_size(0), n);
    DCHECK_EQ(ks_shape.dim_size(1), edof);
    DCHECK_EQ(ks_shape.dim_size(0), edof);

    TensorShape ii_shape({n*edof*edof});
    TensorShape jj_shape({n*edof*edof});
    TensorShape vv_shape({n*edof*edof});
            
    // create output tensor
    
    Tensor* ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii));
    Tensor* jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto index_tensor = index.flat<int64>().data();
    auto ks_tensor = ks.flat<double>().data();
    auto sdof_tensor = sdof.flat<int64>().data();
    auto ii_tensor = ii->flat<int64>().data();
    auto jj_tensor = jj->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(ii_tensor, jj_tensor, vv_tensor, index_tensor,  ks_tensor,
              n, edof);
  }
};
REGISTER_KERNEL_BUILDER(Name("AssembleOp").Device(DEVICE_CPU), AssembleOpOp);



class AssembleOpGradOp : public OpKernel {
private:
  
public:
  explicit AssembleOpGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& ii = context->input(1);
    const Tensor& jj = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& index = context->input(4);
    const Tensor& ks = context->input(5);
    const Tensor& sdof = context->input(6);

    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& index_shape = index.shape();
    const TensorShape& ks_shape = ks.shape();
    const TensorShape& sdof_shape = sdof.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(index_shape.dims(), 2);
    DCHECK_EQ(ks_shape.dims(), 3);
    DCHECK_EQ(sdof_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    int n = index_shape.dim_size(0), edof = index_shape.dim_size(1);

    TensorShape grad_index_shape(index_shape);
    TensorShape grad_ks_shape(ks_shape);
    TensorShape grad_sdof_shape(sdof_shape);
            
    // create output tensor
    
    Tensor* grad_index = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_index_shape, &grad_index));
    Tensor* grad_ks = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_ks_shape, &grad_ks));
    Tensor* grad_sdof = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_sdof_shape, &grad_sdof));
    
    // get the corresponding Eigen tensors for data access
    
    auto index_tensor = index.flat<int64>().data();
    auto ks_tensor = ks.flat<double>().data();
    auto sdof_tensor = sdof.flat<int64>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_index_tensor = grad_index->flat<int64>().data();
    auto grad_ks_tensor = grad_ks->flat<double>().data();
    auto grad_sdof_tensor = grad_sdof->flat<int64>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_ks_tensor, grad_vv_tensor, vv_tensor, index_tensor, ks_tensor,
              n, edof);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("AssembleOpGrad").Device(DEVICE_CPU), AssembleOpGradOp);

#ifdef USE_GPU
class AssembleOpOpGPU : public OpKernel {
private:
  
public:
  explicit AssembleOpOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& index = context->input(0);
    const Tensor& ks = context->input(1);
    const Tensor& sdof = context->input(2);
    
    
    const TensorShape& index_shape = index.shape();
    const TensorShape& ks_shape = ks.shape();
    const TensorShape& sdof_shape = sdof.shape();
    
    
    DCHECK_EQ(index_shape.dims(), 2);
    DCHECK_EQ(ks_shape.dims(), 3);
    DCHECK_EQ(sdof_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape ii_shape({-1});
    TensorShape jj_shape({-1});
    TensorShape vv_shape({-1});
            
    // create output tensor
    
    Tensor* ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii));
    Tensor* jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto index_tensor = index.flat<int64>().data();
    auto ks_tensor = ks.flat<double>().data();
    auto sdof_tensor = sdof.flat<int64>().data();
    auto ii_tensor = ii->flat<int64>().data();
    auto jj_tensor = jj->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("AssembleOp").Device(DEVICE_GPU), AssembleOpOpGPU);



class AssembleOpGradOpGPU : public OpKernel {
private:
  
public:
  explicit AssembleOpGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_ii = context->input(0);
    const Tensor& grad_jj = context->input(1);
    const Tensor& grad_vv = context->input(2);
    const Tensor& ii = context->input(3);
    const Tensor& jj = context->input(4);
    const Tensor& vv = context->input(5);
    const Tensor& index = context->input(6);
    const Tensor& ks = context->input(7);
    const Tensor& sdof = context->input(8);
    
    
    const TensorShape& grad_ii_shape = grad_ii.shape();
    const TensorShape& grad_jj_shape = grad_jj.shape();
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& index_shape = index.shape();
    const TensorShape& ks_shape = ks.shape();
    const TensorShape& sdof_shape = sdof.shape();
    
    
    DCHECK_EQ(grad_ii_shape.dims(), 1);
    DCHECK_EQ(grad_jj_shape.dims(), 1);
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(index_shape.dims(), 2);
    DCHECK_EQ(ks_shape.dims(), 3);
    DCHECK_EQ(sdof_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_index_shape(index_shape);
    TensorShape grad_ks_shape(ks_shape);
    TensorShape grad_sdof_shape(sdof_shape);
            
    // create output tensor
    
    Tensor* grad_index = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_index_shape, &grad_index));
    Tensor* grad_ks = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_ks_shape, &grad_ks));
    Tensor* grad_sdof = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_sdof_shape, &grad_sdof));
    
    // get the corresponding Eigen tensors for data access
    
    auto index_tensor = index.flat<int64>().data();
    auto ks_tensor = ks.flat<double>().data();
    auto sdof_tensor = sdof.flat<int64>().data();
    auto grad_ii_tensor = grad_ii.flat<int64>().data();
    auto grad_jj_tensor = grad_jj.flat<int64>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_index_tensor = grad_index->flat<int64>().data();
    auto grad_ks_tensor = grad_ks->flat<double>().data();
    auto grad_sdof_tensor = grad_sdof->flat<int64>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("AssembleOpGrad").Device(DEVICE_GPU), AssembleOpGradOpGPU);

#endif