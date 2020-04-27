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
#include "SolveBatchedRhs.h"


REGISTER_OP("SolveBatchedRhs")

.Input("a : double")
.Input("rhs : double")
.Output("v : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a_shape));
        shape_inference::ShapeHandle rhs_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &rhs_shape));

        c->set_output(0, c->Matrix(-1,-1));
    return Status::OK();
  });

REGISTER_OP("SolveBatchedRhsGrad")

.Input("grad_v : double")
.Input("v : double")
.Input("a : double")
.Input("rhs : double")
.Output("grad_a : double")
.Output("grad_rhs : double");


class SolveBatchedRhsOp : public OpKernel {
private:
  
public:
  explicit SolveBatchedRhsOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& rhs = context->input(1);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& rhs_shape = rhs.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 2);
    DCHECK_EQ(rhs_shape.dims(), 2);

    // extra check
        
    // create output shape
    int nbatch = rhs_shape.dim_size(0);
    int m = a_shape.dim_size(0), n = a_shape.dim_size(1);
    TensorShape v_shape({nbatch, n});
            
    // create output tensor
    
    Tensor* v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, v_shape, &v));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto v_tensor = v->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(v_tensor, a_tensor, rhs_tensor, m, n, nbatch);

  }
};
REGISTER_KERNEL_BUILDER(Name("SolveBatchedRhs").Device(DEVICE_CPU), SolveBatchedRhsOp);



class SolveBatchedRhsGradOp : public OpKernel {
private:
  
public:
  explicit SolveBatchedRhsGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_v = context->input(0);
    const Tensor& v = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& rhs = context->input(3);
    
    
    const TensorShape& grad_v_shape = grad_v.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& rhs_shape = rhs.shape();
    
    
    DCHECK_EQ(grad_v_shape.dims(), 2);
    DCHECK_EQ(v_shape.dims(), 2);
    DCHECK_EQ(a_shape.dims(), 2);
    DCHECK_EQ(rhs_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_rhs_shape(rhs_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_rhs_shape, &grad_rhs));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto grad_v_tensor = grad_v.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_rhs_tensor = grad_rhs->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int nbatch = rhs_shape.dim_size(0);
    int m = a_shape.dim_size(0), n = a_shape.dim_size(1);
    grad_a->flat<double>().setZero();
    backward(grad_a_tensor, grad_rhs_tensor,
      grad_v_tensor, v_tensor, a_tensor, rhs_tensor, m, n, nbatch);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SolveBatchedRhsGrad").Device(DEVICE_CPU), SolveBatchedRhsGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class SolveBatchedRhsOpGPU : public OpKernel {
private:
  
public:
  explicit SolveBatchedRhsOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& rhs = context->input(1);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& rhs_shape = rhs.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 2);
    DCHECK_EQ(rhs_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape v_shape({-1,-1});
            
    // create output tensor
    
    Tensor* v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, v_shape, &v));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto v_tensor = v->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("SolveBatchedRhs").Device(DEVICE_GPU), SolveBatchedRhsOpGPU);

class SolveBatchedRhsGradOpGPU : public OpKernel {
private:
  
public:
  explicit SolveBatchedRhsGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_v = context->input(0);
    const Tensor& v = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& rhs = context->input(3);
    
    
    const TensorShape& grad_v_shape = grad_v.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& rhs_shape = rhs.shape();
    
    
    DCHECK_EQ(grad_v_shape.dims(), 2);
    DCHECK_EQ(v_shape.dims(), 2);
    DCHECK_EQ(a_shape.dims(), 2);
    DCHECK_EQ(rhs_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_rhs_shape(rhs_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_rhs_shape, &grad_rhs));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto grad_v_tensor = grad_v.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_rhs_tensor = grad_rhs->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SolveBatchedRhsGrad").Device(DEVICE_GPU), SolveBatchedRhsGradOpGPU);

#endif