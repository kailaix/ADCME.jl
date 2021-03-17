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

REGISTER_OP("MyAssign")

.Input("u : Ref(double)")
  .Input("v : double")
  .Output("w : Ref(double)")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle v_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &v_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("MyAssignGrad")

.Input("grad_w : double")
  .Input("w : double")
  .Input("u : double")
  .Input("v : double")
  .Output("grad_u : double")
  .Output("grad_v : double");


class MyAssignOp : public OpKernel {
private:
  
public:
  explicit MyAssignOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    Tensor u = context->mutable_input(0, true);
    const Tensor& v = context->input(1);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);

    context->forward_ref_input_to_ref_output(0,0);
    
    auto u_tensor = u.flat<double>().data();
    auto v_tensor = v.flat<double>().data();

    int n = u_shape.dim_size(0);
    for(int i=0;i<n;i++) u_tensor[i] = v_tensor[i];
  }
};
REGISTER_KERNEL_BUILDER(Name("MyAssign").Device(DEVICE_CPU), MyAssignOp);



class MyAssignGradOp : public OpKernel {
private:
  
public:
  explicit MyAssignGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_w = context->input(0);
    const Tensor& w = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& v = context->input(3);
    
    
    const TensorShape& grad_w_shape = grad_w.shape();
    const TensorShape& w_shape = w.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    
    
    DCHECK_EQ(grad_w_shape.dims(), 1);
    DCHECK_EQ(w_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_v_shape(v_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_v_shape, &grad_v));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto grad_w_tensor = grad_w.flat<double>().data();
    auto w_tensor = w.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_v_tensor = grad_v->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MyAssignGrad").Device(DEVICE_CPU), MyAssignGradOp);

#ifdef USE_GPU
class MyAssignOpGPU : public OpKernel {
private:
  
public:
  explicit MyAssignOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& v = context->input(1);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape w_shape({-1});
            
    // create output tensor
    
    Tensor* w = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, w_shape, &w));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto w_tensor = w->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("MyAssign").Device(DEVICE_GPU), MyAssignOpGPU);



class MyAssignGradOpGPU : public OpKernel {
private:
  
public:
  explicit MyAssignGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_w = context->input(0);
    const Tensor& w = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& v = context->input(3);
    
    
    const TensorShape& grad_w_shape = grad_w.shape();
    const TensorShape& w_shape = w.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    
    
    DCHECK_EQ(grad_w_shape.dims(), 1);
    DCHECK_EQ(w_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_v_shape(v_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_v_shape, &grad_v));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto grad_w_tensor = grad_w.flat<double>().data();
    auto w_tensor = w.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_v_tensor = grad_v->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MyAssignGrad").Device(DEVICE_GPU), MyAssignGradOpGPU);

#endif