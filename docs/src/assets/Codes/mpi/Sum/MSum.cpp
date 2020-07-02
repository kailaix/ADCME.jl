#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "MSum.h"


REGISTER_OP("MSum")
.Input("a : double")
.Output("b : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &a_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });

REGISTER_OP("MSumGrad")
.Input("grad_b : double")
.Input("b : double")
.Input("a : double")
.Output("grad_a : double");

/*-------------------------------------------------------------------------------------*/

class MSumOp : public OpKernel {
private:
  
public:
  explicit MSumOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    
    
    const TensorShape& a_shape = a.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape b_shape({});
            
    // create output tensor
    
    Tensor* b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, b_shape, &b));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(b_tensor, a_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("MSum").Device(DEVICE_CPU), MSumOp);



class MSumGradOp : public OpKernel {
private:
  
public:
  explicit MSumGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_b = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& a = context->input(2);
    
    
    const TensorShape& grad_b_shape = grad_b.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& a_shape = a.shape();
    
    
    DCHECK_EQ(grad_b_shape.dims(), 0);
    DCHECK_EQ(b_shape.dims(), 0);
    DCHECK_EQ(a_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto grad_b_tensor = grad_b.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_a_tensor, grad_b_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MSumGrad").Device(DEVICE_CPU), MSumGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class MSumOpGPU : public OpKernel {
private:
  
public:
  explicit MSumOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    
    
    const TensorShape& a_shape = a.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape b_shape({});
            
    // create output tensor
    
    Tensor* b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, b_shape, &b));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("MSum").Device(DEVICE_GPU), MSumOpGPU);

class MSumGradOpGPU : public OpKernel {
private:
  
public:
  explicit MSumGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_b = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& a = context->input(2);
    
    
    const TensorShape& grad_b_shape = grad_b.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& a_shape = a.shape();
    
    
    DCHECK_EQ(grad_b_shape.dims(), 0);
    DCHECK_EQ(b_shape.dims(), 0);
    DCHECK_EQ(a_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto grad_b_tensor = grad_b.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MSumGrad").Device(DEVICE_GPU), MSumGradOpGPU);

#endif