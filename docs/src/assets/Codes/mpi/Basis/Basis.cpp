#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "Basis.h"


REGISTER_OP("Basis")
.Input("a : double")
.Output("c : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &a_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });

REGISTER_OP("BasisGrad")
.Input("grad_c : double")
.Input("c : double")
.Input("a : double")
.Output("grad_a : double");

/*-------------------------------------------------------------------------------------*/

class BasisOp : public OpKernel {
private:
  
public:
  explicit BasisOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    
    
    const TensorShape& a_shape = a.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape c_shape({});
            
    // create output tensor
    
    Tensor* c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, c_shape, &c));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto c_tensor = c->flat<double>().data();   

    // implement your forward function here 

    // TODO
    forward(c_tensor, a_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("Basis").Device(DEVICE_CPU), BasisOp);



class BasisGradOp : public OpKernel {
private:
  
public:
  explicit BasisGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_c = context->input(0);
    const Tensor& c = context->input(1);
    const Tensor& a = context->input(2);
    
    
    const TensorShape& grad_c_shape = grad_c.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& a_shape = a.shape();
    
    
    DCHECK_EQ(grad_c_shape.dims(), 0);
    DCHECK_EQ(c_shape.dims(), 0);
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
    auto grad_c_tensor = grad_c.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward(
      grad_a_tensor, grad_c_tensor, c_tensor, a_tensor
    );
    
  }
};
REGISTER_KERNEL_BUILDER(Name("BasisGrad").Device(DEVICE_CPU), BasisGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class BasisOpGPU : public OpKernel {
private:
  
public:
  explicit BasisOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    
    
    const TensorShape& a_shape = a.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape c_shape({});
            
    // create output tensor
    
    Tensor* c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, c_shape, &c));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto c_tensor = c->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("Basis").Device(DEVICE_GPU), BasisOpGPU);

class BasisGradOpGPU : public OpKernel {
private:
  
public:
  explicit BasisGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_c = context->input(0);
    const Tensor& c = context->input(1);
    const Tensor& a = context->input(2);
    
    
    const TensorShape& grad_c_shape = grad_c.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& a_shape = a.shape();
    
    
    DCHECK_EQ(grad_c_shape.dims(), 0);
    DCHECK_EQ(c_shape.dims(), 0);
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
    auto grad_c_tensor = grad_c.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("BasisGrad").Device(DEVICE_GPU), BasisGradOpGPU);

#endif