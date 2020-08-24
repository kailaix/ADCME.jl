#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "TriSolve.h"


REGISTER_OP("TriSolve")
.Input("a : double")
.Input("b : double")
.Input("c : double")
.Input("d : double")
.Output("x : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_shape));
        shape_inference::ShapeHandle b_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &b_shape));
        shape_inference::ShapeHandle c_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &c_shape));
        shape_inference::ShapeHandle d_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &d_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("TriSolveGrad")
.Input("grad_x : double")
.Input("x : double")
.Input("a : double")
.Input("b : double")
.Input("c : double")
.Input("d : double")
.Output("grad_a : double")
.Output("grad_b : double")
.Output("grad_c : double")
.Output("grad_d : double");

/*-------------------------------------------------------------------------------------*/

class TriSolveOp : public OpKernel {
private:
  
public:
  explicit TriSolveOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& c = context->input(2);
    const Tensor& d = context->input(3);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 1);

    // extra check
        
    // create output shape
    int n = d_shape.dim_size(0);
    TensorShape x_shape({n});
    
            
    // create output tensor
    
    Tensor* x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_shape, &x));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto d_tensor = d.flat<double>().data();
    auto x_tensor = x->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    TriSolve_forward(x_tensor, a_tensor, b_tensor, c_tensor, d_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("TriSolve").Device(DEVICE_CPU), TriSolveOp);



class TriSolveGradOp : public OpKernel {
private:
  
public:
  explicit TriSolveGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_x = context->input(0);
    const Tensor& x = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& b = context->input(3);
    const Tensor& c = context->input(4);
    const Tensor& d = context->input(5);
    
    
    const TensorShape& grad_x_shape = grad_x.shape();
    const TensorShape& x_shape = x.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(grad_x_shape.dims(), 1);
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_b_shape(b_shape);
    TensorShape grad_c_shape(c_shape);
    TensorShape grad_d_shape(d_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_b_shape, &grad_b));
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_c_shape, &grad_c));
    Tensor* grad_d = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_d_shape, &grad_d));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto d_tensor = d.flat<double>().data();
    auto grad_x_tensor = grad_x.flat<double>().data();
    auto x_tensor = x.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_b_tensor = grad_b->flat<double>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();
    auto grad_d_tensor = grad_d->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int n = x_shape.dim_size(0);
    TriSolve_backward(
      grad_a_tensor, grad_b_tensor, grad_c_tensor, grad_d_tensor, grad_x_tensor, x_tensor,
      a_tensor, b_tensor, c_tensor, d_tensor, n);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("TriSolveGrad").Device(DEVICE_CPU), TriSolveGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class TriSolveOpGPU : public OpKernel {
private:
  
public:
  explicit TriSolveOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& c = context->input(2);
    const Tensor& d = context->input(3);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape x_shape({-1});
            
    // create output tensor
    
    Tensor* x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_shape, &x));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto d_tensor = d.flat<double>().data();
    auto x_tensor = x->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("TriSolve").Device(DEVICE_GPU), TriSolveOpGPU);

class TriSolveGradOpGPU : public OpKernel {
private:
  
public:
  explicit TriSolveGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_x = context->input(0);
    const Tensor& x = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& b = context->input(3);
    const Tensor& c = context->input(4);
    const Tensor& d = context->input(5);
    
    
    const TensorShape& grad_x_shape = grad_x.shape();
    const TensorShape& x_shape = x.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(grad_x_shape.dims(), 1);
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_b_shape(b_shape);
    TensorShape grad_c_shape(c_shape);
    TensorShape grad_d_shape(d_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_b_shape, &grad_b));
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_c_shape, &grad_c));
    Tensor* grad_d = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_d_shape, &grad_d));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto d_tensor = d.flat<double>().data();
    auto grad_x_tensor = grad_x.flat<double>().data();
    auto x_tensor = x.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_b_tensor = grad_b->flat<double>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();
    auto grad_d_tensor = grad_d->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("TriSolveGrad").Device(DEVICE_GPU), TriSolveGradOpGPU);

#endif