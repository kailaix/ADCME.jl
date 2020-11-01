#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "InterpDimOne.h"


REGISTER_OP("InterpDimOne")
.Input("x : double")
.Input("y : double")
.Input("z : double")
.Output("w : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &x_shape));
        shape_inference::ShapeHandle y_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &y_shape));
        shape_inference::ShapeHandle z_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &z_shape));

        c->set_output(0, c->input(2));
    return Status::OK();
  });

REGISTER_OP("InterpDimOneGrad")
.Input("grad_w : double")
.Input("w : double")
.Input("x : double")
.Input("y : double")
.Input("z : double")
.Output("grad_x : double")
.Output("grad_y : double")
.Output("grad_z : double");

/*-------------------------------------------------------------------------------------*/

class InterpDimOneOp : public OpKernel {
private:
  
public:
  explicit InterpDimOneOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    const Tensor& y = context->input(1);
    const Tensor& z = context->input(2);
    
    
    const TensorShape& x_shape = x.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& z_shape = z.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(y_shape.dims(), 1);
    DCHECK_EQ(z_shape.dims(), 1);

    // extra check
        
    // create output shape
    int ni = z_shape.dim_size(0);
    int Ni = x_shape.dim_size(0);
    TensorShape w_shape({ni});
            
    // create output tensor
    
    Tensor* w = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, w_shape, &w));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto z_tensor = z.flat<double>().data();
    auto w_tensor = w->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    InterpDimOneForward(w_tensor, x_tensor, y_tensor, z_tensor, ni, Ni, context);

  }
};
REGISTER_KERNEL_BUILDER(Name("InterpDimOne").Device(DEVICE_CPU), InterpDimOneOp);



class InterpDimOneGradOp : public OpKernel {
private:
  
public:
  explicit InterpDimOneGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_w = context->input(0);
    const Tensor& w = context->input(1);
    const Tensor& x = context->input(2);
    const Tensor& y = context->input(3);
    const Tensor& z = context->input(4);
    
    
    const TensorShape& grad_w_shape = grad_w.shape();
    const TensorShape& w_shape = w.shape();
    const TensorShape& x_shape = x.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& z_shape = z.shape();
    
    
    DCHECK_EQ(grad_w_shape.dims(), 1);
    DCHECK_EQ(w_shape.dims(), 1);
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(y_shape.dims(), 1);
    DCHECK_EQ(z_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_x_shape(x_shape);
    TensorShape grad_y_shape(y_shape);
    TensorShape grad_z_shape(z_shape);
            
    // create output tensor
    
    Tensor* grad_x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_x_shape, &grad_x));
    Tensor* grad_y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_y_shape, &grad_y));
    Tensor* grad_z = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_z_shape, &grad_z));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto z_tensor = z.flat<double>().data();
    auto grad_w_tensor = grad_w.flat<double>().data();
    auto w_tensor = w.flat<double>().data();
    auto grad_x_tensor = grad_x->flat<double>().data();
    auto grad_y_tensor = grad_y->flat<double>().data();
    auto grad_z_tensor = grad_z->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int ni = z_shape.dim_size(0);
    int Ni = x_shape.dim_size(0);
    grad_y->flat<double>().setZero();
    InterpDimOneBackward(grad_y_tensor, grad_w_tensor, w_tensor, x_tensor, y_tensor, z_tensor, ni, Ni);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("InterpDimOneGrad").Device(DEVICE_CPU), InterpDimOneGradOp);
