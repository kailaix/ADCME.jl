#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "RollingFunctions.h"


REGISTER_OP("RollingFunctions")
.Input("u : double")
.Input("window : int64")
.Input("op : string")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle window_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &window_shape));
        shape_inference::ShapeHandle op_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &op_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("RollingFunctionsGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("u : double")
.Input("window : int64")
.Input("op : string")
.Output("grad_u : double")
.Output("grad_window : int64")
.Output("grad_op : string");

/*-------------------------------------------------------------------------------------*/

class RollingFunctionsOp : public OpKernel {
private:
  
public:
  explicit RollingFunctionsOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& window = context->input(1);
    const Tensor& op = context->input(2);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& window_shape = window.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(window_shape.dims(), 0);

    // extra check
        
    // create output shape
    auto window_tensor = window.flat<int64>().data();
    int window_size = *window_tensor;
    int n = u_shape.dim_size(0);
    TensorShape out_shape({n-window_size+1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    
    string op_tensor = string(op.flat<tstring>().data()->c_str());
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    out->flat<double>().setZero();
    RollingFunctionForward(out_tensor, u_tensor, window_size, n, op_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("RollingFunctions").Device(DEVICE_CPU), RollingFunctionsOp);



class RollingFunctionsGradOp : public OpKernel {
private:
  
public:
  explicit RollingFunctionsGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& window = context->input(3);
    const Tensor& op = context->input(4);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& window_shape = window.shape();
    const TensorShape& op_shape = op.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(window_shape.dims(), 0);
    DCHECK_EQ(op_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_window_shape(window_shape);
    TensorShape grad_op_shape(op_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_window = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_window_shape, &grad_window));
    Tensor* grad_op = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_op_shape, &grad_op));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto window_tensor = window.flat<int64>().data();
    auto op_tensor = string(*op.flat<string>().data());
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int window_size = *window_tensor;
    int n = u_shape.dim_size(0);
    grad_u->flat<double>().setZero();
    RollingFunctionBackward(grad_u_tensor, grad_out_tensor, out_tensor, u_tensor, window_size, n, op_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("RollingFunctionsGrad").Device(DEVICE_CPU), RollingFunctionsGradOp);

