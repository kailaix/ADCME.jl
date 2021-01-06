#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "PrintTensor.h"


REGISTER_OP("PrintTensor")
.Input("in : double")
.Input("info : string")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle in_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &in_shape));
        shape_inference::ShapeHandle info_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &info_shape));

        c->set_output(0, c->Matrix(-1,-1));
    return Status::OK();
  });

REGISTER_OP("PrintTensorGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("in : double")
.Input("info : string")
.Output("grad_in : double")
.Output("grad_info : string");

/*-------------------------------------------------------------------------------------*/

class PrintTensorOp : public OpKernel {
private:
  
public:
  explicit PrintTensorOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& in = context->input(0);
    const Tensor& info = context->input(1);
    
    
    const TensorShape& in_shape = in.shape();
    
    
    DCHECK_EQ(in_shape.dims(), 2);

    // extra check
        
    // create output shape
    int m = in_shape.dim_size(0), n = in_shape.dim_size(1);
    TensorShape out_shape({m, n});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto in_tensor = in.flat<double>().data();
    string info_tensor = string(info.flat<tstring>().data()->c_str());
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    print_matrix_forward(out_tensor, in_tensor, info_tensor, m, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("PrintTensor").Device(DEVICE_CPU), PrintTensorOp);



class PrintTensorGradOp : public OpKernel {
private:
  
public:
  explicit PrintTensorGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& in = context->input(2);
    const Tensor& info = context->input(3);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& in_shape = in.shape();
    const TensorShape& info_shape = info.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 2);
    DCHECK_EQ(out_shape.dims(), 2);
    DCHECK_EQ(in_shape.dims(), 2);
    DCHECK_EQ(info_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_in_shape(in_shape);
    TensorShape grad_info_shape(info_shape);
            
    // create output tensor
    
    Tensor* grad_in = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_in_shape, &grad_in));
    Tensor* grad_info = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_info_shape, &grad_info));
    
    // get the corresponding Eigen tensors for data access
    
    auto in_tensor = in.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_in_tensor = grad_in->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int m = in_shape.dim_size(0), n = in_shape.dim_size(1);
    print_matrix_backward(grad_in_tensor, grad_out_tensor, m, n);
    
    
  }
};
REGISTER_KERNEL_BUILDER(Name("PrintTensorGrad").Device(DEVICE_CPU), PrintTensorGradOp);
