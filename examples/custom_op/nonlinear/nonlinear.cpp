#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
using namespace tensorflow;

#include "nonlinear.h"

REGISTER_OP("NonLinear")
  .Input("v: double")
  .Input("w: double")
  .Output("u: double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle v_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &v_shape));

    shape_inference::ShapeHandle w_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &w_shape));
    
    shape_inference::DimensionHandle u_shape = c->Dim(v_shape, 0);
  
    c->set_output(0, c->Vector(u_shape));
    return Status::OK();
  });
class NonLinearOp : public OpKernel {
public:
  explicit NonLinearOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    const Tensor& v = context->input(0);
    const Tensor& w = context->input(1);
    
    const TensorShape& v_shape = v.shape();
    const TensorShape& w_shape = w.shape();
    
    DCHECK_EQ(v_shape.dims(), 1);
    DCHECK_EQ(w_shape.dims(), 2);

    DCHECK_EQ(w_shape.dim_size(0), w_shape.dim_size(1));
    
    // create output shape
    TensorShape u_shape;
    u_shape.AddDim(v_shape.dim_size(0));
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, u_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    auto v_tensor = v.flat<double>().data();
    auto w_tensor = w.flat<double>().data();
    auto u_tensor = output->flat<double>().data();
    int n = v_shape.dim_size(0);
    
    forward(u_tensor, v_tensor, w_tensor, n);
  }
};
REGISTER_KERNEL_BUILDER(Name("NonLinear").Device(DEVICE_CPU), NonLinearOp);


REGISTER_OP("NonLinearGrad")
  .Input("du: double")
  .Input("v: double")
  .Input("w: double")
  .Output("dv: double")
  .Output("dw: double");
class NonLinearGradOp : public OpKernel {
public:
  explicit NonLinearGradOp(OpKernelConstruction* context) : OpKernel(context) {
  }
  
  void Compute(OpKernelContext* context) override {
    
    // some checks to be sure ...
    DCHECK_EQ(3, context->num_inputs());
    
    const Tensor& du = context->input(0);
    const Tensor& v = context->input(1);
    const Tensor& w = context->input(2);
    
    const TensorShape& du_shape = v.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& w_shape = w.shape();
    
    DCHECK_EQ(du_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);
    DCHECK_EQ(w_shape.dims(), 2);

    int N = v_shape.dim_size(0);
    TensorShape dv_shape;
    dv_shape.AddDim(N);
    TensorShape dw_shape;
    dw_shape.AddDim(N);
    dw_shape.AddDim(N);
            
    // create output tensor
    Tensor* dv = NULL, *dw = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, dv_shape, &dv));
    OP_REQUIRES_OK(context, context->allocate_output(1, dw_shape, &dw));
    
    auto du_tensor = du.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto w_tensor = w.flat<double>().data();
    auto dv_tensor = dv->flat<double>().data();
    auto dw_tensor = dw->flat<double>().data();
    
    double *pBuffer = new double[2*N];
    backward(du_tensor, v_tensor, w_tensor, dv_tensor, dw_tensor, N, pBuffer);
    delete [] pBuffer;
  }
};
REGISTER_KERNEL_BUILDER(Name("NonLinearGrad").Device(DEVICE_CPU), NonLinearGradOp);

