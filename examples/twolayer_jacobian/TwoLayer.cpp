#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
using namespace tensorflow;
// If you want to use the PyTorch feature, uncomment the following line
// #include "la.h" 
#include "TwoLayer.h"

REGISTER_OP("TwoLayer")
  
  .Input("x : double")
  .Input("w1 : double")
  .Input("b1 : double")
  .Input("w2 : double")
  .Input("b2 : double")
  .Output("y : double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &x_shape));
        shape_inference::ShapeHandle w1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &w1_shape));
        shape_inference::ShapeHandle b1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &b1_shape));
        shape_inference::ShapeHandle w2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &w2_shape));
        shape_inference::ShapeHandle b2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &b2_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });
class TwoLayerOp : public OpKernel {
private:
  
public:
  explicit TwoLayerOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(5, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    const Tensor& w1 = context->input(1);
    const Tensor& b1 = context->input(2);
    const Tensor& w2 = context->input(3);
    const Tensor& b2 = context->input(4);
    
    
    const TensorShape& x_shape = x.shape();
    const TensorShape& w1_shape = w1.shape();
    const TensorShape& b1_shape = b1.shape();
    const TensorShape& w2_shape = w2.shape();
    const TensorShape& b2_shape = b2.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(w1_shape.dims(), 1);
    DCHECK_EQ(b1_shape.dims(), 1);
    DCHECK_EQ(w2_shape.dims(), 1);
    DCHECK_EQ(b2_shape.dims(), 1);

    // extra check
        
    // create output shape
    int n = x_shape.dim_size(0);
    TensorShape y_shape({n*n});
            
    // create output tensor
    
    Tensor* y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, y_shape, &y));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto w1_tensor = w1.flat<double>().data();
    auto b1_tensor = b1.flat<double>().data();
    auto w2_tensor = w2.flat<double>().data();
    auto b2_tensor = b2.flat<double>().data();
    auto y_tensor = y->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(y_tensor, x_tensor, w1_tensor, w2_tensor, b1_tensor, b2_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("TwoLayer").Device(DEVICE_CPU), TwoLayerOp);
