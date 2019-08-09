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
  .Output("y : double")
  .Output("g : double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &x_shape));

        c->set_output(0, c->input(0));
        c->set_output(1, c->Matrix(c->Dim(c->input(0),0), c->Dim(c->input(0),0)));
    return Status::OK();
  });
class TwoLayerOp : public OpKernel {
private:
  
public:
  explicit TwoLayerOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    
    
    const TensorShape& x_shape = x.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 1);

    // extra check
        
    // create output shape
    int n = x_shape.dim_size(0);
    
    TensorShape y_shape({n});
    TensorShape g_shape({n,n});
            
    // create output tensor
    
    Tensor* y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, y_shape, &y));
    Tensor* g = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, g_shape, &g));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y->flat<double>().data();
    auto g_tensor = g->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(g_tensor, y_tensor, x_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("TwoLayer").Device(DEVICE_CPU), TwoLayerOp);

