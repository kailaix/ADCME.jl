#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 
void return_double(int n, double *b, const double*a);
using namespace tensorflow;


REGISTER_OP("GpuTest")

.Input("a : double")
.Output("b : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_shape));

        c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("GpuTestGrad")

.Input("grad_b : double")
.Input("b : double")
.Input("a : double")
.Output("grad_a : double");


class GpuTestOpGPU : public OpKernel {
private:
  
public:
  explicit GpuTestOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    
    
    const TensorShape& a_shape = a.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);

    // extra check
        
    // create output shape
    int n = a_shape.dim_size(0);
    TensorShape b_shape({n});
            
    // create output tensor
    
    Tensor* b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, b_shape, &b));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    return_double(n, b_tensor, a_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("GpuTest").Device(DEVICE_GPU), GpuTestOpGPU);
