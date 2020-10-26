#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/platform/cpu_info.h"

#include<cmath>

// Signatures for GPU kernels here 

using namespace tensorflow;

#include "TestThreadPool.h"


REGISTER_OP("TestThreadPool")
.Input("a : double")
.Output("b : int64")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &a_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });

REGISTER_OP("TestThreadPoolGrad")
.Input("b : int64")
.Input("a : double")
.Output("grad_a : double");

/*-------------------------------------------------------------------------------------*/

class TestThreadPoolOp : public OpKernel {
private:
  
public:
  explicit TestThreadPoolOp(OpKernelConstruction* context) : OpKernel(context) {

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
    auto b_tensor = b->flat<int64>().data();   

    // implement your forward function here 

    // TODO:
    threadpool_print(context);

  }
};
REGISTER_KERNEL_BUILDER(Name("TestThreadPool").Device(DEVICE_CPU), TestThreadPoolOp);

