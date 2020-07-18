#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
using namespace tensorflow;
#include "julia.h"
#include "DoItOp.h"

#include <thread>

REGISTER_OP("DoItOp")
  
  .Input("x : double")
  .Output("y : double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &x_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });
class DoItOpOp : public OpKernel {
private:
public:
  explicit DoItOpOp(OpKernelConstruction* context) : OpKernel(context) {
    jl_eval_string("println(\"Constructing DoItOp\");");
    std::cout<<"Construction: " << std::this_thread::get_id()<<std::endl;
  }

  bool IsExpensive() override { 
    std::cout<<"IsExpensive()"<<std::endl;
    return false; 
  };
  void Compute(OpKernelContext* context) override {   
    
    std::cout<<"Compute: " << std::this_thread::get_id()<<std::endl;

    // std::cout << "parallel: " << context->run_all_kernels_inline() << std::endl;
 
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    
    
    const TensorShape& x_shape = x.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 1);

    // extra check
        
    // create output shape
    int n = x_shape.dim_size(0);
    TensorShape y_shape({n});
            
    // create output tensor
    
    Tensor* y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, y_shape, &y));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y->flat<double>().data();   

    // implement your forward function here 
    thread::ThreadPool* const worker_threads =
        context->device()->tensorflow_cpu_worker_threads()->workers;

    
    // TODO:
    forward(y_tensor,x_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("DoItOp").Device(DEVICE_CPU), DoItOpOp);


