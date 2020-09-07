#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<cmath>
#include <chrono>
#include <thread>
#include <ctime>
// Signatures for GPU kernels here 


using namespace tensorflow;


REGISTER_OP("SleepFor")
.Input("tin : double")
.Output("t : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle tin_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &tin_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });

class SleepForOp : public OpKernel {
private:
  
public:
  explicit SleepForOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& tin = context->input(0);
    
    
    const TensorShape& tin_shape = tin.shape();
    
    
    DCHECK_EQ(tin_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape t_shape({});
            
    // create output tensor
    
    Tensor* t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, t_shape, &t));
    
    // get the corresponding Eigen tensors for data access
    
    auto tin_tensor = tin.flat<double>().data();
    auto t_tensor = t->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    std::chrono::time_point<std::chrono::system_clock>  m_StartTime = std::chrono::system_clock::now();
    
    std::this_thread::sleep_for (std::chrono::duration<double>(*tin_tensor));
    std::chrono::time_point<std::chrono::system_clock>  endTime = std::chrono::system_clock::now();
    *t_tensor = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count()/1000;

  }
};
REGISTER_KERNEL_BUILDER(Name("SleepFor").Device(DEVICE_CPU), SleepForOp);




REGISTER_OP("Timer")
.Input("deps : double")
.Output("t : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle deps_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &deps_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });
/*-------------------------------------------------------------------------------------*/

class TimerOp : public OpKernel {
private:
  
public:
  explicit TimerOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& deps = context->input(0);
    
    
    const TensorShape& deps_shape = deps.shape();
    
    
    DCHECK_EQ(deps_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape t_shape({});
            
    // create output tensor
    
    Tensor* t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, t_shape, &t));
    
    // get the corresponding Eigen tensors for data access
    
    auto deps_tensor = deps.flat<double>().data();
    auto t_tensor = t->flat<double>().data();   

    // implement your forward function here 

    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = time_point_cast<milliseconds>(now).time_since_epoch().count();

    // TODO:
    *t_tensor = ms/1000.0;

  }
};
REGISTER_KERNEL_BUILDER(Name("Timer").Device(DEVICE_CPU), TimerOp);

