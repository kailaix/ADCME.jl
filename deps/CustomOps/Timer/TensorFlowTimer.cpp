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
#include "TensorFlowTimer.h"

REGISTER_OP("SetTensorFlowTimer")
.Input("i : int32")
  .Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle i_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &i_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });
class SetTensorFlowTimerOp : public OpKernel {
private:
  
public:
  explicit SetTensorFlowTimerOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& i = context->input(0);
    
    
    const TensorShape& i_shape = i.shape();
    
    
    DCHECK_EQ(i_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto i_tensor = i.flat<int32>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    if(TimerMaps.find(*i_tensor)!=TimerMaps.end()){
      TimerMaps[*i_tensor]->Set();
    }
    else{
      TimerMaps[*i_tensor] = new Timer();
    }
    *out_tensor = 1.0;
  }
};
REGISTER_KERNEL_BUILDER(Name("SetTensorFlowTimer").Device(DEVICE_CPU), SetTensorFlowTimerOp);


REGISTER_OP("GetTensorFlowTimer")
.Input("i : int32")
  .Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle i_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &i_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });
class GetTensorFlowTimerOp : public OpKernel {
private:
  
public:
  explicit GetTensorFlowTimerOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& i = context->input(0);
    
    
    const TensorShape& i_shape = i.shape();
    
    
    DCHECK_EQ(i_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto i_tensor = i.flat<int32>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    if(TimerMaps.find(*i_tensor)==TimerMaps.end()){
      printf("ADCME: Timer: Referred timer %d not found.\n", *i_tensor);
      *out_tensor = -1.0;
    }
    else{
      *out_tensor = TimerMaps[*i_tensor]->Get();
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("GetTensorFlowTimer").Device(DEVICE_CPU), GetTensorFlowTimerOp);