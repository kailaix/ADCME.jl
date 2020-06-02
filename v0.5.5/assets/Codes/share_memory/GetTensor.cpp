#include "Saver.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
#include<map>
#include<eigen3/Eigen/Core>
using std::string;

using namespace tensorflow;

REGISTER_OP("GetTensor")
.Input("handle : string")
  .Output("val : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle handle_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

class GetTensorOp : public OpKernel {
private:
  
public:
  explicit GetTensorOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    const Tensor& handle = context->input(0);    
    auto handle_tensor = handle.flat<string>().data();

    auto val_shape = TensorShape({10});   
    Tensor *val = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, val_shape, &val));

    if (!ds.vdata.count(string(*handle_tensor))){
        printf("[Get] Key %s does not exist.\n", string(*handle_tensor).c_str());
    }
    else{
      printf("[Get] Key %s exists.\n", string(*handle_tensor).c_str());
      auto v = ds.vdata[string(*handle_tensor)];
      for(int i=0;i<10;i++){
        val->flat<double>().data()[i] = v[i];
      }
    }
    printf("========Existing Keys========\n");
    for(auto & kv: ds.vdata){
      printf("Key %s\n", kv.first.c_str());
    }
    printf("\n");
    

  }
};
REGISTER_KERNEL_BUILDER(Name("GetTensor").Device(DEVICE_CPU), GetTensorOp);
