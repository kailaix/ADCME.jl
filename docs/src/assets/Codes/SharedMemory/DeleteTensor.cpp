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

REGISTER_OP("DeleteTensor")
.Input("handle : string")
  .Output("val : bool")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle handle_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });

class DeleteTensorOp : public OpKernel {
private:
  
public:
  explicit DeleteTensorOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    const Tensor& handle = context->input(0);    
    auto handle_tensor = handle.flat<string>().data();

    auto val_shape = TensorShape({});   
    Tensor *val = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, val_shape, &val));

    if (ds.vdata.count(string(*handle_tensor))){
      ds.vdata.erase(string(*handle_tensor));
      printf("[Delete] Erase key %s.\n", string(*handle_tensor).c_str());
      *(val->flat<bool>().data()) = true;
    }
    else{
      printf("[Delete] Key %s does not exist.\n", string(*handle_tensor).c_str());
      *(val->flat<bool>().data()) = false;
    }
    printf("========Existing Keys========\n");
    for(auto & kv: ds.vdata){
      printf("Key %s\n", kv.first.c_str());
    }
    printf("\n");
  }
};
REGISTER_KERNEL_BUILDER(Name("DeleteTensor").Device(DEVICE_CPU), DeleteTensorOp);
