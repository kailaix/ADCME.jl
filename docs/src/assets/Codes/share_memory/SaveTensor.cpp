#include "Saver.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
#include<eigen3/Eigen/Core>
using std::string;
using namespace tensorflow;

REGISTER_OP("SaveTensor")

.Input("handle : string")
  .Input("val : double")
  .Output("out : string")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle handle_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle_shape));
        shape_inference::ShapeHandle val_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &val_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });

class SaveTensorOp : public OpKernel {
private:
  
public:
  explicit SaveTensorOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& handle = context->input(0);
    const Tensor& val = context->input(1);
    
    
    const TensorShape& val_shape = val.shape();
    
    
    DCHECK_EQ(val_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    auto handle_tensor = handle.flat<string>().data();
    auto val_tensor = val.flat<double>().data();
    auto out_tensor = out->flat<string>().data();   

    // implement your forward function here 
    // context->tensors_[string(*handle_tensor)] = val;
    ds.vdata[string(*handle_tensor)] = std::vector<double>(val_tensor, val_tensor+10);
    *out_tensor = *handle_tensor;    
    printf("[Add] %s to collections.\n", string(*handle_tensor).c_str());
    printf("========Existing Keys========\n");
    for(auto & kv: ds.vdata){
      printf("Key %s\n", kv.first.c_str());
    }
    printf("\n");
  }
};
REGISTER_KERNEL_BUILDER(Name("SaveTensor").Device(DEVICE_CPU), SaveTensorOp);
