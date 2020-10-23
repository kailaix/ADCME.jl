#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/resource_mgr.h"

#include<cmath>
#include<string> 
using std::string;

using namespace tensorflow;

struct MyVar: public ResourceBase{
  string DebugString() const { return "MyVar"; };
  mutex mu;
  int32 val;
};

REGISTER_OP("TestResourceManager")

.Input("u : int32")
  .Output("v : int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &u_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });

class TestResourceManagerOp : public OpKernel {
private:
  
public:
  explicit TestResourceManagerOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    
    
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape v_shape({});
            
    // create output tensor
    
    Tensor* v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, v_shape, &v));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<int32>().data();
    auto v_tensor = v->flat<int32>().data();   

    // implement your forward function here 

    // TODO:
    auto rm = context->resource_manager();
    MyVar* my_var;
    Status s = rm->LookupOrCreate<MyVar>("my_container", "my_name", &my_var, [&](MyVar** ret){
      printf("Create a new container\n");
      *ret = new MyVar;
      (*ret)->val = *u_tensor;
      return Status::OK();
    });
    DCHECK_EQ(s, Status::OK());
    my_var->val += 1;
    my_var->Unref();
    
    
    *v_tensor = my_var->val;
    printf("Current Value=%d\n", *v_tensor);
    

  }
};
REGISTER_KERNEL_BUILDER(Name("TestResourceManager").Device(DEVICE_CPU), TestResourceManagerOp);

