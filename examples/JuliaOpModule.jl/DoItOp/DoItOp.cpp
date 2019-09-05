#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
using namespace tensorflow;
#include "julia.h"
#include "Python.h"
// If you want to use the PyTorch feature, uncomment the following line
// #include "la.h" 
jl_value_t* array_type;
uv_async_t* uv_async_cond;

#include "DoItOp.h"

REGISTER_OP("DoItOp")
  
  .Input("x : double")
  .Output("y : double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &x_shape));

        c->set_output(0, c->Vector(-1));

        array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);


        printf("Created array_type\n");
        foo();
        printf("My id = %d\n\n", jl_threadid());

        PyGILState_STATE py_threadstate;
        py_threadstate = PyGILState_Ensure();
        jl_value_t* _cpp_async_cond_handle =
            jl_eval_string("const _cpp_async_cond = Base.AsyncCondition(); _cpp_async_cond.handle");
        auto s = "a = @task printjulia("+ std::to_string(reinterpret_cast<std::size_t>(&global_x))+"); schedule(a)";
        jl_eval_string(s.c_str());
        uv_async_cond = reinterpret_cast<uv_async_t*>(jl_unbox_voidpointer(_cpp_async_cond_handle));
        PyGILState_Release(py_threadstate);

    return Status::OK();
  });
class DoItOpOp : public OpKernel {
private:
  
public:
  explicit DoItOpOp(OpKernelConstruction* context) : OpKernel(context) {
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
            
    // create output tensor
    
    Tensor* y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, y_shape, &y));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(y_tensor,x_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("DoItOp").Device(DEVICE_CPU), DoItOpOp);


