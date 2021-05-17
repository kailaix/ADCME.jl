#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ComputeSin.h"


REGISTER_OP("ComputeSin")
.Input("input : double")
.Output("output : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });


/*-------------------------------------------------------------------------------------*/

class ComputeSinOp : public OpKernel {
private:
  
public:
  explicit ComputeSinOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& input = context->input(0);
    
    
    const TensorShape& input_shape = input.shape();
    
    
    DCHECK_EQ(input_shape.dims(), 1);

    // extra check
        
    // create output shape
    int N = input_shape.dim_size(0);
    TensorShape output_shape({N});
            
    // create output tensor
    
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    
    auto input_tensor = input.flat<double>().data();
    auto output_tensor = output->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    ComputeSin(output_tensor, input_tensor, N);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeSin").Device(DEVICE_CPU), ComputeSinOp);



#ifdef GOOGLE_CUDA

void ComputeSinGPU(double *y, const double *x, int n);

class ComputeSinOpGPU : public OpKernel {
private:
  
public:
  explicit ComputeSinOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& input = context->input(0);
    
    
    const TensorShape& input_shape = input.shape();
    
    
    DCHECK_EQ(input_shape.dims(), 1);

    // extra check
        
    // create output shape
    int n = input_shape.dim_size(0);
    TensorShape output_shape({n});
            
    // create output tensor
    
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    
    auto input_tensor = input.flat<double>().data();
    auto output_tensor = output->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    ComputeSinGPU(output_tensor, input_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeSin").Device(DEVICE_GPU), ComputeSinOpGPU);

#endif