#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

void matvec_forward(double *y, const double *a, const double *x, int m){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    cudaStat = cudaMemset(y, 0, m*sizeof(double));

    double one = 1.0, zero = 0.0;

    cublasCreate(&handle);
    stat = cublasDgemv(handle, CUBLAS_OP_T, m, m, &one, a, m, x, 1, &zero, y, 1);

    if (stat != CUBLAS_STATUS_SUCCESS){
      printf("Matrix Vector Product failed\n");
    }

    cublasDestroy(handle);

}

REGISTER_OP("MatVec")
.Input("a : double")
.Input("x : double")
.Output("y : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a_shape));
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &x_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });


class MatVecOpGPU : public OpKernel {
private:
  
public:
  explicit MatVecOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& x = context->input(1);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& x_shape = x.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 2);
    DCHECK_EQ(x_shape.dims(), 1);

    // extra check
        
    // create output shape
    int m = x_shape.dim_size(0);
    TensorShape y_shape({m});
            
    // create output tensor
    
    Tensor* y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, y_shape, &y));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    matvec_forward(y_tensor, a_tensor, x_tensor, m);

  }
};
REGISTER_KERNEL_BUILDER(Name("MatVec").Device(DEVICE_GPU), MatVecOpGPU);
