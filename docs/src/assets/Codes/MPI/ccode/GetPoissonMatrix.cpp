#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "GetPoissonMatrix.h"


REGISTER_OP("GetPoissonMatrix")
.Input("kext : double")
.Input("colext : int64")
.Input("colssize : int64")
.Input("deps : double")
.Output("cols : int64")
.Output("vals : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle kext_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &kext_shape));
        shape_inference::ShapeHandle colext_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &colext_shape));
        shape_inference::ShapeHandle colssize_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &colssize_shape));
        shape_inference::ShapeHandle deps_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &deps_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("GetPoissonMatrixGrad")
.Input("grad_vals : double")
.Input("cols : int64")
.Input("vals : double")
.Input("kext : double")
.Input("colext : int64")
.Input("colssize : int64")
.Input("deps : double")
.Output("grad_kext : double")
.Output("grad_colext : int64")
.Output("grad_colssize : int64")
.Output("grad_deps : double");

/*-------------------------------------------------------------------------------------*/

class GetPoissonMatrixOp : public OpKernel {
private:
  
public:
  explicit GetPoissonMatrixOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& kext = context->input(0);
    const Tensor& colext = context->input(1);
    const Tensor& colssize = context->input(2);
    const Tensor& deps = context->input(3);
    
    
    const TensorShape& kext_shape = kext.shape();
    const TensorShape& colext_shape = colext.shape();
    const TensorShape& colssize_shape = colssize.shape();
    const TensorShape& deps_shape = deps.shape();
    
    
    DCHECK_EQ(kext_shape.dims(), 2);
    DCHECK_EQ(colext_shape.dims(), 2);
    DCHECK_EQ(colssize_shape.dims(), 0);
    DCHECK_EQ(deps_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n = kext_shape.dim_size(0)-2;
    auto colssize_tensor = colssize.flat<int64>().data();
    TensorShape cols_shape({*colssize_tensor});
    TensorShape vals_shape({*colssize_tensor});
            
    // create output tensor
    
    Tensor* cols = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, cols_shape, &cols));
    Tensor* vals = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, vals_shape, &vals));
    
    // get the corresponding Eigen tensors for data access
    
    auto kext_tensor = kext.flat<double>().data();
    auto colext_tensor = colext.flat<int64>().data();
    
    auto deps_tensor = deps.flat<double>().data();
    auto cols_tensor = cols->flat<int64>().data();
    auto vals_tensor = vals->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    PoissonMPI::GetPoissonMatrixForward(
      cols_tensor, vals_tensor, colext_tensor, kext_tensor, n
      );

  }
};
REGISTER_KERNEL_BUILDER(Name("GetPoissonMatrix").Device(DEVICE_CPU), GetPoissonMatrixOp);



class GetPoissonMatrixGradOp : public OpKernel {
private:
  
public:
  explicit GetPoissonMatrixGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vals = context->input(0);
    const Tensor& cols = context->input(1);
    const Tensor& vals = context->input(2);
    const Tensor& kext = context->input(3);
    const Tensor& colext = context->input(4);
    const Tensor& colssize = context->input(5);
    const Tensor& deps = context->input(6);
    
    
    const TensorShape& grad_vals_shape = grad_vals.shape();
    const TensorShape& cols_shape = cols.shape();
    const TensorShape& vals_shape = vals.shape();
    const TensorShape& kext_shape = kext.shape();
    const TensorShape& colext_shape = colext.shape();
    const TensorShape& colssize_shape = colssize.shape();
    const TensorShape& deps_shape = deps.shape();
    
    
    DCHECK_EQ(grad_vals_shape.dims(), 1);
    DCHECK_EQ(cols_shape.dims(), 1);
    DCHECK_EQ(vals_shape.dims(), 1);
    DCHECK_EQ(kext_shape.dims(), 2);
    DCHECK_EQ(colext_shape.dims(), 2);
    DCHECK_EQ(colssize_shape.dims(), 0);
    DCHECK_EQ(deps_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_kext_shape(kext_shape);
    TensorShape grad_colext_shape(colext_shape);
    TensorShape grad_colssize_shape(colssize_shape);
    TensorShape grad_deps_shape(deps_shape);
            
    // create output tensor
    
    Tensor* grad_kext = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_kext_shape, &grad_kext));
    Tensor* grad_colext = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_colext_shape, &grad_colext));
    Tensor* grad_colssize = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_colssize_shape, &grad_colssize));
    Tensor* grad_deps = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_deps_shape, &grad_deps));
    
    // get the corresponding Eigen tensors for data access
    
    auto kext_tensor = kext.flat<double>().data();
    auto colext_tensor = colext.flat<int64>().data();
    auto colssize_tensor = colssize.flat<int64>().data();
    auto deps_tensor = deps.flat<double>().data();
    auto grad_vals_tensor = grad_vals.flat<double>().data();
    auto cols_tensor = cols.flat<int64>().data();
    auto vals_tensor = vals.flat<double>().data();
    auto grad_kext_tensor = grad_kext->flat<double>().data();
    auto grad_deps_tensor = grad_deps->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("GetPoissonMatrixGrad").Device(DEVICE_CPU), GetPoissonMatrixGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA

REGISTER_OP("GetPoissonMatrixGpu")
.Input("kext : double")
.Input("colext : int64")
.Input("colssize : int64")
.Input("deps : double")
.Output("cols : int64")
.Output("vals : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle kext_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &kext_shape));
        shape_inference::ShapeHandle colext_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &colext_shape));
        shape_inference::ShapeHandle colssize_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &colssize_shape));
        shape_inference::ShapeHandle deps_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &deps_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("GetPoissonMatrixGpuGrad")
.Input("grad_vals : double")
.Input("cols : int64")
.Input("vals : double")
.Input("kext : double")
.Input("colext : int64")
.Input("colssize : int64")
.Input("deps : double")
.Output("grad_kext : double")
.Output("grad_colext : int64")
.Output("grad_colssize : int64")
.Output("grad_deps : double");

class GetPoissonMatrixOpGPU : public OpKernel {
private:
  
public:
  explicit GetPoissonMatrixOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& kext = context->input(0);
    const Tensor& colext = context->input(1);
    const Tensor& colssize = context->input(2);
    const Tensor& deps = context->input(3);
    
    
    const TensorShape& kext_shape = kext.shape();
    const TensorShape& colext_shape = colext.shape();
    const TensorShape& colssize_shape = colssize.shape();
    const TensorShape& deps_shape = deps.shape();
    
    
    DCHECK_EQ(kext_shape.dims(), 2);
    DCHECK_EQ(colext_shape.dims(), 2);
    DCHECK_EQ(colssize_shape.dims(), 0);
    DCHECK_EQ(deps_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape cols_shape({-1});
    TensorShape vals_shape({-1});
            
    // create output tensor
    
    Tensor* cols = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, cols_shape, &cols));
    Tensor* vals = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, vals_shape, &vals));
    
    // get the corresponding Eigen tensors for data access
    
    auto kext_tensor = kext.flat<double>().data();
    auto colext_tensor = colext.flat<int64>().data();
    auto colssize_tensor = colssize.flat<int64>().data();
    auto deps_tensor = deps.flat<double>().data();
    auto cols_tensor = cols->flat<int64>().data();
    auto vals_tensor = vals->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("GetPoissonMatrixGpu").Device(DEVICE_GPU), GetPoissonMatrixOpGPU);

class GetPoissonMatrixGradOpGPU : public OpKernel {
private:
  
public:
  explicit GetPoissonMatrixGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vals = context->input(0);
    const Tensor& cols = context->input(1);
    const Tensor& vals = context->input(2);
    const Tensor& kext = context->input(3);
    const Tensor& colext = context->input(4);
    const Tensor& colssize = context->input(5);
    const Tensor& deps = context->input(6);
    
    
    const TensorShape& grad_vals_shape = grad_vals.shape();
    const TensorShape& cols_shape = cols.shape();
    const TensorShape& vals_shape = vals.shape();
    const TensorShape& kext_shape = kext.shape();
    const TensorShape& colext_shape = colext.shape();
    const TensorShape& colssize_shape = colssize.shape();
    const TensorShape& deps_shape = deps.shape();
    
    
    DCHECK_EQ(grad_vals_shape.dims(), 1);
    DCHECK_EQ(cols_shape.dims(), 1);
    DCHECK_EQ(vals_shape.dims(), 1);
    DCHECK_EQ(kext_shape.dims(), 2);
    DCHECK_EQ(colext_shape.dims(), 2);
    DCHECK_EQ(colssize_shape.dims(), 0);
    DCHECK_EQ(deps_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_kext_shape(kext_shape);
    TensorShape grad_colext_shape(colext_shape);
    TensorShape grad_colssize_shape(colssize_shape);
    TensorShape grad_deps_shape(deps_shape);
            
    // create output tensor
    
    Tensor* grad_kext = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_kext_shape, &grad_kext));
    Tensor* grad_colext = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_colext_shape, &grad_colext));
    Tensor* grad_colssize = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_colssize_shape, &grad_colssize));
    Tensor* grad_deps = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_deps_shape, &grad_deps));
    
    // get the corresponding Eigen tensors for data access
    
    auto kext_tensor = kext.flat<double>().data();
    auto colext_tensor = colext.flat<int64>().data();
    auto colssize_tensor = colssize.flat<int64>().data();
    auto deps_tensor = deps.flat<double>().data();
    auto grad_vals_tensor = grad_vals.flat<double>().data();
    auto cols_tensor = cols.flat<int64>().data();
    auto vals_tensor = vals.flat<double>().data();
    auto grad_kext_tensor = grad_kext->flat<double>().data();
    auto grad_deps_tensor = grad_deps->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("GetPoissonMatrixGpuGrad").Device(DEVICE_GPU), GetPoissonMatrixGradOpGPU);

#endif