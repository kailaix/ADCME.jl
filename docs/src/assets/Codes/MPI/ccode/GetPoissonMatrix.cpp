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



REGISTER_OP("GetPoissonGrad")
.Input("x : double")
.Input("uext : double")
.Input("cn : int64")
.Output("gradk : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x_shape));
        shape_inference::ShapeHandle uext_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &uext_shape));
        shape_inference::ShapeHandle cn_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &cn_shape));

        c->set_output(0, c->Matrix(-1,-1));
    return Status::OK();
  });

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



class GetPoissonGradOp : public OpKernel {
private:
  
public:
  explicit GetPoissonGradOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    const Tensor& uext = context->input(1);
    const Tensor& cn = context->input(2);
    
    
    const TensorShape& x_shape = x.shape();
    const TensorShape& uext_shape = uext.shape();
    const TensorShape& cn_shape = cn.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 2);
    DCHECK_EQ(uext_shape.dims(), 2);
    DCHECK_EQ(cn_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n = uext_shape.dim_size(0)-2;
    TensorShape gradk_shape({n+2, n+2});
            
    // create output tensor
    
    Tensor* gradk = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, gradk_shape, &gradk));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto uext_tensor = uext.flat<double>().data();
    auto cn_tensor = cn.flat<int64>().data();
    auto gradk_tensor = gradk->flat<double>().data();   

    // implement your forward function here 
    int N = *cn_tensor;
    // TODO:
    gradk->flat<double>().setZero();
    PoissonMPI::GetPoissonGrad(gradk_tensor, x_tensor, uext_tensor, N, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("GetPoissonGrad").Device(DEVICE_CPU), GetPoissonGradOp);

