#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>


#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif
using namespace tensorflow;
#include "TriLu.h"


REGISTER_OP("TriLu")

.Input("u : double")
.Input("num : int64")
.Input("lu : int64")
.Output("v : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &u_shape));
        shape_inference::ShapeHandle num_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &num_shape));
        shape_inference::ShapeHandle lu_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &lu_shape));

        c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("TriLuGrad")

.Input("grad_v : double")
.Input("v : double")
.Input("u : double")
.Input("num : int64")
.Input("lu : int64")
.Output("grad_u : double")
.Output("grad_num : int64")
.Output("grad_lu : int64");


class TriLuOp : public OpKernel {
private:
  
public:
  explicit TriLuOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& num = context->input(1);
    const Tensor& lu = context->input(2);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& num_shape = num.shape();
    const TensorShape& lu_shape = lu.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 3);
    DCHECK_EQ(num_shape.dims(), 0);
    DCHECK_EQ(lu_shape.dims(), 0);

    // extra check
        
    // create output shape
    int m = u_shape.dim_size(1), n = u_shape.dim_size(2), batch_size=u_shape.dim_size(0);
    TensorShape v_shape({batch_size, m, n});
            
    // create output tensor
    
    Tensor* v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, v_shape, &v));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto num_tensor = num.flat<int64>().data();
    auto lu_tensor = lu.flat<int64>().data();
    auto v_tensor = v->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    v->flat<double>().setZero();
    if (*lu_tensor==0)
      forward_triu(v_tensor, u_tensor, m, n, *num_tensor, batch_size);
    else
      forward_tril(v_tensor, u_tensor, m, n, *num_tensor, batch_size);

  }
};
REGISTER_KERNEL_BUILDER(Name("TriLu").Device(DEVICE_CPU), TriLuOp);



class TriLuGradOp : public OpKernel {
private:
  
public:
  explicit TriLuGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_v = context->input(0);
    const Tensor& v = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& num = context->input(3);
    const Tensor& lu = context->input(4);
    
    
    const TensorShape& grad_v_shape = grad_v.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& num_shape = num.shape();
    const TensorShape& lu_shape = lu.shape();
    
    
    DCHECK_EQ(grad_v_shape.dims(), 3);
    DCHECK_EQ(v_shape.dims(), 3);
    DCHECK_EQ(u_shape.dims(), 3);
    DCHECK_EQ(num_shape.dims(), 0);
    DCHECK_EQ(lu_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_num_shape(num_shape);
    TensorShape grad_lu_shape(lu_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_num = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_num_shape, &grad_num));
    Tensor* grad_lu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_lu_shape, &grad_lu));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto num_tensor = num.flat<int64>().data();
    auto lu_tensor = lu.flat<int64>().data();
    auto grad_v_tensor = grad_v.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int m = u_shape.dim_size(1), n = u_shape.dim_size(2), batch_size=u_shape.dim_size(0);
    grad_u->flat<double>().setZero();
    if (*lu_tensor==0)
      backward_triu(
        grad_u_tensor, grad_v_tensor,
        v_tensor, u_tensor, m, n, *num_tensor, batch_size
      );
    else 
      backward_tril(
        grad_u_tensor, grad_v_tensor,
        v_tensor, u_tensor, m, n, *num_tensor, batch_size
      );

  }
};
REGISTER_KERNEL_BUILDER(Name("TriLuGrad").Device(DEVICE_CPU), TriLuGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class TriLuOpGPU : public OpKernel {
private:
  
public:
  explicit TriLuOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& num = context->input(1);
    const Tensor& lu = context->input(2);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& num_shape = num.shape();
    const TensorShape& lu_shape = lu.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 3);
    DCHECK_EQ(num_shape.dims(), 0);
    DCHECK_EQ(lu_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape v_shape({-1,-1,-1});
            
    // create output tensor
    
    Tensor* v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, v_shape, &v));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto num_tensor = num.flat<int64>().data();
    auto lu_tensor = lu.flat<int64>().data();
    auto v_tensor = v->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("TriLu").Device(DEVICE_GPU), TriLuOpGPU);

class TriLuGradOpGPU : public OpKernel {
private:
  
public:
  explicit TriLuGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_v = context->input(0);
    const Tensor& v = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& num = context->input(3);
    const Tensor& lu = context->input(4);
    
    
    const TensorShape& grad_v_shape = grad_v.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& num_shape = num.shape();
    const TensorShape& lu_shape = lu.shape();
    
    
    DCHECK_EQ(grad_v_shape.dims(), 3);
    DCHECK_EQ(v_shape.dims(), 3);
    DCHECK_EQ(u_shape.dims(), 3);
    DCHECK_EQ(num_shape.dims(), 0);
    DCHECK_EQ(lu_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_num_shape(num_shape);
    TensorShape grad_lu_shape(lu_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_num = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_num_shape, &grad_num));
    Tensor* grad_lu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_lu_shape, &grad_lu));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto num_tensor = num.flat<int64>().data();
    auto lu_tensor = lu.flat<int64>().data();
    auto grad_v_tensor = grad_v.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("TriLuGrad").Device(DEVICE_GPU), TriLuGradOpGPU);

#endif