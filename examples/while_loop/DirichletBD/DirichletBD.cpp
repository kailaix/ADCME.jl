#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
using namespace tensorflow;
// If you want to use the PyTorch feature, uncomment the following line
// #include "la.h" 
#include "DirichletBD.h"

REGISTER_OP("DirichletBD")
  
  .Input("ii : int32")
  .Input("jj : int32")
  .Input("dof : int32")
  .Input("vv : double")
  .Output("uu : double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii_shape));
        shape_inference::ShapeHandle jj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &jj_shape));
        shape_inference::ShapeHandle dof_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &dof_shape));
        shape_inference::ShapeHandle vv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &vv_shape));

        // c->set_output(0, c->Vector(-1));
    return Status::OK();
  });
class DirichletBDOp : public OpKernel {
private:
  
public:
  explicit DirichletBDOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& ii = context->input(0);
    const Tensor& jj = context->input(1);
    const Tensor& dof = context->input(2);
    const Tensor& vv = context->input(3);
    
    
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& dof_shape = dof.shape();
    const TensorShape& vv_shape = vv.shape();
    
    
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(dof_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);

    // extra check
        
    // create output shape
    int d = dof_shape.dim_size(0), n = vv.dim_size(0);
    
    TensorShape uu_shape({n});
            
    // create output tensor
    
    Tensor* uu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, uu_shape, &uu));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int32>().data();
    auto jj_tensor = jj.flat<int32>().data();
    auto dof_tensor = dof.flat<int32>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto uu_tensor = uu->flat<double>().data();   

    // implement your forward function here 

    // TODO:    
    forward(uu_tensor, dof_tensor, d, ii_tensor, jj_tensor, vv_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("DirichletBD").Device(DEVICE_CPU), DirichletBDOp);


REGISTER_OP("DirichletBDGrad")
  
  .Input("grad_uu : double")
  .Input("uu : double")
  .Input("ii : int32")
  .Input("jj : int32")
  .Input("dof : int32")
  .Input("vv : double")
  .Output("grad_ii : int32")
  .Output("grad_jj : int32")
  .Output("grad_dof : int32")
  .Output("grad_vv : double");
class DirichletBDGradOp : public OpKernel {
private:
  
public:
  explicit DirichletBDGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_uu = context->input(0);
    const Tensor& uu = context->input(1);
    const Tensor& ii = context->input(2);
    const Tensor& jj = context->input(3);
    const Tensor& dof = context->input(4);
    const Tensor& vv = context->input(5);
    
    
    const TensorShape& grad_uu_shape = grad_uu.shape();
    const TensorShape& uu_shape = uu.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& dof_shape = dof.shape();
    const TensorShape& vv_shape = vv.shape();
    
    
    DCHECK_EQ(grad_uu_shape.dims(), 1);
    DCHECK_EQ(uu_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(dof_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        int d = dof_shape.dim_size(0), n = vv.dim_size(0);
    // create output shape
    
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_jj_shape(jj_shape);
    TensorShape grad_dof_shape(dof_shape);
    TensorShape grad_vv_shape(vv_shape);
            
    // create output tensor
    
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii_shape, &grad_ii));
    Tensor* grad_jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_jj_shape, &grad_jj));
    Tensor* grad_dof = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_dof_shape, &grad_dof));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_vv_shape, &grad_vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int32>().data();
    auto jj_tensor = jj.flat<int32>().data();
    auto dof_tensor = dof.flat<int32>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_uu_tensor = grad_uu.flat<double>().data();
    auto uu_tensor = uu.flat<double>().data();
    auto grad_ii_tensor = grad_ii->flat<int32>().data();
    auto grad_jj_tensor = grad_jj->flat<int32>().data();
    auto grad_dof_tensor = grad_dof->flat<int32>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_vv_tensor, grad_uu_tensor, dof_tensor, d, ii_tensor, jj_tensor, vv_tensor, n);
  }
};
REGISTER_KERNEL_BUILDER(Name("DirichletBDGrad").Device(DEVICE_CPU), DirichletBDGradOp);

