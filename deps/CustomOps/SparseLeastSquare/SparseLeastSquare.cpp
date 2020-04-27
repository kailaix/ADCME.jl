#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
using namespace tensorflow;
// If you want to use the PyTorch feature, uncomment the following line
// #include "la.h" 
#include "SparseLeastSquare.h"

REGISTER_OP("SparseLeastSquare")

.Input("ii : int32")
  .Input("jj : int32")
  .Input("vv : double")
  .Input("ff : double")
  .Input("n : int32")
  .Output("u : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii_shape));
        shape_inference::ShapeHandle jj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &jj_shape));
        shape_inference::ShapeHandle vv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &vv_shape));
        shape_inference::ShapeHandle ff_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &ff_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &n_shape));

        c->set_output(0, c->Matrix(c->Dim(c->input(3), 0), -1 ));
    return Status::OK();
  });
class SparseLeastSquareOp : public OpKernel {
private:
  
public:
  explicit SparseLeastSquareOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(5, context->num_inputs());
    
    
    const Tensor& ii = context->input(0);
    const Tensor& jj = context->input(1);
    const Tensor& vv = context->input(2);
    const Tensor& ff = context->input(3);
    const Tensor& n = context->input(4);
    
    
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& ff_shape = ff.shape();
    const TensorShape& n_shape = n.shape();
    
    
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(ff_shape.dims(), 2);
    DCHECK_EQ(n_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n_ = *(n.flat<int>().data());
    int m = ff_shape.dim_size(1);
    int nv = ii_shape.dim_size(0);  
    int nbatch = ff_shape.dim_size(0);
    TensorShape u_shape({nbatch, n_});
            
    // create output tensor
    
    Tensor* u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, u_shape, &u));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int>().data();
    auto jj_tensor = jj.flat<int>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto ff_tensor = ff.flat<double>().data();
    auto n_tensor = n.flat<int>().data();
    auto u_tensor = u->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(u_tensor, ii_tensor, jj_tensor, vv_tensor, nv, ff_tensor, m, n_, nbatch);

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseLeastSquare").Device(DEVICE_CPU), SparseLeastSquareOp);


REGISTER_OP("SparseLeastSquareGrad")
  
  .Input("grad_u : double")
  .Input("u : double")
  .Input("ii : int32")
  .Input("jj : int32")
  .Input("vv : double")
  .Input("ff : double")
  .Input("n : int32")
  .Output("grad_ii : int32")
  .Output("grad_jj : int32")
  .Output("grad_vv : double")
  .Output("grad_ff : double")
  .Output("grad_n : int32");
class SparseLeastSquareGradOp : public OpKernel {
private:
  
public:
  explicit SparseLeastSquareGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_u = context->input(0);
    const Tensor& u = context->input(1);
    const Tensor& ii = context->input(2);
    const Tensor& jj = context->input(3);
    const Tensor& vv = context->input(4);
    const Tensor& ff = context->input(5);
    const Tensor& n = context->input(6);
    
    
    const TensorShape& grad_u_shape = grad_u.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& ff_shape = ff.shape();
    const TensorShape& n_shape = n.shape();
    
    
    DCHECK_EQ(grad_u_shape.dims(), 2);
    DCHECK_EQ(u_shape.dims(), 2);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(ff_shape.dims(), 2);
    DCHECK_EQ(n_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
    int n_ = *(n.flat<int>().data());
    int m = ff_shape.dim_size(1);
    int nbatch = ff_shape.dim_size(0);
    int nv = ii_shape.dim_size(0); 
    // create output shape
    
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_jj_shape(jj_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_ff_shape(ff_shape);
    TensorShape grad_n_shape(n_shape);
            
    // create output tensor
    
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii_shape, &grad_ii));
    Tensor* grad_jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_jj_shape, &grad_jj));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_vv_shape, &grad_vv));
    Tensor* grad_ff = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_ff_shape, &grad_ff));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_n_shape, &grad_n));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int>().data();
    auto jj_tensor = jj.flat<int>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto ff_tensor = ff.flat<double>().data();
    auto n_tensor = n.flat<int>().data();
    auto grad_u_tensor = grad_u.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto grad_ii_tensor = grad_ii->flat<int>().data();
    auto grad_jj_tensor = grad_jj->flat<int>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();
    auto grad_ff_tensor = grad_ff->flat<double>().data();
    auto grad_n_tensor = grad_n->flat<int>().data();   

    // implement your backward function here 

    // TODO:
    grad_vv->flat<double>().setZero();
    grad_ff->flat<double>().setZero();
    backward(grad_vv_tensor, grad_ff_tensor, grad_u_tensor, ii_tensor, jj_tensor, vv_tensor, \
        u_tensor, ff_tensor, nv, m, n_, nbatch);    
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseLeastSquareGrad").Device(DEVICE_CPU), SparseLeastSquareGradOp);

