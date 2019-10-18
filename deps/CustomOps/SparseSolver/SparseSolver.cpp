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
#include "SparseSolver.h"

REGISTER_OP("SparseSolver")
  .Input("ii : int64")
  .Input("jj : int64")
  .Input("vv : double")
  .Input("kk : int64")
  .Input("ff : double")
  .Input("d : int64")
  .Output("u : double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii_shape));
        shape_inference::ShapeHandle jj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &jj_shape));
        shape_inference::ShapeHandle vv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &vv_shape));
        shape_inference::ShapeHandle kk_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &kk_shape));
        shape_inference::ShapeHandle ff_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &ff_shape));
        shape_inference::ShapeHandle d_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &d_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });
class SparseSolverOp : public OpKernel {
private:
  
public:
  explicit SparseSolverOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& ii = context->input(0);
    const Tensor& jj = context->input(1);
    const Tensor& vv = context->input(2);
    const Tensor& kk = context->input(3);
    const Tensor& ff = context->input(4);
    const Tensor& d = context->input(5);
    
    
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& kk_shape = kk.shape();
    const TensorShape& ff_shape = ff.shape();
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(kk_shape.dims(), 1);
    DCHECK_EQ(ff_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 0);

    // extra check
        
    // create output shape
    int nv = vv_shape.dim_size(0), nf = ff_shape.dim_size(0);
    TensorShape u_shape({*d.flat<int64>().data()});
            
    // create output tensor
    
    Tensor* u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, u_shape, &u));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto kk_tensor = kk.flat<int64>().data();
    auto ff_tensor = ff.flat<double>().data();
    auto d_tensor = d.flat<int64>().data();
    auto u_tensor = u->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(u_tensor, ii_tensor, jj_tensor, vv_tensor, nv, kk_tensor, ff_tensor, nf, *d_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseSolver").Device(DEVICE_CPU), SparseSolverOp);


REGISTER_OP("SparseSolverGrad")
  
  .Input("grad_u : double")
  .Input("u : double")
  .Input("ii : int64")
  .Input("jj : int64")
  .Input("vv : double")
  .Input("kk : int64")
  .Input("ff : double")
  .Input("d : int64")
  .Output("grad_ii : int64")
  .Output("grad_jj : int64")
  .Output("grad_vv : double")
  .Output("grad_kk : int64")
  .Output("grad_ff : double")
  .Output("grad_d : int64");
class SparseSolverGradOp : public OpKernel {
private:
  
public:
  explicit SparseSolverGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_u = context->input(0);
    const Tensor& u = context->input(1);
    const Tensor& ii = context->input(2);
    const Tensor& jj = context->input(3);
    const Tensor& vv = context->input(4);
    const Tensor& kk = context->input(5);
    const Tensor& ff = context->input(6);
    const Tensor& d = context->input(7);
    
    
    const TensorShape& grad_u_shape = grad_u.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& kk_shape = kk.shape();
    const TensorShape& ff_shape = ff.shape();
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(grad_u_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(kk_shape.dims(), 1);
    DCHECK_EQ(ff_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
    int nv = vv_shape.dim_size(0), nf = ff_shape.dim_size(0);

    // create output shape
    
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_jj_shape(jj_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_kk_shape(kk_shape);
    TensorShape grad_ff_shape(ff_shape);
    TensorShape grad_d_shape(d_shape);
            
    // create output tensor
    
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii_shape, &grad_ii));
    Tensor* grad_jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_jj_shape, &grad_jj));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_vv_shape, &grad_vv));
    Tensor* grad_kk = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_kk_shape, &grad_kk));
    Tensor* grad_ff = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_ff_shape, &grad_ff));
    Tensor* grad_d = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_d_shape, &grad_d));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto kk_tensor = kk.flat<int64>().data();
    auto ff_tensor = ff.flat<double>().data();
    auto d_tensor = d.flat<int64>().data();
    auto grad_u_tensor = grad_u.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto grad_ii_tensor = grad_ii->flat<int64>().data();
    auto grad_jj_tensor = grad_jj->flat<int64>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();
    auto grad_kk_tensor = grad_kk->flat<int64>().data();
    auto grad_ff_tensor = grad_ff->flat<double>().data();
    auto grad_d_tensor = grad_d->flat<int64>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_vv_tensor, grad_u_tensor, ii_tensor, jj_tensor, vv_tensor, u_tensor, nv, *d_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseSolverGrad").Device(DEVICE_CPU), SparseSolverGradOp);

