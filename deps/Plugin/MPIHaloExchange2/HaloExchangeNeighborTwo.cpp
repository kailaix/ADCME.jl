#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "HaloExchangeNeighborTwo.h"


REGISTER_OP("HaloExchangeNeighborTwo")
.Input("u : double")
.Input("fill_value : double")
.Input("m : int64")
.Input("n : int64")
.Input("tag : int64")
.Input("w : double")
.Output("uext : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &u_shape));
        shape_inference::ShapeHandle fill_value_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &fill_value_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &n_shape));
        shape_inference::ShapeHandle tag_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &tag_shape));
        shape_inference::ShapeHandle w_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &w_shape));

        c->set_output(0, c->Matrix(-1,-1));
    return Status::OK();
  });

REGISTER_OP("HaloExchangeNeighborTwoGrad")
.Input("grad_uext : double")
.Input("uext : double")
.Input("u : double")
.Input("fill_value : double")
.Input("m : int64")
.Input("n : int64")
.Input("tag : int64")
.Input("w : double")
.Output("grad_u : double")
.Output("grad_fill_value : double")
.Output("grad_m : int64")
.Output("grad_n : int64")
.Output("grad_tag : int64")
.Output("grad_w : double");

/*-------------------------------------------------------------------------------------*/

class HaloExchangeNeighborTwoOp : public OpKernel {
private:
  
public:
  explicit HaloExchangeNeighborTwoOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& fill_value = context->input(1);
    const Tensor& m = context->input(2);
    const Tensor& n = context->input(3);
    const Tensor& tag = context->input(4);
    const Tensor& w = context->input(5);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& fill_value_shape = fill_value.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& tag_shape = tag.shape();
    const TensorShape& w_shape = w.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 2);
    DCHECK_EQ(fill_value_shape.dims(), 0);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(tag_shape.dims(), 0);
    DCHECK_EQ(w_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n_ = u_shape.dim_size(0);
    TensorShape uext_shape({n_+4, n_+4});
            
    // create output tensor
    
    Tensor* uext = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, uext_shape, &uext));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto fill_value_tensor = fill_value.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto tag_tensor = tag.flat<int64>().data();
    auto w_tensor = w.flat<double>().data();
    auto uext_tensor = uext->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    uext->flat<double>().setZero();
    HaloExchangeNeighborTwo_forward(uext_tensor, u_tensor, *fill_value_tensor,
      *m_tensor, *n_tensor, n_,
      *tag_tensor);


  }
};
REGISTER_KERNEL_BUILDER(Name("HaloExchangeNeighborTwo").Device(DEVICE_CPU), HaloExchangeNeighborTwoOp);



class HaloExchangeNeighborTwoGradOp : public OpKernel {
private:
  
public:
  explicit HaloExchangeNeighborTwoGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_uext = context->input(0);
    const Tensor& uext = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& fill_value = context->input(3);
    const Tensor& m = context->input(4);
    const Tensor& n = context->input(5);
    const Tensor& tag = context->input(6);
    const Tensor& w = context->input(7);
    
    
    const TensorShape& grad_uext_shape = grad_uext.shape();
    const TensorShape& uext_shape = uext.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& fill_value_shape = fill_value.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& tag_shape = tag.shape();
    const TensorShape& w_shape = w.shape();
    
    
    DCHECK_EQ(grad_uext_shape.dims(), 2);
    DCHECK_EQ(uext_shape.dims(), 2);
    DCHECK_EQ(u_shape.dims(), 2);
    DCHECK_EQ(fill_value_shape.dims(), 0);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(tag_shape.dims(), 0);
    DCHECK_EQ(w_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_fill_value_shape(fill_value_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_tag_shape(tag_shape);
    TensorShape grad_w_shape(w_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_fill_value = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_fill_value_shape, &grad_fill_value));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_n_shape, &grad_n));
    Tensor* grad_tag = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_tag_shape, &grad_tag));
    Tensor* grad_w = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_w_shape, &grad_w));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto fill_value_tensor = fill_value.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto tag_tensor = tag.flat<int64>().data();
    auto w_tensor = w.flat<double>().data();
    auto grad_uext_tensor = grad_uext.flat<double>().data();
    auto uext_tensor = uext.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_fill_value_tensor = grad_fill_value->flat<double>().data();
    auto grad_w_tensor = grad_w->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int n_ = u_shape.dim_size(0);
    HaloExchangeNeighborTwo_backward(
      grad_u_tensor, grad_uext_tensor, uext_tensor, u_tensor, *fill_value_tensor,
      *m_tensor, *n_tensor, n_, *tag_tensor);
      
  }
};
REGISTER_KERNEL_BUILDER(Name("HaloExchangeNeighborTwoGrad").Device(DEVICE_CPU), HaloExchangeNeighborTwoGradOp);

