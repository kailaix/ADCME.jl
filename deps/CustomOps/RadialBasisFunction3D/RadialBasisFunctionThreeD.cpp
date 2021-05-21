#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "Common.h"


REGISTER_OP("RadialBasisFunctionThreeD")
.Input("x : double")
.Input("y : double")
.Input("z : double")
.Input("xc : double")
.Input("yc : double")
.Input("zc : double")
.Input("eps : double")
.Input("c : double")
.Input("d : double")
.Input("kind : int64")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &x_shape));
        shape_inference::ShapeHandle y_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &y_shape));
        shape_inference::ShapeHandle z_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &z_shape));
        shape_inference::ShapeHandle xc_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &xc_shape));
        shape_inference::ShapeHandle yc_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &yc_shape));
        shape_inference::ShapeHandle zc_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &zc_shape));
        shape_inference::ShapeHandle eps_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 1, &eps_shape));
        shape_inference::ShapeHandle c_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &c_shape));
        shape_inference::ShapeHandle d_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 1, &d_shape));
        shape_inference::ShapeHandle kind_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 0, &kind_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("RadialBasisFunctionThreeDGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("x : double")
.Input("y : double")
.Input("z : double")
.Input("xc : double")
.Input("yc : double")
.Input("zc : double")
.Input("eps : double")
.Input("c : double")
.Input("d : double")
.Input("kind : int64")
.Output("grad_x : double")
.Output("grad_y : double")
.Output("grad_z : double")
.Output("grad_xc : double")
.Output("grad_yc : double")
.Output("grad_zc : double")
.Output("grad_eps : double")
.Output("grad_c : double")
.Output("grad_d : double")
.Output("grad_kind : int64");

/*-------------------------------------------------------------------------------------*/

class RadialBasisFunctionThreeDOp : public OpKernel {
private:
  
public:
  explicit RadialBasisFunctionThreeDOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(10, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    const Tensor& y = context->input(1);
    const Tensor& z = context->input(2);
    const Tensor& xc = context->input(3);
    const Tensor& yc = context->input(4);
    const Tensor& zc = context->input(5);
    const Tensor& eps = context->input(6);
    const Tensor& c = context->input(7);
    const Tensor& d = context->input(8);
    const Tensor& kind = context->input(9);
    
    
    const TensorShape& x_shape = x.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& z_shape = z.shape();
    const TensorShape& xc_shape = xc.shape();
    const TensorShape& yc_shape = yc.shape();
    const TensorShape& zc_shape = zc.shape();
    const TensorShape& eps_shape = eps.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& d_shape = d.shape();
    const TensorShape& kind_shape = kind.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(y_shape.dims(), 1);
    DCHECK_EQ(z_shape.dims(), 1);
    DCHECK_EQ(xc_shape.dims(), 1);
    DCHECK_EQ(yc_shape.dims(), 1);
    DCHECK_EQ(zc_shape.dims(), 1);
    DCHECK_EQ(eps_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 1);
    DCHECK_EQ(kind_shape.dims(), 0);

    // extra check
        
    // create output shape
    int nc = c.dim_size(0);
    int nxyz = x.dim_size(0);
    int nd = d.dim_size(0);

    TensorShape out_shape({nxyz});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto z_tensor = z.flat<double>().data();
    auto xc_tensor = xc.flat<double>().data();
    auto yc_tensor = yc.flat<double>().data();
    auto zc_tensor = zc.flat<double>().data();
    auto eps_tensor = eps.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto d_tensor = d.flat<double>().data();
    auto kind_tensor = kind.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    RBF3D::forward(
      out_tensor, x_tensor, y_tensor, z_tensor, 
      eps_tensor, xc_tensor, yc_tensor, zc_tensor, 
      c_tensor, nc, d_tensor, nd, nxyz, *kind_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("RadialBasisFunctionThreeD").Device(DEVICE_CPU), RadialBasisFunctionThreeDOp);



class RadialBasisFunctionThreeDGradOp : public OpKernel {
private:
  
public:
  explicit RadialBasisFunctionThreeDGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& x = context->input(2);
    const Tensor& y = context->input(3);
    const Tensor& z = context->input(4);
    const Tensor& xc = context->input(5);
    const Tensor& yc = context->input(6);
    const Tensor& zc = context->input(7);
    const Tensor& eps = context->input(8);
    const Tensor& c = context->input(9);
    const Tensor& d = context->input(10);
    const Tensor& kind = context->input(11);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& x_shape = x.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& z_shape = z.shape();
    const TensorShape& xc_shape = xc.shape();
    const TensorShape& yc_shape = yc.shape();
    const TensorShape& zc_shape = zc.shape();
    const TensorShape& eps_shape = eps.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& d_shape = d.shape();
    const TensorShape& kind_shape = kind.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(y_shape.dims(), 1);
    DCHECK_EQ(z_shape.dims(), 1);
    DCHECK_EQ(xc_shape.dims(), 1);
    DCHECK_EQ(yc_shape.dims(), 1);
    DCHECK_EQ(zc_shape.dims(), 1);
    DCHECK_EQ(eps_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 1);
    DCHECK_EQ(kind_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_x_shape(x_shape);
    TensorShape grad_y_shape(y_shape);
    TensorShape grad_z_shape(z_shape);
    TensorShape grad_xc_shape(xc_shape);
    TensorShape grad_yc_shape(yc_shape);
    TensorShape grad_zc_shape(zc_shape);
    TensorShape grad_eps_shape(eps_shape);
    TensorShape grad_c_shape(c_shape);
    TensorShape grad_d_shape(d_shape);
    TensorShape grad_kind_shape(kind_shape);
            
    // create output tensor
    
    Tensor* grad_x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_x_shape, &grad_x));
    Tensor* grad_y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_y_shape, &grad_y));
    Tensor* grad_z = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_z_shape, &grad_z));
    Tensor* grad_xc = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_xc_shape, &grad_xc));
    Tensor* grad_yc = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_yc_shape, &grad_yc));
    Tensor* grad_zc = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_zc_shape, &grad_zc));
    Tensor* grad_eps = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_eps_shape, &grad_eps));
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_c_shape, &grad_c));
    Tensor* grad_d = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_d_shape, &grad_d));
    Tensor* grad_kind = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_kind_shape, &grad_kind));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto z_tensor = z.flat<double>().data();
    auto xc_tensor = xc.flat<double>().data();
    auto yc_tensor = yc.flat<double>().data();
    auto zc_tensor = zc.flat<double>().data();
    auto eps_tensor = eps.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto d_tensor = d.flat<double>().data();
    auto kind_tensor = kind.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_x_tensor = grad_x->flat<double>().data();
    auto grad_y_tensor = grad_y->flat<double>().data();
    auto grad_z_tensor = grad_z->flat<double>().data();
    auto grad_xc_tensor = grad_xc->flat<double>().data();
    auto grad_yc_tensor = grad_yc->flat<double>().data();
    auto grad_zc_tensor = grad_zc->flat<double>().data();
    auto grad_eps_tensor = grad_eps->flat<double>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();
    auto grad_d_tensor = grad_d->flat<double>().data();   

    // implement your backward function here 

    // TODO:

    int nc = c.dim_size(0);
    int nxyz = x.dim_size(0);
    int nd = d.dim_size(0);

    grad_xc->flat<double>().setZero();
    grad_yc->flat<double>().setZero();
    grad_zc->flat<double>().setZero();
    grad_eps->flat<double>().setZero();
    grad_c->flat<double>().setZero();
    grad_d->flat<double>().setZero();

    RBF3D::backward(
        grad_xc_tensor, grad_yc_tensor, grad_zc_tensor,
        grad_c_tensor, grad_d_tensor, grad_eps_tensor,
        grad_out_tensor, 
        out_tensor, x_tensor, y_tensor, z_tensor, 
        eps_tensor, xc_tensor, yc_tensor, zc_tensor, 
        c_tensor, nc, d_tensor, nd, nxyz, *kind_tensor);
      
  }
};
REGISTER_KERNEL_BUILDER(Name("RadialBasisFunctionThreeDGrad").Device(DEVICE_CPU), RadialBasisFunctionThreeDGradOp);
