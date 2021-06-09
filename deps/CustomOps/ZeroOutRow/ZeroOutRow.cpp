#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ZeroOutRow.h"


REGISTER_OP("ZeroOutRow")
.Input("indices : int64")
.Input("vv : double")
.Input("bd : int64")
.Output("oindices : int64")
.Output("ovv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle indices_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices_shape));
        shape_inference::ShapeHandle vv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &vv_shape));
        shape_inference::ShapeHandle bd_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &bd_shape));

        c->set_output(0, c->Matrix(-1,2));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ZeroOutRowGrad")
.Input("grad_ovv : double")
.Input("oindices : int64")
.Input("ovv : double")
.Input("indices : int64")
.Input("vv : double")
.Input("bd : int64")
.Output("grad_indices : int64")
.Output("grad_vv : double")
.Output("grad_bd : int64");

/*-------------------------------------------------------------------------------------*/

class ZeroOutRowOp : public OpKernel {
private:
  
public:
  explicit ZeroOutRowOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& indices = context->input(0);
    const Tensor& vv = context->input(1);
    const Tensor& bd = context->input(2);
    
    
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& bd_shape = bd.shape();
    
    
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(bd_shape.dims(), 1);

    // extra check
        
    // create output shape
    int N = vv_shape.dim_size(0);
    int nbd = bd_shape.dim_size(0);
    ZeroOutRow zor(indices.flat<int64>().data(), vv.flat<double>().data(), N, 
            bd.flat<int64>().data(), nbd);
      
    int Nout = zor.forward();
    
    TensorShape oindices_shape({Nout,2});
    TensorShape ovv_shape({Nout});
            
    // create output tensor
    
    Tensor* oindices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, oindices_shape, &oindices));
    Tensor* ovv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, ovv_shape, &ovv));
    
    // get the corresponding Eigen tensors for data access
    
    auto indices_tensor = indices.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto bd_tensor = bd.flat<int64>().data();
    auto oindices_tensor = oindices->flat<int64>().data();
    auto ovv_tensor = ovv->flat<double>().data();   

    // implement your forward function here 
    zor.move_data(oindices_tensor, ovv_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("ZeroOutRow").Device(DEVICE_CPU), ZeroOutRowOp);



class ZeroOutRowGradOp : public OpKernel {
private:
  
public:
  explicit ZeroOutRowGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_ovv = context->input(0);
    const Tensor& oindices = context->input(1);
    const Tensor& ovv = context->input(2);
    const Tensor& indices = context->input(3);
    const Tensor& vv = context->input(4);
    const Tensor& bd = context->input(5);
    
    
    const TensorShape& grad_ovv_shape = grad_ovv.shape();
    const TensorShape& oindices_shape = oindices.shape();
    const TensorShape& ovv_shape = ovv.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& bd_shape = bd.shape();
    
    
    DCHECK_EQ(grad_ovv_shape.dims(), 1);
    DCHECK_EQ(oindices_shape.dims(), 2);
    DCHECK_EQ(ovv_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(bd_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    int N = vv_shape.dim_size(0);
    int nbd = bd_shape.dim_size(0);
    ZeroOutRow zor(indices.flat<int64>().data(), vv.flat<double>().data(), N, 
            bd.flat<int64>().data(), nbd);
    
    TensorShape grad_indices_shape(indices_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_bd_shape(bd_shape);
            
    // create output tensor
    
    Tensor* grad_indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_indices_shape, &grad_indices));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_vv_shape, &grad_vv));
    Tensor* grad_bd = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_bd_shape, &grad_bd));
    
    // get the corresponding Eigen tensors for data access
    
    auto indices_tensor = indices.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto bd_tensor = bd.flat<int64>().data();
    auto grad_ovv_tensor = grad_ovv.flat<double>().data();
    auto oindices_tensor = oindices.flat<int64>().data();
    auto ovv_tensor = ovv.flat<double>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    zor.backward(grad_vv_tensor, grad_ovv_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ZeroOutRowGrad").Device(DEVICE_CPU), ZeroOutRowGradOp);
