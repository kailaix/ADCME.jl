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
#include "Solve.h"


REGISTER_OP("Solve")

.Input("rhs : double")
.Input("ii : int64")
.Input("jj : int64")
.Input("vv : double")
.Input("o : int64")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle rhs_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &rhs_shape));
        shape_inference::ShapeHandle ii_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &ii_shape));
        shape_inference::ShapeHandle jj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &jj_shape));
        shape_inference::ShapeHandle vv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &vv_shape));
        shape_inference::ShapeHandle o_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &o_shape));

        c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("SolveGrad")

.Input("grad_out : double")
.Input("out : double")
.Input("rhs : double")
.Input("ii : int64")
.Input("jj : int64")
.Input("vv : double")
.Input("o : int64")
.Output("grad_rhs : double")
.Output("grad_ii : int64")
.Output("grad_jj : int64")
.Output("grad_vv : double")
.Output("grad_o : int64");


class SolveOp : public OpKernel {
private:
  
public:
  explicit SolveOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(5, context->num_inputs());
    
    
    const Tensor& rhs = context->input(0);
    const Tensor& ii = context->input(1);
    const Tensor& jj = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& o = context->input(4);
    
    
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& o_shape = o.shape();
    
    
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(o_shape.dims(), 0);

    // extra check
        
    // create output shape
    int d = rhs_shape.dim_size(0);
    TensorShape out_shape({d});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto rhs_tensor = rhs.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto o_tensor = o.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(out_tensor, rhs_tensor, d, int(*o_tensor));

  }
};
REGISTER_KERNEL_BUILDER(Name("Solve").Device(DEVICE_CPU), SolveOp);



class SolveGradOp : public OpKernel {
private:
  
public:
  explicit SolveGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& rhs = context->input(2);
    const Tensor& ii = context->input(3);
    const Tensor& jj = context->input(4);
    const Tensor& vv = context->input(5);
    const Tensor& o = context->input(6);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& o_shape = o.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(o_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_rhs_shape(rhs_shape);
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_jj_shape(jj_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_o_shape(o_shape);
            
    // create output tensor
    
    Tensor* grad_rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_rhs_shape, &grad_rhs));
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_ii_shape, &grad_ii));
    Tensor* grad_jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_jj_shape, &grad_jj));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_vv_shape, &grad_vv));
    Tensor* grad_o = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_o_shape, &grad_o));
    
    // get the corresponding Eigen tensors for data access
    
    auto rhs_tensor = rhs.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto o_tensor = o.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_rhs_tensor = grad_rhs->flat<double>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int d = rhs_shape.dim_size(0), N = vv_shape.dim_size(0);
    backward(grad_rhs_tensor, grad_vv_tensor, grad_out_tensor, out_tensor, 
          ii_tensor, jj_tensor, N, d, *o_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SolveGrad").Device(DEVICE_CPU), SolveGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class SolveOpGPU : public OpKernel {
private:
  
public:
  explicit SolveOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(5, context->num_inputs());
    
    
    const Tensor& rhs = context->input(0);
    const Tensor& ii = context->input(1);
    const Tensor& jj = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& o = context->input(4);
    
    
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& o_shape = o.shape();
    
    
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(o_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto rhs_tensor = rhs.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto o_tensor = o.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("Solve").Device(DEVICE_GPU), SolveOpGPU);

class SolveGradOpGPU : public OpKernel {
private:
  
public:
  explicit SolveGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& rhs = context->input(2);
    const Tensor& ii = context->input(3);
    const Tensor& jj = context->input(4);
    const Tensor& vv = context->input(5);
    const Tensor& o = context->input(6);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& o_shape = o.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(o_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_rhs_shape(rhs_shape);
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_jj_shape(jj_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_o_shape(o_shape);
            
    // create output tensor
    
    Tensor* grad_rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_rhs_shape, &grad_rhs));
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_ii_shape, &grad_ii));
    Tensor* grad_jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_jj_shape, &grad_jj));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_vv_shape, &grad_vv));
    Tensor* grad_o = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_o_shape, &grad_o));
    
    // get the corresponding Eigen tensors for data access
    
    auto rhs_tensor = rhs.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto o_tensor = o.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_rhs_tensor = grad_rhs->flat<double>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SolveGrad").Device(DEVICE_GPU), SolveGradOpGPU);

#endif