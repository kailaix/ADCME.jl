#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
#include "TestFun.h"

#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif

using namespace tensorflow;

REGISTER_OP("TestFun")

.Input("u : float")
  .Input("v : int32")
  .Output("g : float")
  .Output("w : double")
  .Output("s : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &u_shape));
        shape_inference::ShapeHandle v_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &v_shape));

        c->set_output(0, c->Vector(10));
        c->set_output(1, c->Matrix(10,10));
        c->set_output(2, c->Scalar());
    return Status::OK();
  });

REGISTER_OP("TestFunGrad")

.Input("grad_g : float")
.Input("grad_w : double")
.Input("grad_s : double")
  .Input("g : float")
  .Input("w : double")
  .Input("s : double")
  .Input("u : float")
  .Input("v : int32")
  .Output("grad_u : float")
  .Output("grad_v : int32");


class TestFunOp : public OpKernel {
private:
  
public:
  explicit TestFunOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& v = context->input(1);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 2);
    DCHECK_EQ(v_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape g_shape({10});
    TensorShape w_shape({10,10});
    TensorShape s_shape({});
            
    // create output tensor
    
    Tensor* g = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, g_shape, &g));
    Tensor* w = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, w_shape, &w));
    Tensor* s = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, s_shape, &s));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<float>().data();
    auto v_tensor = v.flat<int32>().data();
    auto g_tensor = g->flat<float>().data();
    auto w_tensor = w->flat<double>().data();
    auto s_tensor = s->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("TestFun").Device(DEVICE_CPU), TestFunOp);



class TestFunGradOp : public OpKernel {
private:
  
public:
  explicit TestFunGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_g = context->input(0);
    const Tensor& grad_w = context->input(1);
    const Tensor& grad_s = context->input(2);
    const Tensor& g = context->input(3);
    const Tensor& w = context->input(4);
    const Tensor& s = context->input(5);
    const Tensor& u = context->input(6);
    const Tensor& v = context->input(7);
    
    
    const TensorShape& grad_g_shape = grad_g.shape();
    const TensorShape& grad_w_shape = grad_w.shape();
    const TensorShape& grad_s_shape = grad_s.shape();
    const TensorShape& g_shape = g.shape();
    const TensorShape& w_shape = w.shape();
    const TensorShape& s_shape = s.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    
    
    DCHECK_EQ(grad_g_shape.dims(), 1);
    DCHECK_EQ(grad_w_shape.dims(), 2);
    DCHECK_EQ(grad_s_shape.dims(), 0);
    DCHECK_EQ(g_shape.dims(), 1);
    DCHECK_EQ(w_shape.dims(), 2);
    DCHECK_EQ(s_shape.dims(), 0);
    DCHECK_EQ(u_shape.dims(), 2);
    DCHECK_EQ(v_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_v_shape(v_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_v_shape, &grad_v));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<float>().data();
    auto v_tensor = v.flat<int32>().data();
    auto grad_g_tensor = grad_g.flat<float>().data();
    auto grad_w_tensor = grad_w.flat<double>().data();
    auto grad_s_tensor = grad_s.flat<double>().data();
    auto g_tensor = g.flat<float>().data();
    auto w_tensor = w.flat<double>().data();
    auto s_tensor = s.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<float>().data();
    auto grad_v_tensor = grad_v->flat<int32>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("TestFunGrad").Device(DEVICE_CPU), TestFunGradOp);

#ifdef USE_GPU
class TestFunOpGPU : public OpKernel {
private:
  
public:
  explicit TestFunOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& v = context->input(1);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 2);
    DCHECK_EQ(v_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape g_shape({10});
    TensorShape w_shape({10,10});
    TensorShape s_shape({});
            
    // create output tensor
    
    Tensor* g = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, g_shape, &g));
    Tensor* w = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, w_shape, &w));
    Tensor* s = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, s_shape, &s));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<float>().data();
    auto v_tensor = v.flat<int32>().data();
    auto g_tensor = g->flat<float>().data();
    auto w_tensor = w->flat<double>().data();
    auto s_tensor = s->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("TestFun").Device(DEVICE_GPU), TestFunOpGPU);



class TestFunGradOpGPU : public OpKernel {
private:
  
public:
  explicit TestFunGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_g = context->input(0);
    const Tensor& grad_w = context->input(1);
    const Tensor& grad_s = context->input(2);
    const Tensor& g = context->input(3);
    const Tensor& w = context->input(4);
    const Tensor& s = context->input(5);
    const Tensor& u = context->input(6);
    const Tensor& v = context->input(7);
    
    
    const TensorShape& grad_g_shape = grad_g.shape();
    const TensorShape& grad_w_shape = grad_w.shape();
    const TensorShape& grad_s_shape = grad_s.shape();
    const TensorShape& g_shape = g.shape();
    const TensorShape& w_shape = w.shape();
    const TensorShape& s_shape = s.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    
    
    DCHECK_EQ(grad_g_shape.dims(), 1);
    DCHECK_EQ(grad_w_shape.dims(), 2);
    DCHECK_EQ(grad_s_shape.dims(), 0);
    DCHECK_EQ(g_shape.dims(), 1);
    DCHECK_EQ(w_shape.dims(), 2);
    DCHECK_EQ(s_shape.dims(), 0);
    DCHECK_EQ(u_shape.dims(), 2);
    DCHECK_EQ(v_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_v_shape(v_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_v_shape, &grad_v));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<float>().data();
    auto v_tensor = v.flat<int32>().data();
    auto grad_g_tensor = grad_g.flat<float>().data();
    auto grad_w_tensor = grad_w.flat<double>().data();
    auto grad_s_tensor = grad_s.flat<double>().data();
    auto g_tensor = g.flat<float>().data();
    auto w_tensor = w.flat<double>().data();
    auto s_tensor = s.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<float>().data();
    auto grad_v_tensor = grad_v->flat<int32>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("TestFunGrad").Device(DEVICE_GPU), TestFunGradOpGPU);

#endif