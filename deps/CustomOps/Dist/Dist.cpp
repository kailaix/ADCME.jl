#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
#include "Dist.h"

#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif

using namespace tensorflow;

REGISTER_OP("Dist")

.Input("x : double")
  .Input("y : double")
  .Input("order : int64")
  .Output("m : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x_shape));
        shape_inference::ShapeHandle y_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &y_shape));
        shape_inference::ShapeHandle order_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &order_shape));

        c->set_output(0, c->Matrix(c->Dim(c->input(0), 0),c->Dim(c->input(1), 0)));
    return Status::OK();
  });

REGISTER_OP("DistGrad")

.Input("grad_m : double")
  .Input("m : double")
  .Input("x : double")
  .Input("y : double")
  .Input("order : int64")
  .Output("grad_x : double")
  .Output("grad_y : double")
  .Output("grad_order : int64");


class DistOp : public OpKernel {
private:
  
public:
  explicit DistOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    const Tensor& y = context->input(1);
    const Tensor& order = context->input(2);
    
    
    const TensorShape& x_shape = x.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& order_shape = order.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 2);
    DCHECK_EQ(y_shape.dims(), 2);
    DCHECK_EQ(order_shape.dims(), 0);

    // extra check
        
    // create output shape
    int nx = x_shape.dim_size(0), ny = y_shape.dim_size(0);
    int d = x_shape.dim_size(1);
    DCHECK_EQ(y_shape.dim_size(1), d);
    TensorShape m_shape({nx, ny});
            
    // create output tensor
    
    Tensor* m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, m_shape, &m));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto order_tensor = order.flat<int64>().data();
    auto m_tensor = m->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(m_tensor, x_tensor, nx, y_tensor, ny, *order_tensor, d);
  }
};
REGISTER_KERNEL_BUILDER(Name("Dist").Device(DEVICE_CPU), DistOp);



class DistGradOp : public OpKernel {
private:
  
public:
  explicit DistGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_m = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& x = context->input(2);
    const Tensor& y = context->input(3);
    const Tensor& order = context->input(4);
    
    
    const TensorShape& grad_m_shape = grad_m.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& x_shape = x.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& order_shape = order.shape();
    
    
    DCHECK_EQ(grad_m_shape.dims(), 2);
    DCHECK_EQ(m_shape.dims(), 2);
    DCHECK_EQ(x_shape.dims(), 2);
    DCHECK_EQ(y_shape.dims(), 2);
    DCHECK_EQ(order_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    int nx = x_shape.dim_size(0), ny = y_shape.dim_size(0);
    int d = x_shape.dim_size(1);
    
    TensorShape grad_x_shape(x_shape);
    TensorShape grad_y_shape(y_shape);
    TensorShape grad_order_shape(order_shape);
            
    // create output tensor
    
    Tensor* grad_x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_x_shape, &grad_x));
    Tensor* grad_y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_y_shape, &grad_y));
    Tensor* grad_order = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_order_shape, &grad_order));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto order_tensor = order.flat<int64>().data();
    auto grad_m_tensor = grad_m.flat<double>().data();
    auto m_tensor = m.flat<double>().data();
    auto grad_x_tensor = grad_x->flat<double>().data();
    auto grad_y_tensor = grad_y->flat<double>().data();
    auto grad_order_tensor = grad_order->flat<int64>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_x_tensor, grad_y_tensor, grad_m_tensor, m_tensor, x_tensor, nx, y_tensor, ny, 
          *order_tensor, d);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("DistGrad").Device(DEVICE_CPU), DistGradOp);

#ifdef USE_GPU
class DistOpGPU : public OpKernel {
private:
  
public:
  explicit DistOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    const Tensor& y = context->input(1);
    const Tensor& order = context->input(2);
    
    
    const TensorShape& x_shape = x.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& order_shape = order.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 2);
    DCHECK_EQ(y_shape.dims(), 2);
    DCHECK_EQ(order_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape m_shape({-1,-1});
            
    // create output tensor
    
    Tensor* m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, m_shape, &m));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto order_tensor = order.flat<int64>().data();
    auto m_tensor = m->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("Dist").Device(DEVICE_GPU), DistOpGPU);



class DistGradOpGPU : public OpKernel {
private:
  
public:
  explicit DistGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_m = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& x = context->input(2);
    const Tensor& y = context->input(3);
    const Tensor& order = context->input(4);
    
    
    const TensorShape& grad_m_shape = grad_m.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& x_shape = x.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& order_shape = order.shape();
    
    
    DCHECK_EQ(grad_m_shape.dims(), 2);
    DCHECK_EQ(m_shape.dims(), 2);
    DCHECK_EQ(x_shape.dims(), 2);
    DCHECK_EQ(y_shape.dims(), 2);
    DCHECK_EQ(order_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_x_shape(x_shape);
    TensorShape grad_y_shape(y_shape);
    TensorShape grad_order_shape(order_shape);
            
    // create output tensor
    
    Tensor* grad_x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_x_shape, &grad_x));
    Tensor* grad_y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_y_shape, &grad_y));
    Tensor* grad_order = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_order_shape, &grad_order));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto order_tensor = order.flat<int64>().data();
    auto grad_m_tensor = grad_m.flat<double>().data();
    auto m_tensor = m.flat<double>().data();
    auto grad_x_tensor = grad_x->flat<double>().data();
    auto grad_y_tensor = grad_y->flat<double>().data();
    auto grad_order_tensor = grad_order->flat<int64>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("DistGrad").Device(DEVICE_GPU), DistGradOpGPU);

#endif