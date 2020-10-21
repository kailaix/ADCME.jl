#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

#include <string>
using std::string;
#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif
using namespace tensorflow;
#include "ExtendedNn.h"


REGISTER_OP("ExtendedNn")

.Input("x : double")
.Input("config : int64")
.Input("theta : double")
.Input("activation : string")
.Output("out : double")
.Output("sensitivity : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &x_shape));
        shape_inference::ShapeHandle config_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &config_shape));
        shape_inference::ShapeHandle theta_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &theta_shape));
        shape_inference::ShapeHandle activation_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &activation_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ExtendedNnGrad")

.Input("grad_out : double")
.Input("grad_sensitivity : double")
.Input("out : double")
.Input("sensitivity : double")
.Input("x : double")
.Input("config : int64")
.Input("theta : double")
.Input("activation : string")
.Output("grad_x : double")
.Output("grad_config : int64")
.Output("grad_theta : double")
.Output("grad_activation : string");


class ExtendedNnOp : public OpKernel {
private:
  
public:
  explicit ExtendedNnOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    const Tensor& config = context->input(1);
    const Tensor& theta = context->input(2);
    const Tensor& activation = context->input(3);
    
    
    const TensorShape& x_shape = x.shape();
    const TensorShape& config_shape = config.shape();
    const TensorShape& theta_shape = theta.shape();
    auto config_tensor = config.flat<int64>().data();

    
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(config_shape.dims(), 1);
    DCHECK_EQ(theta_shape.dims(), 1);

    // extra check
        
    // create output shape
    int m = config_shape.dim_size(0);
    int n = x_shape.dim_size(0)/config_tensor[0];
    int q = config_tensor[m-1];

    
    TensorShape out_shape({n*q});
    TensorShape sensitivity_shape({n*config_tensor[0]*q});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    Tensor* sensitivity = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, sensitivity_shape, &sensitivity));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto theta_tensor = theta.flat<double>().data();
    string activation_tensor = string(*activation.flat<string>().data());
    auto out_tensor = out->flat<double>().data();
    auto sensitivity_tensor = sensitivity->flat<double>().data();   

    // implement your forward function here 

    // TODO:

    int N = get_num_theta(config_tensor, m);
    DCHECK_EQ(N, theta_shape.dim_size(0));

    if (activation_tensor.compare(string("tanh"))==0)
          forward_tanh(
            out_tensor, sensitivity_tensor, 
            x_tensor, n, 
            config_tensor, m, theta_tensor);

    else if (activation_tensor.compare(string("relu"))==0)
          forward_relu(
            out_tensor, sensitivity_tensor, 
            x_tensor, n, 
            config_tensor, m, theta_tensor);

    else 
      DCHECK_EQ(0, 1);

  }
};
REGISTER_KERNEL_BUILDER(Name("ExtendedNn").Device(DEVICE_CPU), ExtendedNnOp);



class ExtendedNnGradOp : public OpKernel {
private:
  
public:
  explicit ExtendedNnGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& grad_sensitivity = context->input(1);
    const Tensor& out = context->input(2);
    const Tensor& sensitivity = context->input(3);
    const Tensor& x = context->input(4);
    const Tensor& config = context->input(5);
    const Tensor& theta = context->input(6);
    const Tensor& activation = context->input(7);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& grad_sensitivity_shape = grad_sensitivity.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& sensitivity_shape = sensitivity.shape();
    const TensorShape& x_shape = x.shape();
    const TensorShape& config_shape = config.shape();
    const TensorShape& theta_shape = theta.shape();
    const TensorShape& activation_shape = activation.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(grad_sensitivity_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(sensitivity_shape.dims(), 1);
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(config_shape.dims(), 1);
    DCHECK_EQ(theta_shape.dims(), 1);
    DCHECK_EQ(activation_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
    auto config_tensor = config.flat<int64>().data();
    int n_theta = theta_shape.dim_size(0);
    int m = config_shape.dim_size(0);
    int n = x_shape.dim_size(0)/config_tensor[0];
        
    // create output shape
    
    TensorShape grad_x_shape(x_shape);
    TensorShape grad_config_shape(config_shape);
    TensorShape grad_theta_shape(theta_shape);
    TensorShape grad_activation_shape(activation_shape);
            
    // create output tensor
    
    Tensor* grad_x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_x_shape, &grad_x));
    Tensor* grad_config = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_config_shape, &grad_config));
    Tensor* grad_theta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_theta_shape, &grad_theta));
    Tensor* grad_activation = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_activation_shape, &grad_activation));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    
    auto theta_tensor = theta.flat<double>().data();
    string activation_tensor = string(*activation.flat<string>().data());
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto grad_sensitivity_tensor = grad_sensitivity.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto sensitivity_tensor = sensitivity.flat<double>().data();
    auto grad_x_tensor = grad_x->flat<double>().data();
    auto grad_config_tensor = grad_config->flat<int64>().data();
    auto grad_theta_tensor = grad_theta->flat<double>().data();

    // implement your backward function here 

    // TODO:

    if (activation_tensor.compare("tanh")==0)
        backward_tanh(
          grad_x_tensor, grad_theta_tensor, grad_out_tensor, out_tensor, sensitivity_tensor, 
          x_tensor, n, config_tensor, m, theta_tensor, n_theta);
    else if (activation_tensor.compare("relu")==0)
        backward_relu(
          grad_x_tensor, grad_theta_tensor, grad_out_tensor, out_tensor, sensitivity_tensor, 
          x_tensor, n, config_tensor, m, theta_tensor, n_theta);
    else 
      DCHECK_EQ(0,1);

    
  }
};
REGISTER_KERNEL_BUILDER(Name("ExtendedNnGrad").Device(DEVICE_CPU), ExtendedNnGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class ExtendedNnOpGPU : public OpKernel {
private:
  
public:
  explicit ExtendedNnOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    const Tensor& config = context->input(1);
    const Tensor& theta = context->input(2);
    const Tensor& activation = context->input(3);
    
    
    const TensorShape& x_shape = x.shape();
    const TensorShape& config_shape = config.shape();
    const TensorShape& theta_shape = theta.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(config_shape.dims(), 1);
    DCHECK_EQ(theta_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1});
    TensorShape sensitivity_shape({-1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    Tensor* sensitivity = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, sensitivity_shape, &sensitivity));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto config_tensor = config.flat<int64>().data();
    auto theta_tensor = theta.flat<double>().data();
    string activation_tensor = string(*activation.flat<string>().data());
    auto out_tensor = out->flat<double>().data();
    auto sensitivity_tensor = sensitivity->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("ExtendedNn").Device(DEVICE_GPU), ExtendedNnOpGPU);

class ExtendedNnGradOpGPU : public OpKernel {
private:
  
public:
  explicit ExtendedNnGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& grad_sensitivity = context->input(1);
    const Tensor& out = context->input(2);
    const Tensor& sensitivity = context->input(3);
    const Tensor& x = context->input(4);
    const Tensor& config = context->input(5);
    const Tensor& theta = context->input(6);
    const Tensor& activation = context->input(7);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& grad_sensitivity_shape = grad_sensitivity.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& sensitivity_shape = sensitivity.shape();
    const TensorShape& x_shape = x.shape();
    const TensorShape& config_shape = config.shape();
    const TensorShape& theta_shape = theta.shape();
    const TensorShape& activation_shape = activation.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(grad_sensitivity_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(sensitivity_shape.dims(), 1);
    DCHECK_EQ(x_shape.dims(), 1);
    DCHECK_EQ(config_shape.dims(), 1);
    DCHECK_EQ(theta_shape.dims(), 1);
    DCHECK_EQ(activation_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_x_shape(x_shape);
    TensorShape grad_config_shape(config_shape);
    TensorShape grad_theta_shape(theta_shape);
    TensorShape grad_activation_shape(activation_shape);
            
    // create output tensor
    
    Tensor* grad_x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_x_shape, &grad_x));
    Tensor* grad_config = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_config_shape, &grad_config));
    Tensor* grad_theta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_theta_shape, &grad_theta));
    Tensor* grad_activation = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_activation_shape, &grad_activation));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto config_tensor = config.flat<int64>().data();
    auto theta_tensor = theta.flat<double>().data();
    auto activation_tensor = activation.flat<string>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto grad_sensitivity_tensor = grad_sensitivity.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto sensitivity_tensor = sensitivity.flat<double>().data();
    auto grad_x_tensor = grad_x->flat<double>().data();
    auto grad_config_tensor = grad_config->flat<int64>().data();
    auto grad_theta_tensor = grad_theta->flat<double>().data();
    auto activation_tensor = string(*activation.flat<string>().data());   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ExtendedNnGrad").Device(DEVICE_GPU), ExtendedNnGradOpGPU);

#endif