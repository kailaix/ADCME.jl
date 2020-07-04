#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

#include<mutex>

// Signatures for GPU kernels here 
std::mutex mu;

using namespace tensorflow;

REGISTER_OP("PoissonOp")
.Input("u : double")
.Input("f : double")
.Input("h : double")
.Output("ut : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &u_shape));
        shape_inference::ShapeHandle f_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &f_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &h_shape));

        c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("PoissonOpGrad")
.Input("grad_ut : double")
.Input("ut : double")
.Input("u : double")
.Input("f : double")
.Input("h : double")
.Output("grad_u : double")
.Output("grad_f : double")
.Output("grad_h : double");

/*-------------------------------------------------------------------------------------*/

class PoissonOpOp : public OpKernel {
private:
  double *out_up, *out_down, *out_left, *out_right;
  double *in_up, *in_down, *in_left, *in_right;
  bool initialized;
public:
  explicit PoissonOpOp(OpKernelConstruction* context) : OpKernel(context) {
    initialized = false;
  }

  void forward(double *ut, const double *u, const double *f, double h, int m, int n);

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& f = context->input(1);
    const Tensor& h = context->input(2);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& f_shape = f.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 2);
    DCHECK_EQ(f_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    int m = u_shape.dim_size(0), n = u_shape.dim_size(1);

    if(!initialized){
      std::lock_guard<std::mutex> guard(mu);
      in_left = new double[m];
      in_right = new double[m];
      in_up = new double[n];
      in_down = new double[n];
      out_left = new double[m];
      out_right = new double[m];
      for(int i = 0; i<m;i++){
        in_left[i] = in_right[i] = out_left[i] = out_right[i] = 0.0;
      }
      for(int j=0;j<n;j++) in_up[j] = in_down[j] = 0.0;
      initialized = true;
    }
        
    // create output shape
    
    TensorShape ut_shape({m, n});
            
    // create output tensor
    
    Tensor* ut = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ut_shape, &ut));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto f_tensor = f.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto ut_tensor = ut->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(ut_tensor, u_tensor, f_tensor, *h_tensor, m, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("PoissonOp").Device(DEVICE_CPU), PoissonOpOp);



class PoissonOpGradOp : public OpKernel {
private:
  
public:
  explicit PoissonOpGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_ut = context->input(0);
    const Tensor& ut = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& f = context->input(3);
    const Tensor& h = context->input(4);
    
    
    const TensorShape& grad_ut_shape = grad_ut.shape();
    const TensorShape& ut_shape = ut.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& f_shape = f.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_ut_shape.dims(), 2);
    DCHECK_EQ(ut_shape.dims(), 2);
    DCHECK_EQ(u_shape.dims(), 2);
    DCHECK_EQ(f_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_f_shape(f_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_f = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_f_shape, &grad_f));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto f_tensor = f.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_ut_tensor = grad_ut.flat<double>().data();
    auto ut_tensor = ut.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_f_tensor = grad_f->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("PoissonOpGrad").Device(DEVICE_CPU), PoissonOpGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class PoissonOpOpGPU : public OpKernel {
private:
  
public:
  explicit PoissonOpOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& f = context->input(1);
    const Tensor& h = context->input(2);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& f_shape = f.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 2);
    DCHECK_EQ(f_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape ut_shape({-1,-1});
            
    // create output tensor
    
    Tensor* ut = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ut_shape, &ut));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto f_tensor = f.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto ut_tensor = ut->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("PoissonOp").Device(DEVICE_GPU), PoissonOpOpGPU);

class PoissonOpGradOpGPU : public OpKernel {
private:
  
public:
  explicit PoissonOpGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_ut = context->input(0);
    const Tensor& ut = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& f = context->input(3);
    const Tensor& h = context->input(4);
    
    
    const TensorShape& grad_ut_shape = grad_ut.shape();
    const TensorShape& ut_shape = ut.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& f_shape = f.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_ut_shape.dims(), 2);
    DCHECK_EQ(ut_shape.dims(), 2);
    DCHECK_EQ(u_shape.dims(), 2);
    DCHECK_EQ(f_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_f_shape(f_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_f = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_f_shape, &grad_f));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto f_tensor = f.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_ut_tensor = grad_ut.flat<double>().data();
    auto ut_tensor = ut.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_f_tensor = grad_f->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("PoissonOpGrad").Device(DEVICE_GPU), PoissonOpGradOpGPU);

#endif

#include "PoissonOp.h"
