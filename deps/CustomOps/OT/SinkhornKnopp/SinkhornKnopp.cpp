#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
#include "SinkhornKnopp.h"

#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif

using namespace tensorflow;

REGISTER_OP("SinkhornKnopp")

.Input("a : double")
  .Input("b : double")
  .Input("m : double")
  .Input("reg : double")
  .Input("iter : int64")
  .Input("tol : double")
  .Input("method : int64")
  .Output("p : double")
  .Output("l : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_shape));
        shape_inference::ShapeHandle b_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &b_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &m_shape));
        shape_inference::ShapeHandle reg_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &reg_shape));
        shape_inference::ShapeHandle iter_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &iter_shape));
        shape_inference::ShapeHandle tol_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &tol_shape));
        shape_inference::ShapeHandle method_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &method_shape));

        c->set_output(0, c->input(2));
        c->set_output(1, c->Scalar());
    return Status::OK();
  });

REGISTER_OP("SinkhornKnoppGrad")

.Input("grad_p : double")
.Input("grad_l : double")
  .Input("p : double")
  .Input("l : double")
  .Input("a : double")
  .Input("b : double")
  .Input("m : double")
  .Input("reg : double")
  .Input("iter : int64")
  .Input("tol : double")
  .Input("method : int64")
  .Output("grad_a : double")
  .Output("grad_b : double")
  .Output("grad_m : double")
  .Output("grad_reg : double")
  .Output("grad_iter : int64")
  .Output("grad_tol : double")
  .Output("grad_method : int64");


class SinkhornKnoppOp : public OpKernel {
private:
  
public:
  explicit SinkhornKnoppOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(7, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& m = context->input(2);
    const Tensor& reg = context->input(3);
    const Tensor& iter = context->input(4);
    const Tensor& tol = context->input(5);
    const Tensor& method = context->input(6);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& reg_shape = reg.shape();
    const TensorShape& iter_shape = iter.shape();
    const TensorShape& tol_shape = tol.shape();
    const TensorShape& method_shape = method.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 2);
    DCHECK_EQ(reg_shape.dims(), 0);
    DCHECK_EQ(iter_shape.dims(), 0);
    DCHECK_EQ(tol_shape.dims(), 0);
    DCHECK_EQ(method_shape.dims(), 0);

    // extra check
        
    // create output shape
    int dim_a = a_shape.dim_size(0), dim_b = b_shape.dim_size(0);
    DCHECK_EQ(m_shape.dim_size(0), dim_a);
    DCHECK_EQ(m_shape.dim_size(1), dim_b);

    TensorShape p_shape({dim_a, dim_b});
    TensorShape l_shape({});
            
    // create output tensor
    
    Tensor* p = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, p_shape, &p));
    Tensor* l = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, l_shape, &l));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto m_tensor = m.flat<double>().data();
    auto reg_tensor = reg.flat<double>().data();
    auto iter_tensor = iter.flat<int64>().data();
    auto tol_tensor = tol.flat<double>().data();
    auto method_tensor = method.flat<int64>().data();
    auto p_tensor = p->flat<double>().data();
    auto l_tensor = l->flat<double>().data();   

    // implement your forward function here 

    // TODO:

    forward(l_tensor, p_tensor, a_tensor, b_tensor, m_tensor, dim_a, dim_b, 
          *reg_tensor, *iter_tensor, *tol_tensor, *method_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("SinkhornKnopp").Device(DEVICE_CPU), SinkhornKnoppOp);



class SinkhornKnoppGradOp : public OpKernel {
private:
  
public:
  explicit SinkhornKnoppGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_p = context->input(0);
    const Tensor& grad_l = context->input(1);
    const Tensor& p = context->input(2);
    const Tensor& l = context->input(3);
    const Tensor& a = context->input(4);
    const Tensor& b = context->input(5);
    const Tensor& m = context->input(6);
    const Tensor& reg = context->input(7);
    const Tensor& iter = context->input(8);
    const Tensor& tol = context->input(9);
    const Tensor& method = context->input(10);
    
    
    const TensorShape& grad_p_shape = grad_p.shape();
    const TensorShape& grad_l_shape = grad_l.shape();
    const TensorShape& p_shape = p.shape();
    const TensorShape& l_shape = l.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& reg_shape = reg.shape();
    const TensorShape& iter_shape = iter.shape();
    const TensorShape& tol_shape = tol.shape();
    const TensorShape& method_shape = method.shape();
    
    
    DCHECK_EQ(grad_p_shape.dims(), 2);
    DCHECK_EQ(grad_l_shape.dims(), 0);
    DCHECK_EQ(p_shape.dims(), 2);
    DCHECK_EQ(l_shape.dims(), 0);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 2);
    DCHECK_EQ(reg_shape.dims(), 0);
    DCHECK_EQ(iter_shape.dims(), 0);
    DCHECK_EQ(tol_shape.dims(), 0);
    DCHECK_EQ(method_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
    int dim_a = a_shape.dim_size(0), dim_b = b_shape.dim_size(0);
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_b_shape(b_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_reg_shape(reg_shape);
    TensorShape grad_iter_shape(iter_shape);
    TensorShape grad_tol_shape(tol_shape);
    TensorShape grad_method_shape(method_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_b_shape, &grad_b));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_m_shape, &grad_m));
    Tensor* grad_reg = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_reg_shape, &grad_reg));
    Tensor* grad_iter = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_iter_shape, &grad_iter));
    Tensor* grad_tol = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_tol_shape, &grad_tol));
    Tensor* grad_method = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_method_shape, &grad_method));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto m_tensor = m.flat<double>().data();
    auto reg_tensor = reg.flat<double>().data();
    auto iter_tensor = iter.flat<int64>().data();
    auto tol_tensor = tol.flat<double>().data();
    auto method_tensor = method.flat<int64>().data();
    auto grad_p_tensor = grad_p.flat<double>().data();
    auto grad_l_tensor = grad_l.flat<double>().data();
    auto p_tensor = p.flat<double>().data();
    auto l_tensor = l.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_b_tensor = grad_b->flat<double>().data();
    auto grad_m_tensor = grad_m->flat<double>().data();
    auto grad_reg_tensor = grad_reg->flat<double>().data();
    auto grad_iter_tensor = grad_iter->flat<int64>().data();
    auto grad_tol_tensor = grad_tol->flat<double>().data();
    auto grad_method_tensor = grad_method->flat<int64>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_m_tensor, p_tensor, dim_a, dim_b, *method_tensor);    
  }
};
REGISTER_KERNEL_BUILDER(Name("SinkhornKnoppGrad").Device(DEVICE_CPU), SinkhornKnoppGradOp);

#ifdef USE_GPU
class SinkhornKnoppOpGPU : public OpKernel {
private:
  
public:
  explicit SinkhornKnoppOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(7, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& m = context->input(2);
    const Tensor& reg = context->input(3);
    const Tensor& iter = context->input(4);
    const Tensor& tol = context->input(5);
    const Tensor& method = context->input(6);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& reg_shape = reg.shape();
    const TensorShape& iter_shape = iter.shape();
    const TensorShape& tol_shape = tol.shape();
    const TensorShape& method_shape = method.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 2);
    DCHECK_EQ(reg_shape.dims(), 0);
    DCHECK_EQ(iter_shape.dims(), 0);
    DCHECK_EQ(tol_shape.dims(), 0);
    DCHECK_EQ(method_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape p_shape({-1,-1});
    TensorShape l_shape({});
            
    // create output tensor
    
    Tensor* p = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, p_shape, &p));
    Tensor* l = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, l_shape, &l));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto m_tensor = m.flat<double>().data();
    auto reg_tensor = reg.flat<double>().data();
    auto iter_tensor = iter.flat<int64>().data();
    auto tol_tensor = tol.flat<double>().data();
    auto method_tensor = method.flat<int64>().data();
    auto p_tensor = p->flat<double>().data();
    auto l_tensor = l->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("SinkhornKnopp").Device(DEVICE_GPU), SinkhornKnoppOpGPU);



class SinkhornKnoppGradOpGPU : public OpKernel {
private:
  
public:
  explicit SinkhornKnoppGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_p = context->input(0);
    const Tensor& grad_l = context->input(1);
    const Tensor& p = context->input(2);
    const Tensor& l = context->input(3);
    const Tensor& a = context->input(4);
    const Tensor& b = context->input(5);
    const Tensor& m = context->input(6);
    const Tensor& reg = context->input(7);
    const Tensor& iter = context->input(8);
    const Tensor& tol = context->input(9);
    const Tensor& method = context->input(10);
    
    
    const TensorShape& grad_p_shape = grad_p.shape();
    const TensorShape& grad_l_shape = grad_l.shape();
    const TensorShape& p_shape = p.shape();
    const TensorShape& l_shape = l.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& reg_shape = reg.shape();
    const TensorShape& iter_shape = iter.shape();
    const TensorShape& tol_shape = tol.shape();
    const TensorShape& method_shape = method.shape();
    
    
    DCHECK_EQ(grad_p_shape.dims(), 2);
    DCHECK_EQ(grad_l_shape.dims(), 0);
    DCHECK_EQ(p_shape.dims(), 2);
    DCHECK_EQ(l_shape.dims(), 0);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 2);
    DCHECK_EQ(reg_shape.dims(), 0);
    DCHECK_EQ(iter_shape.dims(), 0);
    DCHECK_EQ(tol_shape.dims(), 0);
    DCHECK_EQ(method_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_b_shape(b_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_reg_shape(reg_shape);
    TensorShape grad_iter_shape(iter_shape);
    TensorShape grad_tol_shape(tol_shape);
    TensorShape grad_method_shape(method_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_b_shape, &grad_b));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_m_shape, &grad_m));
    Tensor* grad_reg = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_reg_shape, &grad_reg));
    Tensor* grad_iter = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_iter_shape, &grad_iter));
    Tensor* grad_tol = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_tol_shape, &grad_tol));
    Tensor* grad_method = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_method_shape, &grad_method));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto m_tensor = m.flat<double>().data();
    auto reg_tensor = reg.flat<double>().data();
    auto iter_tensor = iter.flat<int64>().data();
    auto tol_tensor = tol.flat<double>().data();
    auto method_tensor = method.flat<int64>().data();
    auto grad_p_tensor = grad_p.flat<double>().data();
    auto grad_l_tensor = grad_l.flat<double>().data();
    auto p_tensor = p.flat<double>().data();
    auto l_tensor = l.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_b_tensor = grad_b->flat<double>().data();
    auto grad_m_tensor = grad_m->flat<double>().data();
    auto grad_reg_tensor = grad_reg->flat<double>().data();
    auto grad_iter_tensor = grad_iter->flat<int64>().data();
    auto grad_tol_tensor = grad_tol->flat<double>().data();
    auto grad_method_tensor = grad_method->flat<int64>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SinkhornKnoppGrad").Device(DEVICE_GPU), SinkhornKnoppGradOpGPU);

#endif