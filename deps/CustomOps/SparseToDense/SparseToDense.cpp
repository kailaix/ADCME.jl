#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "SparseToDense.h"


REGISTER_OP("SparseToDenseAD")
.Input("ij : int64")
.Input("vv : double")
.Input("m : int64")
.Input("n : int64")
.Output("a : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ij_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &ij_shape));
        shape_inference::ShapeHandle vv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &vv_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &n_shape));

        c->set_output(0, c->Matrix(-1,-1));
    return Status::OK();
  });

REGISTER_OP("SparseToDenseADGrad")
.Input("grad_a : double")
.Input("a : double")
.Input("ij : int64")
.Input("vv : double")
.Input("m : int64")
.Input("n : int64")
.Output("grad_ij : int64")
.Output("grad_vv : double")
.Output("grad_m : int64")
.Output("grad_n : int64");

/*-------------------------------------------------------------------------------------*/

class SparseToDenseADOp : public OpKernel {
private:
  
public:
  explicit SparseToDenseADOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& ij = context->input(0);
    const Tensor& vv = context->input(1);
    const Tensor& m = context->input(2);
    const Tensor& n = context->input(3);
    
    
    const TensorShape& ij_shape = ij.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    
    
    DCHECK_EQ(ij_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);

    // extra check
        
    // create output shape
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    int m_ = *m_tensor, n_ = *n_tensor;
    TensorShape a_shape({m_, n_});
            
    // create output tensor
    
    Tensor* a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, a_shape, &a));
    
    // get the corresponding Eigen tensors for data access
    
    auto ij_tensor = ij.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    
    auto a_tensor = a->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    int N = vv_shape.dim_size(0);
    a->flat<double>().setZero();
    forward(a_tensor, ij_tensor, vv_tensor, N, m_, n_);

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseToDenseAD").Device(DEVICE_CPU), SparseToDenseADOp);



class SparseToDenseADGradOp : public OpKernel {
private:
  
public:
  explicit SparseToDenseADGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_a = context->input(0);
    const Tensor& a = context->input(1);
    const Tensor& ij = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& m = context->input(4);
    const Tensor& n = context->input(5);
    
    
    const TensorShape& grad_a_shape = grad_a.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& ij_shape = ij.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    
    
    DCHECK_EQ(grad_a_shape.dims(), 2);
    DCHECK_EQ(a_shape.dims(), 2);
    DCHECK_EQ(ij_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ij_shape(ij_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
            
    // create output tensor
    
    Tensor* grad_ij = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ij_shape, &grad_ij));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_vv_shape, &grad_vv));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_n_shape, &grad_n));
    
    // get the corresponding Eigen tensors for data access
    
    auto ij_tensor = ij.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto grad_a_tensor = grad_a.flat<double>().data();
    auto a_tensor = a.flat<double>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int N = vv_shape.dim_size(0);
    backward(grad_vv_tensor, grad_a_tensor, ij_tensor, N, *m_tensor, *n_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseToDenseADGrad").Device(DEVICE_CPU), SparseToDenseADGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class SparseToDenseADOpGPU : public OpKernel {
private:
  
public:
  explicit SparseToDenseADOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& ij = context->input(0);
    const Tensor& vv = context->input(1);
    const Tensor& m = context->input(2);
    const Tensor& n = context->input(3);
    
    
    const TensorShape& ij_shape = ij.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    
    
    DCHECK_EQ(ij_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape a_shape({-1,-1});
            
    // create output tensor
    
    Tensor* a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, a_shape, &a));
    
    // get the corresponding Eigen tensors for data access
    
    auto ij_tensor = ij.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto a_tensor = a->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseToDenseAD").Device(DEVICE_GPU), SparseToDenseADOpGPU);

class SparseToDenseADGradOpGPU : public OpKernel {
private:
  
public:
  explicit SparseToDenseADGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_a = context->input(0);
    const Tensor& a = context->input(1);
    const Tensor& ij = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& m = context->input(4);
    const Tensor& n = context->input(5);
    
    
    const TensorShape& grad_a_shape = grad_a.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& ij_shape = ij.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    
    
    DCHECK_EQ(grad_a_shape.dims(), 2);
    DCHECK_EQ(a_shape.dims(), 2);
    DCHECK_EQ(ij_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ij_shape(ij_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
            
    // create output tensor
    
    Tensor* grad_ij = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ij_shape, &grad_ij));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_vv_shape, &grad_vv));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_n_shape, &grad_n));
    
    // get the corresponding Eigen tensors for data access
    
    auto ij_tensor = ij.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto grad_a_tensor = grad_a.flat<double>().data();
    auto a_tensor = a.flat<double>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseToDenseADGrad").Device(DEVICE_GPU), SparseToDenseADGradOpGPU);

#endif