#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "SparseCompress.h"


REGISTER_OP("SparseCompress")
.Input("indices : int64")
.Input("v : double")
.Output("nindices : int64")
.Output("nv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle indices_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices_shape));
        shape_inference::ShapeHandle v_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &v_shape));

        c->set_output(0, c->Matrix(-1,2));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("SparseCompressGrad")
.Input("grad_nv : double")
.Input("nindices : int64")
.Input("nv : double")
.Input("indices : int64")
.Input("v : double")
.Output("grad_indices : int64")
.Output("grad_v : double");

/*-------------------------------------------------------------------------------------*/

class SparseCompressOp : public OpKernel {
private:
  
public:
  explicit SparseCompressOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& indices = context->input(0);
    const Tensor& v = context->input(1);
    
    
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& v_shape = v.shape();
    
    
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(v_shape.dims(), 1);

    // extra check
        
    // create output shape
    auto indices_tensor = indices.flat<int64>().data();
    auto v_tensor = v.flat<double>().data();

    int N = v_shape.dim_size(0);
    SparseCompressor sc(indices_tensor, v_tensor, N);
    
    TensorShape nindices_shape({sc.nout,2});
    TensorShape nv_shape({sc.nout});
            
    // create output tensor
    
    Tensor* nindices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, nindices_shape, &nindices));
    Tensor* nv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, nv_shape, &nv));
    
    // get the corresponding Eigen tensors for data access
    
    
    auto nindices_tensor = nindices->flat<int64>().data();
    auto nv_tensor = nv->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    sc.forward(nindices_tensor, nv_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseCompress").Device(DEVICE_CPU), SparseCompressOp);



class SparseCompressGradOp : public OpKernel {
private:
  
public:
  explicit SparseCompressGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_nv = context->input(0);
    const Tensor& nindices = context->input(1);
    const Tensor& nv = context->input(2);
    const Tensor& indices = context->input(3);
    const Tensor& v = context->input(4);
    
    
    const TensorShape& grad_nv_shape = grad_nv.shape();
    const TensorShape& nindices_shape = nindices.shape();
    const TensorShape& nv_shape = nv.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& v_shape = v.shape();
    
    
    DCHECK_EQ(grad_nv_shape.dims(), 1);
    DCHECK_EQ(nindices_shape.dims(), 2);
    DCHECK_EQ(nv_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(v_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_indices_shape(indices_shape);
    TensorShape grad_v_shape(v_shape);
            
    // create output tensor
    
    Tensor* grad_indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_indices_shape, &grad_indices));
    Tensor* grad_v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_v_shape, &grad_v));
    
    // get the corresponding Eigen tensors for data access
    
    auto indices_tensor = indices.flat<int64>().data();
    auto v_tensor = v.flat<double>().data();
    auto grad_nv_tensor = grad_nv.flat<double>().data();
    auto nindices_tensor = nindices.flat<int64>().data();
    auto nv_tensor = nv.flat<double>().data();
    auto grad_v_tensor = grad_v->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int N = v_shape.dim_size(0);
    SparseCompressor sc(indices_tensor, v_tensor, N);

    sc.backward(grad_v_tensor, grad_nv_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseCompressGrad").Device(DEVICE_CPU), SparseCompressGradOp);
