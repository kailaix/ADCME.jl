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
#include "SparseConcate.h"

REGISTER_OP("SparseConcate")

.Input("ii1 : int64")
.Input("jj1 : int64")
.Input("vv1 : double")
.Input("m1 : int32")
.Input("n1 : int32")
.Input("ii2 : int64")
.Input("jj2 : int64")
.Input("vv2 : double")
.Input("m2 : int32")
.Input("n2 : int32")
.Input("hcat : bool")
.Output("ii : int64")
.Output("jj : int64")
.Output("vv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii1_shape));
        shape_inference::ShapeHandle jj1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &jj1_shape));
        shape_inference::ShapeHandle vv1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &vv1_shape));
        shape_inference::ShapeHandle m1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &m1_shape));
        shape_inference::ShapeHandle n1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &n1_shape));
        shape_inference::ShapeHandle ii2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &ii2_shape));
        shape_inference::ShapeHandle jj2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 1, &jj2_shape));
        shape_inference::ShapeHandle vv2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &vv2_shape));
        shape_inference::ShapeHandle m2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &m2_shape));
        shape_inference::ShapeHandle n2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 0, &n2_shape));
        shape_inference::ShapeHandle hcat_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(10), 0, &hcat_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("SparseConcateGrad")

.Input("grad_vv : double")
.Input("ii : int64")
.Input("jj : int64")
.Input("vv : double")
.Input("ii1 : int64")
.Input("jj1 : int64")
.Input("vv1 : double")
.Input("m1 : int32")
.Input("n1 : int32")
.Input("ii2 : int64")
.Input("jj2 : int64")
.Input("vv2 : double")
.Input("m2 : int32")
.Input("n2 : int32")
.Input("hcat : bool")
.Output("grad_ii1 : int64")
.Output("grad_jj1 : int64")
.Output("grad_vv1 : double")
.Output("grad_m1 : int32")
.Output("grad_n1 : int32")
.Output("grad_ii2 : int64")
.Output("grad_jj2 : int64")
.Output("grad_vv2 : double")
.Output("grad_m2 : int32")
.Output("grad_n2 : int32")
.Output("grad_hcat : bool");


class SparseConcateOp : public OpKernel {
private:
  
public:
  explicit SparseConcateOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(11, context->num_inputs());
    
    
    const Tensor& ii1 = context->input(0);
    const Tensor& jj1 = context->input(1);
    const Tensor& vv1 = context->input(2);
    const Tensor& m1 = context->input(3);
    const Tensor& n1 = context->input(4);
    const Tensor& ii2 = context->input(5);
    const Tensor& jj2 = context->input(6);
    const Tensor& vv2 = context->input(7);
    const Tensor& m2 = context->input(8);
    const Tensor& n2 = context->input(9);
    const Tensor& hcat = context->input(10);
    
    
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& m1_shape = m1.shape();
    const TensorShape& n1_shape = n1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& m2_shape = m2.shape();
    const TensorShape& n2_shape = n2.shape();
    const TensorShape& hcat_shape = hcat.shape();
    
    
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(m1_shape.dims(), 0);
    DCHECK_EQ(n1_shape.dims(), 0);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(m2_shape.dims(), 0);
    DCHECK_EQ(n2_shape.dims(), 0);
    DCHECK_EQ(hcat_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto m1_tensor = m1.flat<int32>().data();
    auto n1_tensor = n1.flat<int32>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto m2_tensor = m2.flat<int32>().data();
    auto n2_tensor = n2.flat<int32>().data();
    auto hcat_tensor = hcat.flat<bool>().data();

    // implement your forward function here 

    // TODO:
    int N1 = ii1_shape.dim_size(0), N2 = ii2_shape.dim_size(0);
    Forward fwd(ii1_tensor, jj1_tensor, vv1_tensor, N1,
        ii2_tensor, jj2_tensor, vv2_tensor, N2,
        *m1_tensor, *n1_tensor, *m2_tensor, *n2_tensor, *hcat_tensor);
    fwd.fill(context);
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseConcate").Device(DEVICE_CPU), SparseConcateOp);



class SparseConcateGradOp : public OpKernel {
private:
  
public:
  explicit SparseConcateGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& ii = context->input(1);
    const Tensor& jj = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& ii1 = context->input(4);
    const Tensor& jj1 = context->input(5);
    const Tensor& vv1 = context->input(6);
    const Tensor& m1 = context->input(7);
    const Tensor& n1 = context->input(8);
    const Tensor& ii2 = context->input(9);
    const Tensor& jj2 = context->input(10);
    const Tensor& vv2 = context->input(11);
    const Tensor& m2 = context->input(12);
    const Tensor& n2 = context->input(13);
    const Tensor& hcat = context->input(14);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& m1_shape = m1.shape();
    const TensorShape& n1_shape = n1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& m2_shape = m2.shape();
    const TensorShape& n2_shape = n2.shape();
    const TensorShape& hcat_shape = hcat.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(m1_shape.dims(), 0);
    DCHECK_EQ(n1_shape.dims(), 0);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(m2_shape.dims(), 0);
    DCHECK_EQ(n2_shape.dims(), 0);
    DCHECK_EQ(hcat_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ii1_shape(ii1_shape);
    TensorShape grad_jj1_shape(jj1_shape);
    TensorShape grad_vv1_shape(vv1_shape);
    TensorShape grad_m1_shape(m1_shape);
    TensorShape grad_n1_shape(n1_shape);
    TensorShape grad_ii2_shape(ii2_shape);
    TensorShape grad_jj2_shape(jj2_shape);
    TensorShape grad_vv2_shape(vv2_shape);
    TensorShape grad_m2_shape(m2_shape);
    TensorShape grad_n2_shape(n2_shape);
    TensorShape grad_hcat_shape(hcat_shape);
            
    // create output tensor
    
    Tensor* grad_ii1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii1_shape, &grad_ii1));
    Tensor* grad_jj1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_jj1_shape, &grad_jj1));
    Tensor* grad_vv1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_vv1_shape, &grad_vv1));
    Tensor* grad_m1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_m1_shape, &grad_m1));
    Tensor* grad_n1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_n1_shape, &grad_n1));
    Tensor* grad_ii2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_ii2_shape, &grad_ii2));
    Tensor* grad_jj2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_jj2_shape, &grad_jj2));
    Tensor* grad_vv2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_vv2_shape, &grad_vv2));
    Tensor* grad_m2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_m2_shape, &grad_m2));
    Tensor* grad_n2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_n2_shape, &grad_n2));
    Tensor* grad_hcat = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(10, grad_hcat_shape, &grad_hcat));
    
    // get the corresponding Eigen tensors for data access
    
    // auto ii1_tensor = ii1.flat<int64>().data();
    // auto jj1_tensor = jj1.flat<int64>().data();
    // auto vv1_tensor = vv1.flat<double>().data();
    // auto m1_tensor = m1.flat<int32>().data();
    // auto n1_tensor = n1.flat<int32>().data();
    // auto ii2_tensor = ii2.flat<int64>().data();
    // auto jj2_tensor = jj2.flat<int64>().data();
    // auto vv2_tensor = vv2.flat<double>().data();
    // auto m2_tensor = m2.flat<int32>().data();
    // auto n2_tensor = n2.flat<int32>().data();
    auto hcat_tensor = hcat.flat<bool>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    // auto jj_tensor = jj.flat<int64>().data();
    // auto vv_tensor = vv.flat<double>().data();
    // auto grad_ii1_tensor = grad_ii1->flat<int64>().data();
    // auto grad_jj1_tensor = grad_jj1->flat<int64>().data();
    auto grad_vv1_tensor = grad_vv1->flat<double>().data();
    // auto grad_m1_tensor = grad_m1->flat<int32>().data();
    // auto grad_n1_tensor = grad_n1->flat<int32>().data();
    // auto grad_ii2_tensor = grad_ii2->flat<int64>().data();
    // auto grad_jj2_tensor = grad_jj2->flat<int64>().data();
    auto grad_vv2_tensor = grad_vv2->flat<double>().data();
    // auto grad_m2_tensor = grad_m2->flat<int32>().data();
    // auto grad_n2_tensor = grad_n2->flat<int32>().data();
    // auto grad_hcat_tensor = grad_hcat->flat<bool>().data();   

    // implement your backward function here 

    // TODO:

    int N1 = ii1_shape.dim_size(0), N2 = ii2_shape.dim_size(0);
    backward(grad_vv1_tensor, grad_vv2_tensor, grad_vv_tensor, N1, N2);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseConcateGrad").Device(DEVICE_CPU), SparseConcateGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class SparseConcateOpGPU : public OpKernel {
private:
  
public:
  explicit SparseConcateOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(11, context->num_inputs());
    
    
    const Tensor& ii1 = context->input(0);
    const Tensor& jj1 = context->input(1);
    const Tensor& vv1 = context->input(2);
    const Tensor& m1 = context->input(3);
    const Tensor& n1 = context->input(4);
    const Tensor& ii2 = context->input(5);
    const Tensor& jj2 = context->input(6);
    const Tensor& vv2 = context->input(7);
    const Tensor& m2 = context->input(8);
    const Tensor& n2 = context->input(9);
    const Tensor& hcat = context->input(10);
    
    
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& m1_shape = m1.shape();
    const TensorShape& n1_shape = n1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& m2_shape = m2.shape();
    const TensorShape& n2_shape = n2.shape();
    const TensorShape& hcat_shape = hcat.shape();
    
    
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(m1_shape.dims(), 0);
    DCHECK_EQ(n1_shape.dims(), 0);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(m2_shape.dims(), 0);
    DCHECK_EQ(n2_shape.dims(), 0);
    DCHECK_EQ(hcat_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape ii_shape({-1});
    TensorShape jj_shape({-1});
    TensorShape vv_shape({-1});
            
    // create output tensor
    
    Tensor* ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii));
    Tensor* jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto m1_tensor = m1.flat<int32>().data();
    auto n1_tensor = n1.flat<int32>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto m2_tensor = m2.flat<int32>().data();
    auto n2_tensor = n2.flat<int32>().data();
    auto hcat_tensor = hcat.flat<bool>().data();
    auto ii_tensor = ii->flat<int64>().data();
    auto jj_tensor = jj->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseConcate").Device(DEVICE_GPU), SparseConcateOpGPU);

class SparseConcateGradOpGPU : public OpKernel {
private:
  
public:
  explicit SparseConcateGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& ii = context->input(1);
    const Tensor& jj = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& ii1 = context->input(4);
    const Tensor& jj1 = context->input(5);
    const Tensor& vv1 = context->input(6);
    const Tensor& m1 = context->input(7);
    const Tensor& n1 = context->input(8);
    const Tensor& ii2 = context->input(9);
    const Tensor& jj2 = context->input(10);
    const Tensor& vv2 = context->input(11);
    const Tensor& m2 = context->input(12);
    const Tensor& n2 = context->input(13);
    const Tensor& hcat = context->input(14);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& m1_shape = m1.shape();
    const TensorShape& n1_shape = n1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& m2_shape = m2.shape();
    const TensorShape& n2_shape = n2.shape();
    const TensorShape& hcat_shape = hcat.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(m1_shape.dims(), 0);
    DCHECK_EQ(n1_shape.dims(), 0);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(m2_shape.dims(), 0);
    DCHECK_EQ(n2_shape.dims(), 0);
    DCHECK_EQ(hcat_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ii1_shape(ii1_shape);
    TensorShape grad_jj1_shape(jj1_shape);
    TensorShape grad_vv1_shape(vv1_shape);
    TensorShape grad_m1_shape(m1_shape);
    TensorShape grad_n1_shape(n1_shape);
    TensorShape grad_ii2_shape(ii2_shape);
    TensorShape grad_jj2_shape(jj2_shape);
    TensorShape grad_vv2_shape(vv2_shape);
    TensorShape grad_m2_shape(m2_shape);
    TensorShape grad_n2_shape(n2_shape);
    TensorShape grad_hcat_shape(hcat_shape);
            
    // create output tensor
    
    Tensor* grad_ii1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii1_shape, &grad_ii1));
    Tensor* grad_jj1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_jj1_shape, &grad_jj1));
    Tensor* grad_vv1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_vv1_shape, &grad_vv1));
    Tensor* grad_m1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_m1_shape, &grad_m1));
    Tensor* grad_n1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_n1_shape, &grad_n1));
    Tensor* grad_ii2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_ii2_shape, &grad_ii2));
    Tensor* grad_jj2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_jj2_shape, &grad_jj2));
    Tensor* grad_vv2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_vv2_shape, &grad_vv2));
    Tensor* grad_m2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_m2_shape, &grad_m2));
    Tensor* grad_n2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_n2_shape, &grad_n2));
    Tensor* grad_hcat = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(10, grad_hcat_shape, &grad_hcat));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto m1_tensor = m1.flat<int32>().data();
    auto n1_tensor = n1.flat<int32>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto m2_tensor = m2.flat<int32>().data();
    auto n2_tensor = n2.flat<int32>().data();
    auto hcat_tensor = hcat.flat<bool>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_ii1_tensor = grad_ii1->flat<int64>().data();
    auto grad_jj1_tensor = grad_jj1->flat<int64>().data();
    auto grad_vv1_tensor = grad_vv1->flat<double>().data();
    auto grad_m1_tensor = grad_m1->flat<int32>().data();
    auto grad_n1_tensor = grad_n1->flat<int32>().data();
    auto grad_ii2_tensor = grad_ii2->flat<int64>().data();
    auto grad_jj2_tensor = grad_jj2->flat<int64>().data();
    auto grad_vv2_tensor = grad_vv2->flat<double>().data();
    auto grad_m2_tensor = grad_m2->flat<int32>().data();
    auto grad_n2_tensor = grad_n2->flat<int32>().data();
    auto grad_hcat_tensor = grad_hcat->flat<bool>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseConcateGrad").Device(DEVICE_GPU), SparseConcateGradOpGPU);

#endif