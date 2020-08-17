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
#include "SparseScatterUpdate.h"


REGISTER_OP("SparseScatterUpdate")

.Input("ii1 : int64")
.Input("jj1 : int64")
.Input("vv1 : double")
.Input("m1 : int64")
.Input("n1 : int64")
.Input("ii2 : int64")
.Input("jj2 : int64")
.Input("vv2 : double")
.Input("ii : int64")
.Input("jj : int64")
.Output("nii : int64")
.Output("njj : int64")
.Output("nvv : double")
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
        shape_inference::ShapeHandle ii_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 1, &ii_shape));
        shape_inference::ShapeHandle jj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 1, &jj_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("SparseScatterUpdateGrad")

.Input("grad_nvv : double")
.Input("nii : int64")
.Input("njj : int64")
.Input("nvv : double")
.Input("ii1 : int64")
.Input("jj1 : int64")
.Input("vv1 : double")
.Input("m1 : int64")
.Input("n1 : int64")
.Input("ii2 : int64")
.Input("jj2 : int64")
.Input("vv2 : double")
.Input("ii : int64")
.Input("jj : int64")
.Output("grad_ii1 : int64")
.Output("grad_jj1 : int64")
.Output("grad_vv1 : double")
.Output("grad_m1 : int64")
.Output("grad_n1 : int64")
.Output("grad_ii2 : int64")
.Output("grad_jj2 : int64")
.Output("grad_vv2 : double")
.Output("grad_ii : int64")
.Output("grad_jj : int64");


class SparseScatterUpdateOp : public OpKernel {
private:
  
public:
  explicit SparseScatterUpdateOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(10, context->num_inputs());
    
    
    const Tensor& ii1 = context->input(0);
    const Tensor& jj1 = context->input(1);
    const Tensor& vv1 = context->input(2);
    const Tensor& m1 = context->input(3);
    const Tensor& n1 = context->input(4);
    const Tensor& ii2 = context->input(5);
    const Tensor& jj2 = context->input(6);
    const Tensor& vv2 = context->input(7);
    const Tensor& ii = context->input(8);
    const Tensor& jj = context->input(9);
    
    
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& m1_shape = m1.shape();
    const TensorShape& n1_shape = n1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    
    
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(m1_shape.dims(), 0);
    DCHECK_EQ(n1_shape.dims(), 0);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto m1_tensor = m1.flat<int64>().data();
    auto n1_tensor = n1.flat<int64>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();  

    // implement your forward function here 

    int on = ii1_shape.dim_size(0), un = ii2_shape.dim_size(0), m = *m1_tensor, n = *n1_tensor;
    int ni = ii_shape.dim_size(0), nj = jj_shape.dim_size(0);
    IJV_SparseScatterUpdate ijv;
    forward(ii1_tensor, jj1_tensor, vv1_tensor, on, 
          ii2_tensor, jj2_tensor, vv2_tensor, un, m, n, ii_tensor, jj_tensor, ni, nj, ijv);

    int nz = ijv.get_size();
    TensorShape nii_shape({nz});
    TensorShape njj_shape({nz});
    TensorShape nvv_shape({nz});
            
    // create output tensor
    
    Tensor* nii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, nii_shape, &nii));
    Tensor* njj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, njj_shape, &njj));
    Tensor* nvv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, nvv_shape, &nvv));


    auto nii_tensor = nii->flat<int64>().data();
    auto njj_tensor = njj->flat<int64>().data();
    auto nvv_tensor = nvv->flat<double>().data(); 

    for(int i=0;i<nz;i++){
      nii_tensor[i] = ijv.ii[i];
      njj_tensor[i] = ijv.jj[i];
      nvv_tensor[i] = ijv.vv[i];
    }

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseScatterUpdate").Device(DEVICE_CPU), SparseScatterUpdateOp);



class SparseScatterUpdateGradOp : public OpKernel {
private:
  
public:
  explicit SparseScatterUpdateGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_nvv = context->input(0);
    const Tensor& nii = context->input(1);
    const Tensor& njj = context->input(2);
    const Tensor& nvv = context->input(3);
    const Tensor& ii1 = context->input(4);
    const Tensor& jj1 = context->input(5);
    const Tensor& vv1 = context->input(6);
    const Tensor& m1 = context->input(7);
    const Tensor& n1 = context->input(8);
    const Tensor& ii2 = context->input(9);
    const Tensor& jj2 = context->input(10);
    const Tensor& vv2 = context->input(11);
    const Tensor& ii = context->input(12);
    const Tensor& jj = context->input(13);
    
    
    const TensorShape& grad_nvv_shape = grad_nvv.shape();
    const TensorShape& nii_shape = nii.shape();
    const TensorShape& njj_shape = njj.shape();
    const TensorShape& nvv_shape = nvv.shape();
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& m1_shape = m1.shape();
    const TensorShape& n1_shape = n1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    
    
    DCHECK_EQ(grad_nvv_shape.dims(), 1);
    DCHECK_EQ(nii_shape.dims(), 1);
    DCHECK_EQ(njj_shape.dims(), 1);
    DCHECK_EQ(nvv_shape.dims(), 1);
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(m1_shape.dims(), 0);
    DCHECK_EQ(n1_shape.dims(), 0);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);

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
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_jj_shape(jj_shape);
            
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
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_ii_shape, &grad_ii));
    Tensor* grad_jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_jj_shape, &grad_jj));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto m1_tensor = m1.flat<int64>().data();
    auto n1_tensor = n1.flat<int64>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto grad_nvv_tensor = grad_nvv.flat<double>().data();
    auto nii_tensor = nii.flat<int64>().data();
    auto njj_tensor = njj.flat<int64>().data();
    auto nvv_tensor = nvv.flat<double>().data();
    auto grad_ii1_tensor = grad_ii1->flat<int64>().data();
    auto grad_jj1_tensor = grad_jj1->flat<int64>().data();
    auto grad_vv1_tensor = grad_vv1->flat<double>().data();
    auto grad_m1_tensor = grad_m1->flat<int64>().data();
    auto grad_n1_tensor = grad_n1->flat<int64>().data();
    auto grad_ii2_tensor = grad_ii2->flat<int64>().data();
    auto grad_jj2_tensor = grad_jj2->flat<int64>().data();
    auto grad_vv2_tensor = grad_vv2->flat<double>().data();
    auto grad_ii_tensor = grad_ii->flat<int64>().data();
    auto grad_jj_tensor = grad_jj->flat<int64>().data();   

    // implement your backward function here 

    int on = ii1_shape.dim_size(0), un = ii2_shape.dim_size(0), m = *m1_tensor, n = *n1_tensor;
    int ni = ii_shape.dim_size(0), nj = jj_shape.dim_size(0);
    int out_n = nvv_shape.dim_size(0);
    backward(
          grad_vv1_tensor, grad_vv2_tensor, grad_nvv_tensor,
          nii_tensor, njj_tensor, nvv_tensor, out_n, 
          ii1_tensor, jj1_tensor, vv1_tensor, on, 
          ii2_tensor, jj2_tensor, vv2_tensor, un, m, n, ii_tensor, jj_tensor, ni, nj);

    
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseScatterUpdateGrad").Device(DEVICE_CPU), SparseScatterUpdateGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class SparseScatterUpdateOpGPU : public OpKernel {
private:
  
public:
  explicit SparseScatterUpdateOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(10, context->num_inputs());
    
    
    const Tensor& ii1 = context->input(0);
    const Tensor& jj1 = context->input(1);
    const Tensor& vv1 = context->input(2);
    const Tensor& m1 = context->input(3);
    const Tensor& n1 = context->input(4);
    const Tensor& ii2 = context->input(5);
    const Tensor& jj2 = context->input(6);
    const Tensor& vv2 = context->input(7);
    const Tensor& ii = context->input(8);
    const Tensor& jj = context->input(9);
    
    
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& m1_shape = m1.shape();
    const TensorShape& n1_shape = n1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    
    
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(m1_shape.dims(), 0);
    DCHECK_EQ(n1_shape.dims(), 0);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape nii_shape({-1});
    TensorShape njj_shape({-1});
    TensorShape nvv_shape({-1});
            
    // create output tensor
    
    Tensor* nii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, nii_shape, &nii));
    Tensor* njj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, njj_shape, &njj));
    Tensor* nvv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, nvv_shape, &nvv));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto m1_tensor = m1.flat<int64>().data();
    auto n1_tensor = n1.flat<int64>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto nii_tensor = nii->flat<int64>().data();
    auto njj_tensor = njj->flat<int64>().data();
    auto nvv_tensor = nvv->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseScatterUpdate").Device(DEVICE_GPU), SparseScatterUpdateOpGPU);

class SparseScatterUpdateGradOpGPU : public OpKernel {
private:
  
public:
  explicit SparseScatterUpdateGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_nvv = context->input(0);
    const Tensor& nii = context->input(1);
    const Tensor& njj = context->input(2);
    const Tensor& nvv = context->input(3);
    const Tensor& ii1 = context->input(4);
    const Tensor& jj1 = context->input(5);
    const Tensor& vv1 = context->input(6);
    const Tensor& m1 = context->input(7);
    const Tensor& n1 = context->input(8);
    const Tensor& ii2 = context->input(9);
    const Tensor& jj2 = context->input(10);
    const Tensor& vv2 = context->input(11);
    const Tensor& ii = context->input(12);
    const Tensor& jj = context->input(13);
    
    
    const TensorShape& grad_nvv_shape = grad_nvv.shape();
    const TensorShape& nii_shape = nii.shape();
    const TensorShape& njj_shape = njj.shape();
    const TensorShape& nvv_shape = nvv.shape();
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& m1_shape = m1.shape();
    const TensorShape& n1_shape = n1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    
    
    DCHECK_EQ(grad_nvv_shape.dims(), 1);
    DCHECK_EQ(nii_shape.dims(), 1);
    DCHECK_EQ(njj_shape.dims(), 1);
    DCHECK_EQ(nvv_shape.dims(), 1);
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(m1_shape.dims(), 0);
    DCHECK_EQ(n1_shape.dims(), 0);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);

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
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_jj_shape(jj_shape);
            
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
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_ii_shape, &grad_ii));
    Tensor* grad_jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_jj_shape, &grad_jj));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto m1_tensor = m1.flat<int64>().data();
    auto n1_tensor = n1.flat<int64>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto grad_nvv_tensor = grad_nvv.flat<double>().data();
    auto nii_tensor = nii.flat<int64>().data();
    auto njj_tensor = njj.flat<int64>().data();
    auto nvv_tensor = nvv.flat<double>().data();
    auto grad_ii1_tensor = grad_ii1->flat<int64>().data();
    auto grad_jj1_tensor = grad_jj1->flat<int64>().data();
    auto grad_vv1_tensor = grad_vv1->flat<double>().data();
    auto grad_m1_tensor = grad_m1->flat<int64>().data();
    auto grad_n1_tensor = grad_n1->flat<int64>().data();
    auto grad_ii2_tensor = grad_ii2->flat<int64>().data();
    auto grad_jj2_tensor = grad_jj2->flat<int64>().data();
    auto grad_vv2_tensor = grad_vv2->flat<double>().data();
    auto grad_ii_tensor = grad_ii->flat<int64>().data();
    auto grad_jj_tensor = grad_jj->flat<int64>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseScatterUpdateGrad").Device(DEVICE_GPU), SparseScatterUpdateGradOpGPU);

#endif