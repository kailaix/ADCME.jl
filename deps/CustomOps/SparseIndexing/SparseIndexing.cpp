#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
using namespace tensorflow;
// If you want to use the PyTorch feature, uncomment the following line
// #include "la.h" 
#include "SparseIndexing.h"

REGISTER_OP("SparseIndexing")

.Input("ii1 : int64")
  .Input("jj1 : int64")
  .Input("vv1 : double")
  .Input("m : int64")
  .Input("n : int64")
  .Input("ii : int64")
  .Input("jj : int64")
  .Output("ii2 : int64")
  .Output("jj2 : int64")
  .Output("vv2 : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii1_shape));
        shape_inference::ShapeHandle jj1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &jj1_shape));
        shape_inference::ShapeHandle vv1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &vv1_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &n_shape));
        shape_inference::ShapeHandle ii_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &ii_shape));
        shape_inference::ShapeHandle jj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 1, &jj_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });
class SparseIndexingOp : public OpKernel {
private:
  
public:
  explicit SparseIndexingOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(7, context->num_inputs());
    
    
    const Tensor& ii1 = context->input(0);
    const Tensor& jj1 = context->input(1);
    const Tensor& vv1 = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& ii = context->input(5);
    const Tensor& jj = context->input(6);
    
    
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    
    
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);

    // extra check
        
    // create output shape
    int64 n1 = ii1_shape.dim_size(0);

    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();

    int64 nz_ii = ii_shape.dim_size(0), nz_jj = jj_shape.dim_size(0);
    IJV ijv;
    forward(ii1_tensor, jj1_tensor, vv1_tensor, n1, *m_tensor, *n_tensor,
            ii_tensor, nz_ii, jj_tensor, nz_jj, ijv);
    
    int nz = ijv.get_size();
    TensorShape ii2_shape({nz});
    TensorShape jj2_shape({nz});
    TensorShape vv2_shape({nz});
            
    // create output tensor
    
    Tensor* ii2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii2_shape, &ii2));
    Tensor* jj2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj2_shape, &jj2));
    Tensor* vv2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv2_shape, &vv2));
    
    // get the corresponding Eigen tensors for data access
    
    
    auto ii2_tensor = ii2->flat<int64>().data();
    auto jj2_tensor = jj2->flat<int64>().data();
    auto vv2_tensor = vv2->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    for(int k = 0; k<nz;k++){
      ii2_tensor[k] =  ijv.ii[k];
      jj2_tensor[k] =  ijv.jj[k];
      vv2_tensor[k] =  ijv.vv[k];
    }
    // int64 p = 0;
    // for (int64 k = 0; k < C.outerSize(); ++k){
    //     for (SpMat::InnerIterator it(C, k); it; ++it){
    //         ii2_tensor[p] = it.row()+1;
    //         jj2_tensor[p] = it.col()+1; 
    //         vv2_tensor[p] = it.value(); 
    //         p++;
    //     }
    // }
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseIndexing").Device(DEVICE_CPU), SparseIndexingOp);


REGISTER_OP("SparseIndexingGrad")
  
  // .Input("grad_ii2 : int64")
// .Input("grad_jj2 : int64")
.Input("grad_vv2 : double")
  .Input("ii2 : int64")
  .Input("jj2 : int64")
  .Input("vv2 : double")
  .Input("ii1 : int64")
  .Input("jj1 : int64")
  .Input("vv1 : double")
  .Input("m : int64")
  .Input("n : int64")
  .Input("ii : int64")
  .Input("jj : int64")
  .Output("grad_ii1 : int64")
  .Output("grad_jj1 : int64")
  .Output("grad_vv1 : double")
  .Output("grad_m : int64")
  .Output("grad_n : int64")
  .Output("grad_ii : int64")
  .Output("grad_jj : int64");
class SparseIndexingGradOp : public OpKernel {
private:
  
public:
  explicit SparseIndexingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv2 = context->input(0);
    const Tensor& ii2 = context->input(1);
    const Tensor& jj2 = context->input(2);
    const Tensor& vv2 = context->input(3);
    const Tensor& ii1 = context->input(4);
    const Tensor& jj1 = context->input(5);
    const Tensor& vv1 = context->input(6);
    const Tensor& m = context->input(7);
    const Tensor& n = context->input(8);
    const Tensor& ii = context->input(9);
    const Tensor& jj = context->input(10);
    
    
    const TensorShape& grad_vv2_shape = grad_vv2.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    
    
    DCHECK_EQ(grad_vv2_shape.dims(), 1);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);

    // extra check
    // int64 m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ii1_shape(ii1_shape);
    TensorShape grad_jj1_shape(jj1_shape);
    TensorShape grad_vv1_shape(vv1_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_jj_shape(jj_shape);
            
    // create output tensor
    
    Tensor* grad_ii1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii1_shape, &grad_ii1));
    Tensor* grad_jj1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_jj1_shape, &grad_jj1));
    Tensor* grad_vv1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_vv1_shape, &grad_vv1));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_n_shape, &grad_n));
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_ii_shape, &grad_ii));
    Tensor* grad_jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_jj_shape, &grad_jj));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto grad_vv2_tensor = grad_vv2.flat<double>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto grad_ii1_tensor = grad_ii1->flat<int64>().data();
    auto grad_jj1_tensor = grad_jj1->flat<int64>().data();
    auto grad_vv1_tensor = grad_vv1->flat<double>().data();
    auto grad_m_tensor = grad_m->flat<int64>().data();
    auto grad_n_tensor = grad_n->flat<int64>().data();
    auto grad_ii_tensor = grad_ii->flat<int64>().data();
    auto grad_jj_tensor = grad_jj->flat<int64>().data();   

    // implement your backward function here 

    // TODO:
    int64 n1 = ii1_shape.dim_size(0), nnz = ii2_shape.dim_size(0), nz_ii = ii_shape.dim_size(0), nz_jj = jj_shape.dim_size(0);
    backward(grad_vv1_tensor, grad_vv2_tensor, ii2_tensor, jj2_tensor, nnz, ii1_tensor, jj1_tensor, n1,
          ii_tensor, nz_ii, jj_tensor, nz_jj);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseIndexingGrad").Device(DEVICE_CPU), SparseIndexingGradOp);

