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
#include "SparseMatMul.h"

REGISTER_OP("SparseSparseMatMul")

.Input("ii1 : int64")
  .Input("jj1 : int64")
  .Input("vv1 : double")
  .Input("ii2 : int64")
  .Input("jj2 : int64")
  .Input("vv2 : double")
  .Input("m : int64")
  .Input("n : int64")
  .Input("k : int64")
  .Output("ii3 : int64")
  .Output("jj3 : int64")
  .Output("vv3 : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii1_shape));
        shape_inference::ShapeHandle jj1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &jj1_shape));
        shape_inference::ShapeHandle vv1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &vv1_shape));
        shape_inference::ShapeHandle ii2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &ii2_shape));
        shape_inference::ShapeHandle jj2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &jj2_shape));
        shape_inference::ShapeHandle vv2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &vv2_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &n_shape));
        shape_inference::ShapeHandle k_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &k_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });
class SparseSparseMatMulOp : public OpKernel {
private:
  
public:
  explicit SparseSparseMatMulOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(9, context->num_inputs());
    
    
    const Tensor& ii1 = context->input(0);
    const Tensor& jj1 = context->input(1);
    const Tensor& vv1 = context->input(2);
    const Tensor& ii2 = context->input(3);
    const Tensor& jj2 = context->input(4);
    const Tensor& vv2 = context->input(5);
    const Tensor& m = context->input(6);
    const Tensor& n = context->input(7);
    const Tensor& k = context->input(8);
    
    
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& k_shape = k.shape();
    
    
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(k_shape.dims(), 0);

    int n1 = ii1_shape.dim_size(0); int n2 = ii2_shape.dim_size(0);
    // extra check
        
    // create output shape
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto k_tensor = k.flat<int64>().data();
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    SpMat A(*m_tensor, *k_tensor);
    forward(ii1_tensor, jj1_tensor, vv1_tensor, n1, 
            ii2_tensor, jj2_tensor, vv2_tensor, n2, 
            *m_tensor, *n_tensor, *k_tensor, &A);
    
    int n3 = A.nonZeros();
    TensorShape ii3_shape({n3});
    TensorShape jj3_shape({n3});
    TensorShape vv3_shape({n3});
            
    // create output tensor
    
    Tensor* ii3 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii3_shape, &ii3));
    Tensor* jj3 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj3_shape, &jj3));
    Tensor* vv3 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv3_shape, &vv3));
    
    // get the corresponding Eigen tensors for data access
    auto ii3_tensor = ii3->flat<int64>().data();
    auto jj3_tensor = jj3->flat<int64>().data();
    auto vv3_tensor = vv3->flat<double>().data();   

    // implement your forward function here 
    // std::cout << Eigen::MatrixXd(A) << std::endl;
    // TODO:
    int p = 0;
    for (int k = 0; k < A.outerSize(); ++k){
        for (SpMat::InnerIterator it(A, k); it; ++it){
            ii3_tensor[p] = it.row()+1;
            jj3_tensor[p] = it.col()+1; 
            vv3_tensor[p] = it.value(); 
            p++;
        }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseSparseMatMul").Device(DEVICE_CPU), SparseSparseMatMulOp);


#if 0
// todo implement
REGISTER_OP("SparseMatMulGrad")
  
  .Input("grad_ii3 : int64")
.Input("grad_jj3 : int64")
.Input("grad_vv3 : double")
  .Input("ii3 : int64")
  .Input("jj3 : int64")
  .Input("vv3 : double")
  .Input("ii1 : int64")
  .Input("jj1 : int64")
  .Input("vv1 : double")
  .Input("ii2 : int64")
  .Input("jj2 : int64")
  .Input("vv2 : double")
  .Input("m : int64")
  .Input("n : int64")
  .Input("k : int64")
  .Output("grad_ii1 : int64")
  .Output("grad_jj1 : int64")
  .Output("grad_vv1 : double")
  .Output("grad_ii2 : int64")
  .Output("grad_jj2 : int64")
  .Output("grad_vv2 : double")
  .Output("grad_m : int64")
  .Output("grad_n : int64")
  .Output("grad_k : int64");
class SparseMatMulGradOp : public OpKernel {
private:
  
public:
  explicit SparseMatMulGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_ii3 = context->input(0);
    const Tensor& grad_jj3 = context->input(1);
    const Tensor& grad_vv3 = context->input(2);
    const Tensor& ii3 = context->input(3);
    const Tensor& jj3 = context->input(4);
    const Tensor& vv3 = context->input(5);
    const Tensor& ii1 = context->input(6);
    const Tensor& jj1 = context->input(7);
    const Tensor& vv1 = context->input(8);
    const Tensor& ii2 = context->input(9);
    const Tensor& jj2 = context->input(10);
    const Tensor& vv2 = context->input(11);
    const Tensor& m = context->input(12);
    const Tensor& n = context->input(13);
    const Tensor& k = context->input(14);
    
    
    const TensorShape& grad_ii3_shape = grad_ii3.shape();
    const TensorShape& grad_jj3_shape = grad_jj3.shape();
    const TensorShape& grad_vv3_shape = grad_vv3.shape();
    const TensorShape& ii3_shape = ii3.shape();
    const TensorShape& jj3_shape = jj3.shape();
    const TensorShape& vv3_shape = vv3.shape();
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& k_shape = k.shape();
    
    
    DCHECK_EQ(grad_ii3_shape.dims(), 1);
    DCHECK_EQ(grad_jj3_shape.dims(), 1);
    DCHECK_EQ(grad_vv3_shape.dims(), 1);
    DCHECK_EQ(ii3_shape.dims(), 1);
    DCHECK_EQ(jj3_shape.dims(), 1);
    DCHECK_EQ(vv3_shape.dims(), 1);
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(k_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ii1_shape(ii1_shape);
    TensorShape grad_jj1_shape(jj1_shape);
    TensorShape grad_vv1_shape(vv1_shape);
    TensorShape grad_ii2_shape(ii2_shape);
    TensorShape grad_jj2_shape(jj2_shape);
    TensorShape grad_vv2_shape(vv2_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_k_shape(k_shape);
            
    // create output tensor
    
    Tensor* grad_ii1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii1_shape, &grad_ii1));
    Tensor* grad_jj1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_jj1_shape, &grad_jj1));
    Tensor* grad_vv1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_vv1_shape, &grad_vv1));
    Tensor* grad_ii2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_ii2_shape, &grad_ii2));
    Tensor* grad_jj2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_jj2_shape, &grad_jj2));
    Tensor* grad_vv2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_vv2_shape, &grad_vv2));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_n_shape, &grad_n));
    Tensor* grad_k = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_k_shape, &grad_k));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto k_tensor = k.flat<int64>().data();
    auto grad_ii3_tensor = grad_ii3.flat<int64>().data();
    auto grad_jj3_tensor = grad_jj3.flat<int64>().data();
    auto grad_vv3_tensor = grad_vv3.flat<double>().data();
    auto ii3_tensor = ii3.flat<int64>().data();
    auto jj3_tensor = jj3.flat<int64>().data();
    auto vv3_tensor = vv3.flat<double>().data();
    auto grad_ii1_tensor = grad_ii1->flat<int64>().data();
    auto grad_jj1_tensor = grad_jj1->flat<int64>().data();
    auto grad_vv1_tensor = grad_vv1->flat<double>().data();
    auto grad_ii2_tensor = grad_ii2->flat<int64>().data();
    auto grad_jj2_tensor = grad_jj2->flat<int64>().data();
    auto grad_vv2_tensor = grad_vv2->flat<double>().data();
    auto grad_m_tensor = grad_m->flat<int64>().data();
    auto grad_n_tensor = grad_n->flat<int64>().data();
    auto grad_k_tensor = grad_k->flat<int64>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseMatMulGrad").Device(DEVICE_CPU), SparseMatMulGradOp);
#endif


REGISTER_OP("DiagSparseMatMul")

.Input("ii1 : int64")
  .Input("jj1 : int64")
  .Input("vv1 : double")
  .Input("ii2 : int64")
  .Input("jj2 : int64")
  .Input("vv2 : double")
  .Input("m : int64")
  .Input("n : int64")
  .Input("k : int64")
  .Output("ii3 : int64")
  .Output("jj3 : int64")
  .Output("vv3 : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii1_shape));
        shape_inference::ShapeHandle jj1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &jj1_shape));
        shape_inference::ShapeHandle vv1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &vv1_shape));
        shape_inference::ShapeHandle ii2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &ii2_shape));
        shape_inference::ShapeHandle jj2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &jj2_shape));
        shape_inference::ShapeHandle vv2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &vv2_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &n_shape));
        shape_inference::ShapeHandle k_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &k_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });
class DiagSparseMatMulOp : public OpKernel {
private:
  
public:
  explicit DiagSparseMatMulOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(9, context->num_inputs());
    
    
    const Tensor& ii1 = context->input(0);
    const Tensor& jj1 = context->input(1);
    const Tensor& vv1 = context->input(2);
    const Tensor& ii2 = context->input(3);
    const Tensor& jj2 = context->input(4);
    const Tensor& vv2 = context->input(5);
    const Tensor& m = context->input(6);
    const Tensor& n = context->input(7);
    const Tensor& k = context->input(8);
    
    
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& k_shape = k.shape();
    
    
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(k_shape.dims(), 0);

    int n1 = ii1_shape.dim_size(0); int n2 = ii2_shape.dim_size(0);
    // extra check
        
    // create output shape
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto k_tensor = k.flat<int64>().data();
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    SpMat A(*m_tensor, *k_tensor);
    forward_diag_sparse(ii1_tensor, jj1_tensor, vv1_tensor, n1, 
            ii2_tensor, jj2_tensor, vv2_tensor, n2, 
            *m_tensor, *n_tensor, *k_tensor, &A);
    
    int n3 = A.nonZeros();
    TensorShape ii3_shape({n3});
    TensorShape jj3_shape({n3});
    TensorShape vv3_shape({n3});
            
    // create output tensor
    
    Tensor* ii3 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii3_shape, &ii3));
    Tensor* jj3 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj3_shape, &jj3));
    Tensor* vv3 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv3_shape, &vv3));
    
    // get the corresponding Eigen tensors for data access
    auto ii3_tensor = ii3->flat<int64>().data();
    auto jj3_tensor = jj3->flat<int64>().data();
    auto vv3_tensor = vv3->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    int p = 0;
    for (int k = 0; k < A.outerSize(); ++k){
        for (SpMat::InnerIterator it(A, k); it; ++it){
            ii3_tensor[p] = it.row()+1;
            jj3_tensor[p] = it.col()+1; 
            vv3_tensor[p] = it.value(); 
            p++;
        }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("DiagSparseMatMul").Device(DEVICE_CPU), DiagSparseMatMulOp);



REGISTER_OP("SparseDiagMatMul")

.Input("ii1 : int64")
  .Input("jj1 : int64")
  .Input("vv1 : double")
  .Input("ii2 : int64")
  .Input("jj2 : int64")
  .Input("vv2 : double")
  .Input("m : int64")
  .Input("n : int64")
  .Input("k : int64")
  .Output("ii3 : int64")
  .Output("jj3 : int64")
  .Output("vv3 : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii1_shape));
        shape_inference::ShapeHandle jj1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &jj1_shape));
        shape_inference::ShapeHandle vv1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &vv1_shape));
        shape_inference::ShapeHandle ii2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &ii2_shape));
        shape_inference::ShapeHandle jj2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &jj2_shape));
        shape_inference::ShapeHandle vv2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &vv2_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &n_shape));
        shape_inference::ShapeHandle k_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &k_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });
class SparseDiagMatMulOp : public OpKernel {
private:
  
public:
  explicit SparseDiagMatMulOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(9, context->num_inputs());
    
    
    const Tensor& ii1 = context->input(0);
    const Tensor& jj1 = context->input(1);
    const Tensor& vv1 = context->input(2);
    const Tensor& ii2 = context->input(3);
    const Tensor& jj2 = context->input(4);
    const Tensor& vv2 = context->input(5);
    const Tensor& m = context->input(6);
    const Tensor& n = context->input(7);
    const Tensor& k = context->input(8);
    
    
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& k_shape = k.shape();
    
    
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(k_shape.dims(), 0);

    int n1 = ii1_shape.dim_size(0); int n2 = ii2_shape.dim_size(0);
    // extra check
        
    // create output shape
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto k_tensor = k.flat<int64>().data();
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    SpMat A(*m_tensor, *k_tensor);
    forward_sparse_diag(ii1_tensor, jj1_tensor, vv1_tensor, n1, 
            ii2_tensor, jj2_tensor, vv2_tensor, n2, 
            *m_tensor, *n_tensor, *k_tensor, &A);
    
    int n3 = A.nonZeros();
    TensorShape ii3_shape({n3});
    TensorShape jj3_shape({n3});
    TensorShape vv3_shape({n3});
            
    // create output tensor
    
    Tensor* ii3 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii3_shape, &ii3));
    Tensor* jj3 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj3_shape, &jj3));
    Tensor* vv3 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv3_shape, &vv3));
    
    // get the corresponding Eigen tensors for data access
    auto ii3_tensor = ii3->flat<int64>().data();
    auto jj3_tensor = jj3->flat<int64>().data();
    auto vv3_tensor = vv3->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    int p = 0;
    for (int k = 0; k < A.outerSize(); ++k){
        for (SpMat::InnerIterator it(A, k); it; ++it){
            ii3_tensor[p] = it.row()+1;
            jj3_tensor[p] = it.col()+1; 
            vv3_tensor[p] = it.value(); 
            p++;
        }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseDiagMatMul").Device(DEVICE_CPU), SparseDiagMatMulOp);
