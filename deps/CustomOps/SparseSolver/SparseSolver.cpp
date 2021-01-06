#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
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
#include "SparseSolver.h"


REGISTER_OP("SparseSolver")

.Input("ii : int64")
.Input("jj : int64")
.Input("vv : double")
.Input("f : double")
.Input("method : string")
.Output("u : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii_shape));
        shape_inference::ShapeHandle jj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &jj_shape));
        shape_inference::ShapeHandle vv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &vv_shape));
        shape_inference::ShapeHandle f_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &f_shape));
        shape_inference::ShapeHandle method_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &method_shape));

        c->set_output(0, c->input(3));
    return Status::OK();
  });

REGISTER_OP("SparseSolverGrad")

.Input("grad_u : double")
.Input("u : double")
.Input("ii : int64")
.Input("jj : int64")
.Input("vv : double")
.Input("f : double")
.Input("method : string")
.Output("grad_ii : int64")
.Output("grad_jj : int64")
.Output("grad_vv : double")
.Output("grad_f : double")
.Output("grad_method : string");


class SparseSolverOp : public OpKernel {
private:
  
public:
  explicit SparseSolverOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(5, context->num_inputs());
    
    
    const Tensor& ii = context->input(0);
    const Tensor& jj = context->input(1);
    const Tensor& vv = context->input(2);
    const Tensor& f = context->input(3);
    const Tensor& method = context->input(4);
    
    
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& f_shape = f.shape();
    
    
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    int nv = ii_shape.dim_size(0), d = f_shape.dim_size(0);
    DCHECK_EQ(jj_shape.dim_size(0), nv);
    DCHECK_EQ(vv_shape.dim_size(0), nv);
    TensorShape u_shape({d});
            
    // create output tensor
    
    Tensor* u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, u_shape, &u));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto f_tensor = f.flat<double>().data();
    tstring method_tensor = *method.flat<tstring>().data();
    auto u_tensor = u->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    bool flag;
    if (method_tensor==tstring("SparseLU")){
      flag = forward<Eigen::SparseLU<SpMat>>(u_tensor, ii_tensor, jj_tensor, vv_tensor, nv, f_tensor, d);
    }
    else if (method_tensor==tstring("SparseQR")){
      flag = forward<Eigen::SparseQR<SpMat,Eigen::COLAMDOrdering<int>>>(u_tensor, ii_tensor, jj_tensor, vv_tensor, nv, f_tensor, d);
    }
    else if (method_tensor==tstring("SimplicialLDLT")){
      flag = forward<Eigen::SimplicialLDLT<SpMat>>(u_tensor, ii_tensor, jj_tensor, vv_tensor, nv, f_tensor, d);
    }
    else if (method_tensor==tstring("SimplicialLLT")){
      flag = forward<Eigen::SimplicialLLT<SpMat>>(u_tensor, ii_tensor, jj_tensor, vv_tensor, nv, f_tensor, d);
    }
    else{
      OP_REQUIRES_OK(context, 
        Status(error::Code::UNAVAILABLE, "Sparse solver type not supported."));
    }
    if (!flag){
      OP_REQUIRES_OK(context, 
        Status(error::Code::INTERNAL, "Sparse solver factorization failed."));
    }

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseSolver").Device(DEVICE_CPU), SparseSolverOp);



class SparseSolverGradOp : public OpKernel {
private:
  
public:
  explicit SparseSolverGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_u = context->input(0);
    const Tensor& u = context->input(1);
    const Tensor& ii = context->input(2);
    const Tensor& jj = context->input(3);
    const Tensor& vv = context->input(4);
    const Tensor& f = context->input(5);
    const Tensor& method = context->input(6);
    
    
    const TensorShape& grad_u_shape = grad_u.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& f_shape = f.shape();
    const TensorShape& method_shape = method.shape();
    
    
    DCHECK_EQ(grad_u_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 1);
    DCHECK_EQ(method_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_jj_shape(jj_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_f_shape(f_shape);
    TensorShape grad_method_shape(method_shape);
            
    // create output tensor
    
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii_shape, &grad_ii));
    Tensor* grad_jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_jj_shape, &grad_jj));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_vv_shape, &grad_vv));
    Tensor* grad_f = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_f_shape, &grad_f));
    Tensor* grad_method = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_method_shape, &grad_method));
    
    // get the corresponding Eigen tensors for data access
    
    auto vv_tensor = vv.flat<double>().data();
    auto f_tensor = f.flat<double>().data();
    auto grad_u_tensor = grad_u.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto grad_ii_tensor = grad_ii->flat<int64>().data();
    auto grad_jj_tensor = grad_jj->flat<int64>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();
    auto grad_f_tensor = grad_f->flat<double>().data();
    tstring method_tensor = *method.flat<tstring>().data();   

    // implement your backward function here 
    int nv = ii_shape.dim_size(0), d = f_shape.dim_size(0);
    // TODO:
    if (method_tensor == tstring("SparseLU")){
      backward<Eigen::SparseLU<SpMat>>(grad_f_tensor, grad_vv_tensor, grad_u_tensor, 
                              u_tensor, ii_tensor, jj_tensor, vv_tensor, nv, f_tensor, d);
    }
    else if (method_tensor == tstring("SparseQR")){
      backward<Eigen::SparseQR<SpMat,Eigen::COLAMDOrdering<int>>>(grad_f_tensor, grad_vv_tensor, grad_u_tensor, 
                              u_tensor, ii_tensor, jj_tensor, vv_tensor, nv, f_tensor, d);
    }
    else if (method_tensor == tstring("SimplicialLDLT")){
      backward<Eigen::SimplicialLDLT<SpMat>>(grad_f_tensor, grad_vv_tensor, grad_u_tensor, 
                              u_tensor, ii_tensor, jj_tensor, vv_tensor, nv, f_tensor, d);
    }
    else if (method_tensor == tstring("SimplicialLLT")){
      backward<Eigen::SimplicialLLT<SpMat>>(grad_f_tensor, grad_vv_tensor, grad_u_tensor, 
                              u_tensor, ii_tensor, jj_tensor, vv_tensor, nv, f_tensor, d);
    }

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseSolverGrad").Device(DEVICE_CPU), SparseSolverGradOp);
