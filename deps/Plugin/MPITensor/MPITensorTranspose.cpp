#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "MPITensorTranspose.h"


REGISTER_OP("MPITensorTranspose")
.Input("row : int32")
.Input("col : int32")
.Input("ncol : int32")
.Input("val : double")
.Input("n : int64")
.Input("rank : int64")
.Input("nt : int64")
.Output("indices : int64")
.Output("ovals : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle row_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &row_shape));
        shape_inference::ShapeHandle col_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &col_shape));
        shape_inference::ShapeHandle ncol_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &ncol_shape));
        shape_inference::ShapeHandle val_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &val_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &n_shape));
        shape_inference::ShapeHandle rank_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &rank_shape));
        shape_inference::ShapeHandle nt_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &nt_shape));

        c->set_output(0, c->Matrix(-1,2));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("MPITensorTransposeGrad")
.Input("grad_ovals : double")
.Input("indices : int64")
.Input("ovals : double")
.Input("row : int32")
.Input("col : int32")
.Input("ncol : int32")
.Input("val : double")
.Input("n : int64")
.Input("rank : int64")
.Input("nt : int64")
.Output("grad_row : int32")
.Output("grad_col : int32")
.Output("grad_ncol : int32")
.Output("grad_val : double")
.Output("grad_n : int64")
.Output("grad_rank : int64")
.Output("grad_nt : int64");

/*-------------------------------------------------------------------------------------*/

class MPITensorTransposeOp : public OpKernel {
private:
  
public:
  explicit MPITensorTransposeOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(7, context->num_inputs());
    
    
    const Tensor& row = context->input(0);
    const Tensor& col = context->input(1);
    const Tensor& ncol = context->input(2);
    const Tensor& val = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& rank = context->input(5);
    const Tensor& nt = context->input(6);
    
    
    const TensorShape& row_shape = row.shape();
    const TensorShape& col_shape = col.shape();
    const TensorShape& ncol_shape = ncol.shape();
    const TensorShape& val_shape = val.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& rank_shape = rank.shape();
    const TensorShape& nt_shape = nt.shape();
    
    
    DCHECK_EQ(row_shape.dims(), 1);
    DCHECK_EQ(col_shape.dims(), 1);
    DCHECK_EQ(ncol_shape.dims(), 1);
    DCHECK_EQ(val_shape.dims(), 1);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(rank_shape.dims(), 0);
    DCHECK_EQ(nt_shape.dims(), 0);

    // extra check
        
    // create output shape
    auto row_tensor = row.flat<int32>().data();
    auto col_tensor = col.flat<int32>().data();
    auto ncol_tensor = ncol.flat<int32>().data();
    auto val_tensor = val.flat<double>().data();
    auto n_tensor = n.flat<int64>().data();
    auto rank_tensor = rank.flat<int64>().data();
    auto nt_tensor = nt.flat<int64>().data();

    int nrows = *n_tensor;
    int mat_size = *nt_tensor;
    int rank_ = *rank_tensor;
    int n_row = row_shape.dim_size(0);
    MPITensorTranspose::Forward fwd(nrows, mat_size, rank_,
          row_tensor, n_row, col_tensor, ncol_tensor, val_tensor);
    int k = fwd.size();
    TensorShape indices_shape({k,2});
    TensorShape ovals_shape({k});
            
    // create output tensor
    
    Tensor* indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, indices_shape, &indices));
    Tensor* ovals = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, ovals_shape, &ovals));
    
    // get the corresponding Eigen tensors for data access
    
    
    auto indices_tensor = indices->flat<int64>().data();
    auto ovals_tensor = ovals->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    fwd.copy(indices_tensor, ovals_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("MPITensorTranspose").Device(DEVICE_CPU), MPITensorTransposeOp);



class MPITensorTransposeGradOp : public OpKernel {
private:
  
public:
  explicit MPITensorTransposeGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_ovals = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& ovals = context->input(2);
    const Tensor& row = context->input(3);
    const Tensor& col = context->input(4);
    const Tensor& ncol = context->input(5);
    const Tensor& val = context->input(6);
    const Tensor& n = context->input(7);
    const Tensor& rank = context->input(8);
    const Tensor& nt = context->input(9);
    
    
    const TensorShape& grad_ovals_shape = grad_ovals.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& ovals_shape = ovals.shape();
    const TensorShape& row_shape = row.shape();
    const TensorShape& col_shape = col.shape();
    const TensorShape& ncol_shape = ncol.shape();
    const TensorShape& val_shape = val.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& rank_shape = rank.shape();
    const TensorShape& nt_shape = nt.shape();
    
    
    DCHECK_EQ(grad_ovals_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(ovals_shape.dims(), 1);
    DCHECK_EQ(row_shape.dims(), 1);
    DCHECK_EQ(col_shape.dims(), 1);
    DCHECK_EQ(ncol_shape.dims(), 1);
    DCHECK_EQ(val_shape.dims(), 1);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(rank_shape.dims(), 0);
    DCHECK_EQ(nt_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_row_shape(row_shape);
    TensorShape grad_col_shape(col_shape);
    TensorShape grad_ncol_shape(ncol_shape);
    TensorShape grad_val_shape(val_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_rank_shape(rank_shape);
    TensorShape grad_nt_shape(nt_shape);
            
    // create output tensor
    
    Tensor* grad_row = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_row_shape, &grad_row));
    Tensor* grad_col = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_col_shape, &grad_col));
    Tensor* grad_ncol = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_ncol_shape, &grad_ncol));
    Tensor* grad_val = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_val_shape, &grad_val));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_n_shape, &grad_n));
    Tensor* grad_rank = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_rank_shape, &grad_rank));
    Tensor* grad_nt = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_nt_shape, &grad_nt));
    
    // get the corresponding Eigen tensors for data access
    
    auto row_tensor = row.flat<int32>().data();
    auto col_tensor = col.flat<int32>().data();
    auto ncol_tensor = ncol.flat<int32>().data();
    auto val_tensor = val.flat<double>().data();
    auto n_tensor = n.flat<int64>().data();
    auto rank_tensor = rank.flat<int64>().data();
    auto nt_tensor = nt.flat<int64>().data();
    auto grad_ovals_tensor = grad_ovals.flat<double>().data();
    auto indices_tensor = indices.flat<int64>().data();
    auto ovals_tensor = ovals.flat<double>().data();
    auto grad_val_tensor = grad_val->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MPITensorTransposeGrad").Device(DEVICE_CPU), MPITensorTransposeGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class MPITensorTransposeOpGPU : public OpKernel {
private:
  
public:
  explicit MPITensorTransposeOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(7, context->num_inputs());
    
    
    const Tensor& row = context->input(0);
    const Tensor& col = context->input(1);
    const Tensor& ncol = context->input(2);
    const Tensor& val = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& rank = context->input(5);
    const Tensor& nt = context->input(6);
    
    
    const TensorShape& row_shape = row.shape();
    const TensorShape& col_shape = col.shape();
    const TensorShape& ncol_shape = ncol.shape();
    const TensorShape& val_shape = val.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& rank_shape = rank.shape();
    const TensorShape& nt_shape = nt.shape();
    
    
    DCHECK_EQ(row_shape.dims(), 1);
    DCHECK_EQ(col_shape.dims(), 1);
    DCHECK_EQ(ncol_shape.dims(), 1);
    DCHECK_EQ(val_shape.dims(), 1);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(rank_shape.dims(), 0);
    DCHECK_EQ(nt_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape indices_shape({-1,2});
    TensorShape ovals_shape({-1});
            
    // create output tensor
    
    Tensor* indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, indices_shape, &indices));
    Tensor* ovals = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, ovals_shape, &ovals));
    
    // get the corresponding Eigen tensors for data access
    
    auto row_tensor = row.flat<int32>().data();
    auto col_tensor = col.flat<int32>().data();
    auto ncol_tensor = ncol.flat<int32>().data();
    auto val_tensor = val.flat<double>().data();
    auto n_tensor = n.flat<int64>().data();
    auto rank_tensor = rank.flat<int64>().data();
    auto nt_tensor = nt.flat<int64>().data();
    auto indices_tensor = indices->flat<int64>().data();
    auto ovals_tensor = ovals->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("MPITensorTranspose").Device(DEVICE_GPU), MPITensorTransposeOpGPU);

class MPITensorTransposeGradOpGPU : public OpKernel {
private:
  
public:
  explicit MPITensorTransposeGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_ovals = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& ovals = context->input(2);
    const Tensor& row = context->input(3);
    const Tensor& col = context->input(4);
    const Tensor& ncol = context->input(5);
    const Tensor& val = context->input(6);
    const Tensor& n = context->input(7);
    const Tensor& rank = context->input(8);
    const Tensor& nt = context->input(9);
    
    
    const TensorShape& grad_ovals_shape = grad_ovals.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& ovals_shape = ovals.shape();
    const TensorShape& row_shape = row.shape();
    const TensorShape& col_shape = col.shape();
    const TensorShape& ncol_shape = ncol.shape();
    const TensorShape& val_shape = val.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& rank_shape = rank.shape();
    const TensorShape& nt_shape = nt.shape();
    
    
    DCHECK_EQ(grad_ovals_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(ovals_shape.dims(), 1);
    DCHECK_EQ(row_shape.dims(), 1);
    DCHECK_EQ(col_shape.dims(), 1);
    DCHECK_EQ(ncol_shape.dims(), 1);
    DCHECK_EQ(val_shape.dims(), 1);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(rank_shape.dims(), 0);
    DCHECK_EQ(nt_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_row_shape(row_shape);
    TensorShape grad_col_shape(col_shape);
    TensorShape grad_ncol_shape(ncol_shape);
    TensorShape grad_val_shape(val_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_rank_shape(rank_shape);
    TensorShape grad_nt_shape(nt_shape);
            
    // create output tensor
    
    Tensor* grad_row = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_row_shape, &grad_row));
    Tensor* grad_col = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_col_shape, &grad_col));
    Tensor* grad_ncol = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_ncol_shape, &grad_ncol));
    Tensor* grad_val = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_val_shape, &grad_val));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_n_shape, &grad_n));
    Tensor* grad_rank = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_rank_shape, &grad_rank));
    Tensor* grad_nt = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_nt_shape, &grad_nt));
    
    // get the corresponding Eigen tensors for data access
    
    auto row_tensor = row.flat<int32>().data();
    auto col_tensor = col.flat<int32>().data();
    auto ncol_tensor = ncol.flat<int32>().data();
    auto val_tensor = val.flat<double>().data();
    auto n_tensor = n.flat<int64>().data();
    auto rank_tensor = rank.flat<int64>().data();
    auto nt_tensor = nt.flat<int64>().data();
    auto grad_ovals_tensor = grad_ovals.flat<double>().data();
    auto indices_tensor = indices.flat<int64>().data();
    auto ovals_tensor = ovals.flat<double>().data();
    auto grad_val_tensor = grad_val->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MPITensorTransposeGrad").Device(DEVICE_GPU), MPITensorTransposeGradOpGPU);

#endif