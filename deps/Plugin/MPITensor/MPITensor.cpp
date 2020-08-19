#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include "HYPRE_parcsr_ls.h"
#include "HYPRE.h"

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "MPITensor.h"

double MPITensor_Solve_Timer;
int MPITensor_Solve_Count;
extern "C" void MPITensor_Solve_Timer_SetZero(){
  MPITensor_Solve_Timer = 0.0;
  MPITensor_Solve_Count = 0;
}
extern "C" double MPITensor_Solve_Timer_Get(){
  return MPITensor_Solve_Timer/MPITensor_Solve_Count;
}

REGISTER_OP("MPICreateMatrix")
.Input("indices : int64")
.Input("values : double")
.Input("ilower : int64")
.Input("iupper : int64")
.Output("rows : int32")
.Output("ncols : int32")
.Output("cols : int32")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle indices_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices_shape));
        shape_inference::ShapeHandle values_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &values_shape));
        shape_inference::ShapeHandle ilower_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &ilower_shape));
        shape_inference::ShapeHandle iupper_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &iupper_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
        c->set_output(3, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("MPICreateMatrixGrad")
.Input("grad_out : double")
.Input("rows : int32")
.Input("ncols : int32")
.Input("cols : int32")
.Input("out : double")
.Input("indices : int64")
.Input("values : double")
.Input("ilower : int64")
.Input("iupper : int64")
.Output("grad_indices : int64")
.Output("grad_values : double")
.Output("grad_ilower : int64")
.Output("grad_iupper : int64");

/*-------------------------------------------------------------------------------------*/

class MPICreateMatrixOp : public OpKernel {
private:
  
public:
  explicit MPICreateMatrixOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& indices = context->input(0);
    const Tensor& values = context->input(1);
    const Tensor& ilower = context->input(2);
    const Tensor& iupper = context->input(3);
    
    
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& values_shape = values.shape();
    const TensorShape& ilower_shape = ilower.shape();
    const TensorShape& iupper_shape = iupper.shape();
    
    
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(values_shape.dims(), 1);
    DCHECK_EQ(ilower_shape.dims(), 0);
    DCHECK_EQ(iupper_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n = indices_shape.dim_size(0);
    int nrow = MPICreateMatrix_GetNumberOfRows(indices.flat<int64>().data(), n);
    
    TensorShape rows_shape({nrow});
    TensorShape ncols_shape({nrow});
    TensorShape cols_shape({n});
    TensorShape out_shape({n});
            
    // create output tensor
    
    Tensor* rows = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, rows_shape, &rows));
    Tensor* ncols = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, ncols_shape, &ncols));
    Tensor* cols = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, cols_shape, &cols));
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto indices_tensor = indices.flat<int64>().data();
    auto values_tensor = values.flat<double>().data();
    auto ilower_tensor = ilower.flat<int64>().data();
    auto iupper_tensor = iupper.flat<int64>().data();
    auto rows_tensor = rows->flat<int32>().data();
    auto ncols_tensor = ncols->flat<int32>().data();
    auto cols_tensor = cols->flat<int32>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MPICreateMatrix_forward(
      rows_tensor, ncols_tensor, cols_tensor, indices_tensor, n, *ilower_tensor, *iupper_tensor
    );
    for(int i = 0; i<n; i++) out_tensor[i] = values_tensor[i];
  }
};
REGISTER_KERNEL_BUILDER(Name("MPICreateMatrix").Device(DEVICE_CPU), MPICreateMatrixOp);



class MPICreateMatrixGradOp : public OpKernel {
private:
  
public:
  explicit MPICreateMatrixGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& rows = context->input(1);
    const Tensor& ncols = context->input(2);
    const Tensor& cols = context->input(3);
    const Tensor& out = context->input(4);
    const Tensor& indices = context->input(5);
    const Tensor& values = context->input(6);
    const Tensor& ilower = context->input(7);
    const Tensor& iupper = context->input(8);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& rows_shape = rows.shape();
    const TensorShape& ncols_shape = ncols.shape();
    const TensorShape& cols_shape = cols.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& values_shape = values.shape();
    const TensorShape& ilower_shape = ilower.shape();
    const TensorShape& iupper_shape = iupper.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(rows_shape.dims(), 1);
    DCHECK_EQ(ncols_shape.dims(), 1);
    DCHECK_EQ(cols_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(values_shape.dims(), 1);
    DCHECK_EQ(ilower_shape.dims(), 0);
    DCHECK_EQ(iupper_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_indices_shape(indices_shape);
    TensorShape grad_values_shape(values_shape);
    TensorShape grad_ilower_shape(ilower_shape);
    TensorShape grad_iupper_shape(iupper_shape);
            
    // create output tensor
    
    Tensor* grad_indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_indices_shape, &grad_indices));
    Tensor* grad_values = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_values_shape, &grad_values));
    Tensor* grad_ilower = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_ilower_shape, &grad_ilower));
    Tensor* grad_iupper = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_iupper_shape, &grad_iupper));
    
    // get the corresponding Eigen tensors for data access
    
    auto indices_tensor = indices.flat<int64>().data();
    auto values_tensor = values.flat<double>().data();
    auto ilower_tensor = ilower.flat<int64>().data();
    auto iupper_tensor = iupper.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto rows_tensor = rows.flat<int32>().data();
    auto ncols_tensor = ncols.flat<int32>().data();
    auto cols_tensor = cols.flat<int32>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_values_tensor = grad_values->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    for(int i = 0; i<values_shape.dim_size(0); i++)
      grad_values_tensor[i] = grad_out_tensor[i];
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MPICreateMatrixGrad").Device(DEVICE_CPU), MPICreateMatrixGradOp);


/**************************************************************************************************************/



REGISTER_OP("MPIGetMatrix")
.Input("rows : int32")
.Input("ncols : int32")
.Input("cols : int32")
.Input("ilower : int64")
.Input("iupper : int64")
.Input("values : double")
.Output("indices : int64")
.Output("vv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle rows_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &rows_shape));
        shape_inference::ShapeHandle ncols_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &ncols_shape));
        shape_inference::ShapeHandle cols_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &cols_shape));
        shape_inference::ShapeHandle ilower_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &ilower_shape));
        shape_inference::ShapeHandle iupper_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &iupper_shape));
        shape_inference::ShapeHandle values_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &values_shape));

        c->set_output(0, c->Matrix(-1,2));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("MPIGetMatrixGrad")
.Input("grad_vv : double")
.Input("indices : int64")
.Input("vv : double")
.Input("rows : int32")
.Input("ncols : int32")
.Input("cols : int32")
.Input("ilower : int64")
.Input("iupper : int64")
.Input("values : double")
.Output("grad_rows : int32")
.Output("grad_ncols : int32")
.Output("grad_cols : int32")
.Output("grad_ilower : int64")
.Output("grad_iupper : int64")
.Output("grad_values : double");

/*-------------------------------------------------------------------------------------*/

class MPIGetMatrixOp : public OpKernel {
private:
  
public:
  explicit MPIGetMatrixOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& rows = context->input(0);
    const Tensor& ncols = context->input(1);
    const Tensor& cols = context->input(2);
    const Tensor& ilower = context->input(3);
    const Tensor& iupper = context->input(4);
    const Tensor& values = context->input(5);
    
    
    const TensorShape& rows_shape = rows.shape();
    const TensorShape& ncols_shape = ncols.shape();
    const TensorShape& cols_shape = cols.shape();
    const TensorShape& ilower_shape = ilower.shape();
    const TensorShape& iupper_shape = iupper.shape();
    const TensorShape& values_shape = values.shape();
    
    
    DCHECK_EQ(rows_shape.dims(), 1);
    DCHECK_EQ(ncols_shape.dims(), 1);
    DCHECK_EQ(cols_shape.dims(), 1);
    DCHECK_EQ(ilower_shape.dims(), 0);
    DCHECK_EQ(iupper_shape.dims(), 0);
    DCHECK_EQ(values_shape.dims(), 1);

    // extra check
        
    // create output shape
    int n = values_shape.dim_size(0);
    TensorShape indices_shape({n,2});
    TensorShape vv_shape({n});
            
    // create output tensor
    
    Tensor* indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, indices_shape, &indices));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto rows_tensor = rows.flat<int32>().data();
    auto ncols_tensor = ncols.flat<int32>().data();
    auto cols_tensor = cols.flat<int32>().data();
    auto ilower_tensor = ilower.flat<int64>().data();
    auto iupper_tensor = iupper.flat<int64>().data();
    auto values_tensor = values.flat<double>().data();
    auto indices_tensor = indices->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    int nrows = rows_shape.dim_size(0);
    MPIGetMatrix_forward(
      indices_tensor, vv_tensor, 
      rows_tensor, ncols_tensor, cols_tensor, values_tensor, n, nrows, 
      *ilower_tensor, *iupper_tensor
    );

  }
};
REGISTER_KERNEL_BUILDER(Name("MPIGetMatrix").Device(DEVICE_CPU), MPIGetMatrixOp);



class MPIGetMatrixGradOp : public OpKernel {
private:
  
public:
  explicit MPIGetMatrixGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& vv = context->input(2);
    const Tensor& rows = context->input(3);
    const Tensor& ncols = context->input(4);
    const Tensor& cols = context->input(5);
    const Tensor& ilower = context->input(6);
    const Tensor& iupper = context->input(7);
    const Tensor& values = context->input(8);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& rows_shape = rows.shape();
    const TensorShape& ncols_shape = ncols.shape();
    const TensorShape& cols_shape = cols.shape();
    const TensorShape& ilower_shape = ilower.shape();
    const TensorShape& iupper_shape = iupper.shape();
    const TensorShape& values_shape = values.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(rows_shape.dims(), 1);
    DCHECK_EQ(ncols_shape.dims(), 1);
    DCHECK_EQ(cols_shape.dims(), 1);
    DCHECK_EQ(ilower_shape.dims(), 0);
    DCHECK_EQ(iupper_shape.dims(), 0);
    DCHECK_EQ(values_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_rows_shape(rows_shape);
    TensorShape grad_ncols_shape(ncols_shape);
    TensorShape grad_cols_shape(cols_shape);
    TensorShape grad_ilower_shape(ilower_shape);
    TensorShape grad_iupper_shape(iupper_shape);
    TensorShape grad_values_shape(values_shape);
            
    // create output tensor
    
    Tensor* grad_rows = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_rows_shape, &grad_rows));
    Tensor* grad_ncols = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_ncols_shape, &grad_ncols));
    Tensor* grad_cols = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_cols_shape, &grad_cols));
    Tensor* grad_ilower = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_ilower_shape, &grad_ilower));
    Tensor* grad_iupper = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_iupper_shape, &grad_iupper));
    Tensor* grad_values = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_values_shape, &grad_values));
    
    // get the corresponding Eigen tensors for data access
    
    auto rows_tensor = rows.flat<int32>().data();
    auto ncols_tensor = ncols.flat<int32>().data();
    auto cols_tensor = cols.flat<int32>().data();
    auto ilower_tensor = ilower.flat<int64>().data();
    auto iupper_tensor = iupper.flat<int64>().data();
    auto values_tensor = values.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto indices_tensor = indices.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_values_tensor = grad_values->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    for (int i = 0; i < values_shape.dim_size(0); i++)
      grad_values_tensor[i] = grad_vv_tensor[i];
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MPIGetMatrixGrad").Device(DEVICE_CPU), MPIGetMatrixGradOp);

/****************************************************************************************************/

REGISTER_OP("MPITensorSolve")
.Input("rows : int32")
.Input("ncols : int32")
.Input("cols : int32")
.Input("values : double")
.Input("rhs : double")
.Input("ilower : int64")
.Input("iupper : int64")
.Input("solver : string")
.Input("printlevel : int64")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle rows_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &rows_shape));
        shape_inference::ShapeHandle ncols_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &ncols_shape));
        shape_inference::ShapeHandle cols_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &cols_shape));
        shape_inference::ShapeHandle values_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &values_shape));
        shape_inference::ShapeHandle rhs_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &rhs_shape));
        shape_inference::ShapeHandle ilower_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &ilower_shape));
        shape_inference::ShapeHandle iupper_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &iupper_shape));
        shape_inference::ShapeHandle solver_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &solver_shape));
        shape_inference::ShapeHandle printlevel_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &printlevel_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("MPITensorSolveGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("rows : int32")
.Input("ncols : int32")
.Input("cols : int32")
.Input("values : double")
.Input("rhs : double")
.Input("ilower : int64")
.Input("iupper : int64")
.Input("solver : string")
.Input("printlevel : int64")
.Output("grad_rows : int32")
.Output("grad_ncols : int32")
.Output("grad_cols : int32")
.Output("grad_values : double")
.Output("grad_rhs : double")
.Output("grad_ilower : int64")
.Output("grad_iupper : int64")
.Output("grad_solver : string")
.Output("grad_printlevel : int64");

/*-------------------------------------------------------------------------------------*/

class MPITensorSolveOp : public OpKernel {
private:
  
public:
  explicit MPITensorSolveOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(9, context->num_inputs());
    
    
    const Tensor& rows = context->input(0);
    const Tensor& ncols = context->input(1);
    const Tensor& cols = context->input(2);
    const Tensor& values = context->input(3);
    const Tensor& rhs = context->input(4);
    const Tensor& ilower = context->input(5);
    const Tensor& iupper = context->input(6);
    const Tensor& solver = context->input(7);
    const Tensor& printlevel = context->input(8);
    
    
    const TensorShape& rows_shape = rows.shape();
    const TensorShape& ncols_shape = ncols.shape();
    const TensorShape& cols_shape = cols.shape();
    const TensorShape& values_shape = values.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& ilower_shape = ilower.shape();
    const TensorShape& iupper_shape = iupper.shape();
    const TensorShape& printlevel_shape = printlevel.shape();
    
    
    DCHECK_EQ(rows_shape.dims(), 1);
    DCHECK_EQ(ncols_shape.dims(), 1);
    DCHECK_EQ(cols_shape.dims(), 1);
    DCHECK_EQ(values_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(ilower_shape.dims(), 0);
    DCHECK_EQ(iupper_shape.dims(), 0);
    DCHECK_EQ(printlevel_shape.dims(), 0);

    // extra check
        
    // create output shape
    int nrows = rows_shape.dim_size(0);
    int nnz = values_shape.dim_size(0);
    int N = *iupper.flat<int64>().data() - *ilower.flat<int64>().data() + 1;
    TensorShape out_shape({N});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto rows_tensor = rows.flat<int32>().data();
    auto ncols_tensor = ncols.flat<int32>().data();
    auto cols_tensor = cols.flat<int32>().data();
    auto values_tensor = values.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto ilower_tensor = ilower.flat<int64>().data();
    auto iupper_tensor = iupper.flat<int64>().data();
    string solver_tensor = string(*solver.flat<string>().data());
    auto printlevel_tensor = printlevel.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    double t0 = MPI_Wtime();
    MPITensorSolve_forward(
      out_tensor, 
      rows_tensor, ncols_tensor, cols_tensor, values_tensor, 
      rhs_tensor, nrows, nnz, *ilower_tensor, *iupper_tensor,
        *printlevel_tensor, solver_tensor);
    t0 = MPI_Wtime() - t0;
    MPITensor_Solve_Timer += t0;
    MPITensor_Solve_Count ++;
  }
};
REGISTER_KERNEL_BUILDER(Name("MPITensorSolve").Device(DEVICE_CPU), MPITensorSolveOp);

