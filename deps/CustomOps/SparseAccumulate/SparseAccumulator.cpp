#include<cmath>
#include<string> 
using std::string;
#include "SparseAccumulate.h"

using namespace tensorflow;

std::map<int, SparseAccum*> SAMAP;

REGISTER_OP("SparseAccumulator")
.Input("tol : double")
.Input("nrow : int32")
.Input("h : int32")
.Output("handle : int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle tol_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &tol_shape));
        shape_inference::ShapeHandle nrow_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &nrow_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &h_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });

class SparseAccumulatorOp : public OpKernel {
private:
  
public:
  explicit SparseAccumulatorOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& tol = context->input(0);
    const Tensor& nrow = context->input(1);
    const Tensor& h = context->input(2);
    
    
    const TensorShape& tol_shape = tol.shape();
    const TensorShape& nrow_shape = nrow.shape();
    
    
    DCHECK_EQ(tol_shape.dims(), 0);
    DCHECK_EQ(nrow_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape handle_shape({});
            
    // create output tensor
    
    Tensor* handle = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, handle_shape, &handle));
    
    // get the corresponding Eigen tensors for data access
    
    auto tol_tensor = tol.flat<double>().data();
    auto nrow_tensor = nrow.flat<int32>().data();
    auto h_tensor = h.flat<int32>().data();
    auto handle_tensor = handle->flat<int32>().data();   

    // implement your forward function here 

    // TODO:
    *handle_tensor = create_sparse_assembler(SAMAP, *h_tensor, *nrow_tensor, *tol_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseAccumulator").Device(DEVICE_CPU), SparseAccumulatorOp);



REGISTER_OP("SparseAccumulatorAdd")

.Input("handle : int32")
  .Input("nrow : int32")
  .Input("cols : int32")
  .Input("vals : double")
  .Output("op : int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle handle_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle_shape));
        shape_inference::ShapeHandle nrow_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &nrow_shape));
        shape_inference::ShapeHandle cols_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &cols_shape));
        shape_inference::ShapeHandle vals_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &vals_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });
class SparseAccumulatorAddOp : public OpKernel {
private:
  
public:
  explicit SparseAccumulatorAddOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& handle = context->input(0);
    const Tensor& nrow = context->input(1);
    const Tensor& cols = context->input(2);
    const Tensor& vals = context->input(3);
    
    
    const TensorShape& handle_shape = handle.shape();
    const TensorShape& nrow_shape = nrow.shape();
    const TensorShape& cols_shape = cols.shape();
    const TensorShape& vals_shape = vals.shape();
    
    
    DCHECK_EQ(handle_shape.dims(), 0);
    DCHECK_EQ(nrow_shape.dims(), 0);
    DCHECK_EQ(cols_shape.dims(), 1);
    DCHECK_EQ(vals_shape.dims(), 1);

    // extra check
        
    // create output shape
    int n = cols_shape.dim_size(0);
    DCHECK_EQ(vals_shape.dim_size(0), n);
    TensorShape op_shape({});
            
    // create output tensor
    
    Tensor* op = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, op_shape, &op));
    
    // get the corresponding Eigen tensors for data access
    
    auto handle_tensor = handle.flat<int32>().data();
    auto nrow_tensor = nrow.flat<int32>().data();
    auto cols_tensor = cols.flat<int32>().data();
    auto vals_tensor = vals.flat<double>().data();
    auto op_tensor = op->flat<int32>().data();   

    // implement your forward function here 

    // TODO:
    accumulate_sparse_assembler(SAMAP, *handle_tensor, *nrow_tensor, cols_tensor, vals_tensor, n);
    *op_tensor = *handle_tensor;

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseAccumulatorAdd").Device(DEVICE_CPU), SparseAccumulatorAddOp);



REGISTER_OP("SparseAccumulatorCopy")

.Input("handle : int32")
  .Output("rows : int32")
  .Output("cols : int32")
  .Output("vals : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle handle_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });

class SparseAccumulatorCopyOp : public OpKernel {
private:
  
public:
  explicit SparseAccumulatorCopyOp(OpKernelConstruction* context) : OpKernel(context) {
  }
  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& handle = context->input(0);
    
    
    const TensorShape& handle_shape = handle.shape();
    
    
    DCHECK_EQ(handle_shape.dims(), 1);

    // extra check
        
    // create output shape
    auto handle_tensor = handle.flat<int32>().data();
    int n = SAMAP[*handle_tensor]->get_n();
    TensorShape rows_shape({n});
    TensorShape cols_shape({n});
    TensorShape vals_shape({n});
            
    // create output tensor
    
    Tensor* rows = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, rows_shape, &rows));
    Tensor* cols = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, cols_shape, &cols));
    Tensor* vals = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vals_shape, &vals));
    
    // get the corresponding Eigen tensors for data access
    
    
    auto rows_tensor = rows->flat<int32>().data();
    auto cols_tensor = cols->flat<int32>().data();
    auto vals_tensor = vals->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    copy_sparse_assemlber(SAMAP, *handle_tensor, rows_tensor, cols_tensor, vals_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("SparseAccumulatorCopy").Device(DEVICE_CPU), SparseAccumulatorCopyOp);

