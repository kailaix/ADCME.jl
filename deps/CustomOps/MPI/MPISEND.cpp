#include "common.h"

void MPISEND_forward(const double *out, int n, int dest, int tag){
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank( comm , &rank);
  MPI_Send( out , n , MPI_DOUBLE , dest, tag , comm);
}

void MPISEND_backward(
  double *grad_a, const double *grad_out,
  const double *out, int n, int dest, int tag
){
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank( comm , &rank);
  MPI_Status status;
  MPI_Recv( grad_a , n , MPI_DOUBLE , dest , tag , comm , &status);
}

REGISTER_OP("MPISEND")
.Input("a : double")
.Input("dest : int64")
.Input("tag : int64")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_shape));
        shape_inference::ShapeHandle dest_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &dest_shape));
        shape_inference::ShapeHandle tag_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &tag_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("MPISENDGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("a : double")
.Input("dest : int64")
.Input("tag : int64")
.Output("grad_a : double")
.Output("grad_dest : int64")
.Output("grad_tag : int64");

/*-------------------------------------------------------------------------------------*/

class MPISENDOp : public OpKernel {
private:
  
public:
  explicit MPISENDOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& dest = context->input(1);
    const Tensor& tag = context->input(2);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& dest_shape = dest.shape();
    const TensorShape& tag_shape = tag.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(dest_shape.dims(), 0);
    DCHECK_EQ(tag_shape.dims(), 0);

    auto a_tensor = a.flat<double>().data();
    auto dest_tensor = dest.flat<int64>().data();
    auto tag_tensor = tag.flat<int64>().data();
    int n = a_shape.dim_size(0);

    MPISEND_forward(a_tensor,n,*dest_tensor,*tag_tensor);
    context->set_output(0, context->input(0));

  }
};
REGISTER_KERNEL_BUILDER(Name("MPISEND").Device(DEVICE_CPU), MPISENDOp);



class MPISENDGradOp : public OpKernel {
private:
  
public:
  explicit MPISENDGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& dest = context->input(3);
    const Tensor& tag = context->input(4);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& dest_shape = dest.shape();
    const TensorShape& tag_shape = tag.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(dest_shape.dims(), 0);
    DCHECK_EQ(tag_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_dest_shape(dest_shape);
    TensorShape grad_tag_shape(tag_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_dest = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_dest_shape, &grad_dest));
    Tensor* grad_tag = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_tag_shape, &grad_tag));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto dest_tensor = dest.flat<int64>().data();
    auto tag_tensor = tag.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int n = a_shape.dim_size(0);
    MPISEND_backward(grad_a_tensor, grad_out_tensor, out_tensor, n, *dest_tensor, *tag_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MPISENDGrad").Device(DEVICE_CPU), MPISENDGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class MPISENDOpGPU : public OpKernel {
private:
  
public:
  explicit MPISENDOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& dest = context->input(1);
    const Tensor& tag = context->input(2);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& dest_shape = dest.shape();
    const TensorShape& tag_shape = tag.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(dest_shape.dims(), 0);
    DCHECK_EQ(tag_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto dest_tensor = dest.flat<int64>().data();
    auto tag_tensor = tag.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("MPISEND").Device(DEVICE_GPU), MPISENDOpGPU);

class MPISENDGradOpGPU : public OpKernel {
private:
  
public:
  explicit MPISENDGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& dest = context->input(3);
    const Tensor& tag = context->input(4);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& dest_shape = dest.shape();
    const TensorShape& tag_shape = tag.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(dest_shape.dims(), 0);
    DCHECK_EQ(tag_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_dest_shape(dest_shape);
    TensorShape grad_tag_shape(tag_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_dest = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_dest_shape, &grad_dest));
    Tensor* grad_tag = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_tag_shape, &grad_tag));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto dest_tensor = dest.flat<int64>().data();
    auto tag_tensor = tag.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MPISENDGrad").Device(DEVICE_GPU), MPISENDGradOpGPU);

#endif