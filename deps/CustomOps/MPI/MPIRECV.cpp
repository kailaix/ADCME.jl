#include "common.h"

void MPIRECV_forward(double *out, int n, int src, int tag){
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Status status;
  MPI_Recv( out , n , MPI_DOUBLE, src, tag , comm, &status);
}

void MPIRECV_backward(
  const double *grad_out,
  const double *out, int n, int src, int tag
){
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank( comm , &rank);
  MPI_Send( grad_out , n , MPI_DOUBLE , src , tag , comm);
}

REGISTER_OP("MPIRECV")
.Input("a : double")
.Input("src : int64")
.Input("tag : int64")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_shape));
        shape_inference::ShapeHandle src_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &src_shape));
        shape_inference::ShapeHandle tag_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &tag_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("MPIRECVGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("a : double")
.Input("src : int64")
.Input("tag : int64")
.Output("grad_a : double")
.Output("grad_src : int64")
.Output("grad_tag : int64");

/*-------------------------------------------------------------------------------------*/

class MPIRECVOp : public OpKernel {
private:
  
public:
  explicit MPIRECVOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& src = context->input(1);
    const Tensor& tag = context->input(2);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& src_shape = src.shape();
    const TensorShape& tag_shape = tag.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(src_shape.dims(), 0);
    DCHECK_EQ(tag_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n = a_shape.dim_size(0);
    TensorShape out_shape({n});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto src_tensor = src.flat<int64>().data();
    auto tag_tensor = tag.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MPIRECV_forward(out_tensor, n, *src_tensor, *tag_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("MPIRECV").Device(DEVICE_CPU), MPIRECVOp);



class MPIRECVGradOp : public OpKernel {
private:
  
public:
  explicit MPIRECVGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& src = context->input(3);
    const Tensor& tag = context->input(4);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& src_shape = src.shape();
    const TensorShape& tag_shape = tag.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(src_shape.dims(), 0);
    DCHECK_EQ(tag_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_src_shape(src_shape);
    TensorShape grad_tag_shape(tag_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_src = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_src_shape, &grad_src));
    Tensor* grad_tag = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_tag_shape, &grad_tag));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto src_tensor = src.flat<int64>().data();
    auto tag_tensor = tag.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int n = a_shape.dim_size(0);
    MPIRECV_backward(grad_out_tensor, out_tensor, n, *src_tensor, *tag_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("MPIRECVGrad").Device(DEVICE_CPU), MPIRECVGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class MPIRECVOpGPU : public OpKernel {
private:
  
public:
  explicit MPIRECVOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& src = context->input(1);
    const Tensor& tag = context->input(2);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& src_shape = src.shape();
    const TensorShape& tag_shape = tag.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(src_shape.dims(), 0);
    DCHECK_EQ(tag_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto src_tensor = src.flat<int64>().data();
    auto tag_tensor = tag.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("MPIRECV").Device(DEVICE_GPU), MPIRECVOpGPU);

class MPIRECVGradOpGPU : public OpKernel {
private:
  
public:
  explicit MPIRECVGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& src = context->input(3);
    const Tensor& tag = context->input(4);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& src_shape = src.shape();
    const TensorShape& tag_shape = tag.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(src_shape.dims(), 0);
    DCHECK_EQ(tag_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_src_shape(src_shape);
    TensorShape grad_tag_shape(tag_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_src = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_src_shape, &grad_src));
    Tensor* grad_tag = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_tag_shape, &grad_tag));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto src_tensor = src.flat<int64>().data();
    auto tag_tensor = tag.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MPIRECVGrad").Device(DEVICE_GPU), MPIRECVGradOpGPU);

#endif