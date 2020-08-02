#include "common.h"

void MPIBCAST_forward(double *out, const double *a, int m, int root){
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank( comm , &rank);
  if (rank==root){
    memcpy(out, a, sizeof(double)*m);
  }
  MPI_Request request;
  MPI_Status status; 
  MPI_Ibcast(out, m, MPI_DOUBLE, root, comm, &request);
  MPI_Wait( &request , &status);
  
}

void MPIBCAST_backward(
  double *grad_a, const double *grad_out,
  const double *out, const double *a, int m, int root
){
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank( comm , &rank);
  MPI_Request request;
  MPI_Status status; 
  MPI_Ireduce( grad_out , grad_a , m , MPI_DOUBLE , MPI_SUM , root , comm , &request);
  MPI_Wait( &request , &status);
}

REGISTER_OP("MPIBCAST")
.Input("a : double")
.Input("root : int64")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_shape));
        shape_inference::ShapeHandle root_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &root_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("MPIBCASTGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("a : double")
.Input("root : int64")
.Output("grad_a : double")
.Output("grad_root : int64");

/*-------------------------------------------------------------------------------------*/

class MPIBCASTOp : public OpKernel {
private:
  
public:
  explicit MPIBCASTOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& root = context->input(1);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& root_shape = root.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(root_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n = a_shape.dim_size(0);
    TensorShape out_shape({n});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto root_tensor = root.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MPIBCAST_forward(out_tensor, a_tensor, n, *root_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("MPIBCAST").Device(DEVICE_CPU), MPIBCASTOp);



class MPIBCASTGradOp : public OpKernel {
private:
  
public:
  explicit MPIBCASTGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& root = context->input(3);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& root_shape = root.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(root_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_root_shape(root_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_root = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_root_shape, &grad_root));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto root_tensor = root.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int m = a_shape.dim_size(0);
    MPIBCAST_backward(grad_a_tensor, grad_out_tensor, out_tensor, a_tensor, m, *root_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("MPIBCASTGrad").Device(DEVICE_CPU), MPIBCASTGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class MPIBCASTOpGPU : public OpKernel {
private:
  
public:
  explicit MPIBCASTOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& root = context->input(1);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& root_shape = root.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(root_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto root_tensor = root.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("MPIBCAST").Device(DEVICE_GPU), MPIBCASTOpGPU);

class MPIBCASTGradOpGPU : public OpKernel {
private:
  
public:
  explicit MPIBCASTGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& root = context->input(3);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& root_shape = root.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(root_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_root_shape(root_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_root = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_root_shape, &grad_root));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto root_tensor = root.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MPIBCASTGrad").Device(DEVICE_GPU), MPIBCASTGradOpGPU);

#endif