#include "common.h"


void MPISUM_forward(double *out, const double *a, int n, int root){
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Request request;
  MPI_Status status; 
  int rank;
  MPI_Comm_rank( comm , &rank);
  MPI_Ireduce( a , out , n , MPI_DOUBLE , MPI_SUM , root , comm , &request);
  MPI_Wait( &request , &status);
}

void MPISUM_backward(
  double *grad_a, const double *grad_out,
  const double *out, const double *a, int n, int root
){
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank( comm , &rank);
  for(int i =0;i<n;i++) grad_a[i] = grad_out[i];
  MPI_Request request;
  MPI_Status status; 
  MPI_Ibcast(grad_a, n, MPI_DOUBLE, root, comm, &request);
  MPI_Wait( &request , &status);

}

REGISTER_OP("MPISUM")
.Input("a : double")
.Input("node : int64")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_shape));
        shape_inference::ShapeHandle node_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &node_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("MPISUMGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("a : double")
.Input("node : int64")
.Output("grad_a : double")
.Output("grad_node : int64");

/*-------------------------------------------------------------------------------------*/

class MPISUMOp : public OpKernel {
private:
  
public:
  explicit MPISUMOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& node = context->input(1);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& node_shape = node.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(node_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n = a_shape.dim_size(0);
    TensorShape out_shape({n});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto node_tensor = node.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MPISUM_forward(out_tensor, a_tensor, n, *node_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("MPISUM").Device(DEVICE_CPU), MPISUMOp);



class MPISUMGradOp : public OpKernel {
private:
  
public:
  explicit MPISUMGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& node = context->input(3);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& node_shape = node.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(node_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_node_shape(node_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_node = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_node_shape, &grad_node));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto node_tensor = node.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int n = a_shape.dim_size(0);
    MPISUM_backward(grad_a_tensor, grad_out_tensor, out_tensor, a_tensor, n, *node_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MPISUMGrad").Device(DEVICE_CPU), MPISUMGradOp);
