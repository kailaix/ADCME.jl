#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "mpi.h"


void MPIGather_forward(double *out, const double *a, int m, int root){
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Gather( a , m , MPI_DOUBLE , out , m , MPI_DOUBLE , root , comm);
}

void MPIGather_backward(
  double *grad_a, const double *grad_out,
  const double *out, const double *a, int m, int root
){
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Scatter( grad_out , m , MPI_DOUBLE , grad_a ,  m , MPI_DOUBLE , root , comm);
}




REGISTER_OP("MPIGATHER")
.Input("u : double")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("MPIGATHERGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("u : double")
.Output("grad_u : double");

/*-------------------------------------------------------------------------------------*/

class MPIGATHEROp : public OpKernel {
private:
  
public:
  explicit MPIGATHEROp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    
    
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);

    // extra check
        
    // create output shape
    MPI_Comm comm = MPI_COMM_WORLD;
    int size;
    MPI_Comm_size( comm , &size);
    int m = u_shape.dim_size(0);
    TensorShape out_shape({m * size});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MPIGather_forward(out_tensor, u_tensor, m, 0);

  }
};
REGISTER_KERNEL_BUILDER(Name("MPIGATHER").Device(DEVICE_CPU), MPIGATHEROp);



class MPIGATHERGradOp : public OpKernel {
private:
  
public:
  explicit MPIGATHERGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& u = context->input(2);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int m = u_shape.dim_size(0);
    MPIGather_backward(grad_u_tensor, grad_out_tensor, out_tensor, u_tensor, m, 0);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MPIGATHERGrad").Device(DEVICE_CPU), MPIGATHERGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class MPIGATHEROpGPU : public OpKernel {
private:
  
public:
  explicit MPIGATHEROpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    
    
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("MPIGATHER").Device(DEVICE_GPU), MPIGATHEROpGPU);

class MPIGATHERGradOpGPU : public OpKernel {
private:
  
public:
  explicit MPIGATHERGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& u = context->input(2);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MPIGATHERGrad").Device(DEVICE_GPU), MPIGATHERGradOpGPU);

#endif