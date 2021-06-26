#include "common.h"

void MPIBarrier_forward(){
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Barrier(comm);
}

REGISTER_OP("MpiBarrier")
.Input("x : double")
.Output("y : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &x_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });

REGISTER_OP("MpiBarrierGrad")
.Input("grad_y : double")
.Input("y : double")
.Input("x : double")
.Output("grad_x : double");

/*-------------------------------------------------------------------------------------*/

class MpiBarrierOp : public OpKernel {
private:
  
public:
  explicit MpiBarrierOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    
    
    const TensorShape& x_shape = x.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape y_shape({});
            
    // create output tensor
    
    Tensor* y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, y_shape, &y));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MPIBarrier_forward();
    *y_tensor = 0.0;

  }
};
REGISTER_KERNEL_BUILDER(Name("MpiBarrier").Device(DEVICE_CPU), MpiBarrierOp);



class MpiBarrierGradOp : public OpKernel {
private:
  
public:
  explicit MpiBarrierGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_y = context->input(0);
    const Tensor& y = context->input(1);
    const Tensor& x = context->input(2);
    
    
    const TensorShape& grad_y_shape = grad_y.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& x_shape = x.shape();
    
    
    DCHECK_EQ(grad_y_shape.dims(), 0);
    DCHECK_EQ(y_shape.dims(), 0);
    DCHECK_EQ(x_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_x_shape(x_shape);
            
    // create output tensor
    
    Tensor* grad_x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_x_shape, &grad_x));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto grad_y_tensor = grad_y.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto grad_x_tensor = grad_x->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    *grad_x_tensor = 0.0;
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MpiBarrierGrad").Device(DEVICE_CPU), MpiBarrierGradOp);
