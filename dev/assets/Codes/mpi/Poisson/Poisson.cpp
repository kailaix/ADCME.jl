#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include "mpi.h"

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "Poisson.h"


REGISTER_OP("Poisson")
.Input("u : double")
.Input("up : double")
.Input("down : double")
.Input("left : double")
.Input("right : double")
.Input("f : double")
.Input("h : double")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &u_shape));
        shape_inference::ShapeHandle up_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &up_shape));
        shape_inference::ShapeHandle down_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &down_shape));
        shape_inference::ShapeHandle left_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &left_shape));
        shape_inference::ShapeHandle right_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &right_shape));
        shape_inference::ShapeHandle f_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 2, &f_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &h_shape));

        c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("PoissonGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("u : double")
.Input("up : double")
.Input("down : double")
.Input("left : double")
.Input("right : double")
.Input("f : double")
.Input("h : double")
.Output("grad_u : double")
.Output("grad_up : double")
.Output("grad_down : double")
.Output("grad_left : double")
.Output("grad_right : double")
.Output("grad_f : double")
.Output("grad_h : double");

/*-------------------------------------------------------------------------------------*/

class PoissonOp : public OpKernel {
private:
  
public:
  explicit PoissonOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(7, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& up = context->input(1);
    const Tensor& down = context->input(2);
    const Tensor& left = context->input(3);
    const Tensor& right = context->input(4);
    const Tensor& f = context->input(5);
    const Tensor& h = context->input(6);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& up_shape = up.shape();
    const TensorShape& down_shape = down.shape();
    const TensorShape& left_shape = left.shape();
    const TensorShape& right_shape = right.shape();
    const TensorShape& f_shape = f.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 2);
    DCHECK_EQ(up_shape.dims(), 1);
    DCHECK_EQ(down_shape.dims(), 1);
    DCHECK_EQ(left_shape.dims(), 1);
    DCHECK_EQ(right_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n = u_shape.dim_size(0), m = u_shape.dim_size(1);
    TensorShape out_shape({n, m});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto up_tensor = up.flat<double>().data();
    auto down_tensor = down.flat<double>().data();
    auto left_tensor = left.flat<double>().data();
    auto right_tensor = right.flat<double>().data();
    auto f_tensor = f.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(out_tensor, u_tensor, up_tensor, down_tensor, 
      left_tensor, right_tensor, f_tensor, m, n, *h_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("Poisson").Device(DEVICE_CPU), PoissonOp);



class PoissonGradOp : public OpKernel {
private:
  
public:
  explicit PoissonGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& up = context->input(3);
    const Tensor& down = context->input(4);
    const Tensor& left = context->input(5);
    const Tensor& right = context->input(6);
    const Tensor& f = context->input(7);
    const Tensor& h = context->input(8);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& up_shape = up.shape();
    const TensorShape& down_shape = down.shape();
    const TensorShape& left_shape = left.shape();
    const TensorShape& right_shape = right.shape();
    const TensorShape& f_shape = f.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 2);
    DCHECK_EQ(out_shape.dims(), 2);
    DCHECK_EQ(u_shape.dims(), 2);
    DCHECK_EQ(up_shape.dims(), 1);
    DCHECK_EQ(down_shape.dims(), 1);
    DCHECK_EQ(left_shape.dims(), 1);
    DCHECK_EQ(right_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_up_shape(up_shape);
    TensorShape grad_down_shape(down_shape);
    TensorShape grad_left_shape(left_shape);
    TensorShape grad_right_shape(right_shape);
    TensorShape grad_f_shape(f_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_up = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_up_shape, &grad_up));
    Tensor* grad_down = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_down_shape, &grad_down));
    Tensor* grad_left = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_left_shape, &grad_left));
    Tensor* grad_right = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_right_shape, &grad_right));
    Tensor* grad_f = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_f_shape, &grad_f));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto up_tensor = up.flat<double>().data();
    auto down_tensor = down.flat<double>().data();
    auto left_tensor = left.flat<double>().data();
    auto right_tensor = right.flat<double>().data();
    auto f_tensor = f.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_up_tensor = grad_up->flat<double>().data();
    auto grad_down_tensor = grad_down->flat<double>().data();
    auto grad_left_tensor = grad_left->flat<double>().data();
    auto grad_right_tensor = grad_right->flat<double>().data();
    auto grad_f_tensor = grad_f->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("PoissonGrad").Device(DEVICE_CPU), PoissonGradOp);




REGISTER_OP("DataExchange")
.Input("left : double")
.Input("right : double")
.Input("up : double")
.Input("down : double")
.Input("mblock : int64")
.Input("nblock : int64")
.Output("outleft : double")
.Output("outright : double")
.Output("outup : double")
.Output("outdown : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle left_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &left_shape));
        shape_inference::ShapeHandle right_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &right_shape));
        shape_inference::ShapeHandle up_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &up_shape));
        shape_inference::ShapeHandle down_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &down_shape));
        shape_inference::ShapeHandle mblock_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &mblock_shape));
        shape_inference::ShapeHandle nblock_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &nblock_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
        c->set_output(3, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("DataExchangeGrad")
.Input("grad_outleft : double")
.Input("grad_outright : double")
.Input("grad_outup : double")
.Input("grad_outdown : double")
.Input("outleft : double")
.Input("outright : double")
.Input("outup : double")
.Input("outdown : double")
.Input("left : double")
.Input("right : double")
.Input("up : double")
.Input("down : double")
.Input("mblock : int64")
.Input("nblock : int64")
.Output("grad_left : double")
.Output("grad_right : double")
.Output("grad_up : double")
.Output("grad_down : double")
.Output("grad_mblock : int64")
.Output("grad_nblock : int64");

/*-------------------------------------------------------------------------------------*/

class DataExchangeOp : public OpKernel {
private:
  
public:
  explicit DataExchangeOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& left = context->input(0);
    const Tensor& right = context->input(1);
    const Tensor& up = context->input(2);
    const Tensor& down = context->input(3);
    const Tensor& mblock = context->input(4);
    const Tensor& nblock = context->input(5);
    
    
    const TensorShape& left_shape = left.shape();
    const TensorShape& right_shape = right.shape();
    const TensorShape& up_shape = up.shape();
    const TensorShape& down_shape = down.shape();
    const TensorShape& mblock_shape = mblock.shape();
    const TensorShape& nblock_shape = nblock.shape();
    
    
    DCHECK_EQ(left_shape.dims(), 1);
    DCHECK_EQ(right_shape.dims(), 1);
    DCHECK_EQ(up_shape.dims(), 1);
    DCHECK_EQ(down_shape.dims(), 1);
    DCHECK_EQ(mblock_shape.dims(), 0);
    DCHECK_EQ(nblock_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n = left_shape.dim_size(0);
    int m = up_shape.dim_size(0);
    TensorShape outleft_shape({n});
    TensorShape outright_shape({n});
    TensorShape outup_shape({m});
    TensorShape outdown_shape({m});
            
    // create output tensor
    
    Tensor* outleft = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, outleft_shape, &outleft));
    Tensor* outright = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, outright_shape, &outright));
    Tensor* outup = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, outup_shape, &outup));
    Tensor* outdown = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, outdown_shape, &outdown));
    
    // get the corresponding Eigen tensors for data access
    
    auto left_tensor = left.flat<double>().data();
    auto right_tensor = right.flat<double>().data();
    auto up_tensor = up.flat<double>().data();
    auto down_tensor = down.flat<double>().data();
    auto mblock_tensor = mblock.flat<int64>().data();
    auto nblock_tensor = nblock.flat<int64>().data();
    auto outleft_tensor = outleft->flat<double>().data();
    auto outright_tensor = outright->flat<double>().data();
    auto outup_tensor = outup->flat<double>().data();
    auto outdown_tensor = outdown->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    outleft->flat<double>().setZero();
    outright->flat<double>().setZero();
    outup->flat<double>().setZero();
    outdown->flat<double>().setZero();
    data_forward(outleft_tensor, outright_tensor, outup_tensor, outdown_tensor, 
      left_tensor, right_tensor, up_tensor, down_tensor, *mblock_tensor, *nblock_tensor, m, n);


  }
};
REGISTER_KERNEL_BUILDER(Name("DataExchange").Device(DEVICE_CPU), DataExchangeOp);



class DataExchangeGradOp : public OpKernel {
private:
  
public:
  explicit DataExchangeGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_outleft = context->input(0);
    const Tensor& grad_outright = context->input(1);
    const Tensor& grad_outup = context->input(2);
    const Tensor& grad_outdown = context->input(3);
    const Tensor& outleft = context->input(4);
    const Tensor& outright = context->input(5);
    const Tensor& outup = context->input(6);
    const Tensor& outdown = context->input(7);
    const Tensor& left = context->input(8);
    const Tensor& right = context->input(9);
    const Tensor& up = context->input(10);
    const Tensor& down = context->input(11);
    const Tensor& mblock = context->input(12);
    const Tensor& nblock = context->input(13);
    
    
    const TensorShape& grad_outleft_shape = grad_outleft.shape();
    const TensorShape& grad_outright_shape = grad_outright.shape();
    const TensorShape& grad_outup_shape = grad_outup.shape();
    const TensorShape& grad_outdown_shape = grad_outdown.shape();
    const TensorShape& outleft_shape = outleft.shape();
    const TensorShape& outright_shape = outright.shape();
    const TensorShape& outup_shape = outup.shape();
    const TensorShape& outdown_shape = outdown.shape();
    const TensorShape& left_shape = left.shape();
    const TensorShape& right_shape = right.shape();
    const TensorShape& up_shape = up.shape();
    const TensorShape& down_shape = down.shape();
    const TensorShape& mblock_shape = mblock.shape();
    const TensorShape& nblock_shape = nblock.shape();
    
    
    DCHECK_EQ(grad_outleft_shape.dims(), 1);
    DCHECK_EQ(grad_outright_shape.dims(), 1);
    DCHECK_EQ(grad_outup_shape.dims(), 1);
    DCHECK_EQ(grad_outdown_shape.dims(), 1);
    DCHECK_EQ(outleft_shape.dims(), 1);
    DCHECK_EQ(outright_shape.dims(), 1);
    DCHECK_EQ(outup_shape.dims(), 1);
    DCHECK_EQ(outdown_shape.dims(), 1);
    DCHECK_EQ(left_shape.dims(), 1);
    DCHECK_EQ(right_shape.dims(), 1);
    DCHECK_EQ(up_shape.dims(), 1);
    DCHECK_EQ(down_shape.dims(), 1);
    DCHECK_EQ(mblock_shape.dims(), 0);
    DCHECK_EQ(nblock_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_left_shape(left_shape);
    TensorShape grad_right_shape(right_shape);
    TensorShape grad_up_shape(up_shape);
    TensorShape grad_down_shape(down_shape);
    TensorShape grad_mblock_shape(mblock_shape);
    TensorShape grad_nblock_shape(nblock_shape);
            
    // create output tensor
    
    Tensor* grad_left = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_left_shape, &grad_left));
    Tensor* grad_right = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_right_shape, &grad_right));
    Tensor* grad_up = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_up_shape, &grad_up));
    Tensor* grad_down = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_down_shape, &grad_down));
    Tensor* grad_mblock = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_mblock_shape, &grad_mblock));
    Tensor* grad_nblock = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_nblock_shape, &grad_nblock));
    
    // get the corresponding Eigen tensors for data access
    
    auto left_tensor = left.flat<double>().data();
    auto right_tensor = right.flat<double>().data();
    auto up_tensor = up.flat<double>().data();
    auto down_tensor = down.flat<double>().data();
    auto mblock_tensor = mblock.flat<int64>().data();
    auto nblock_tensor = nblock.flat<int64>().data();
    auto grad_outleft_tensor = grad_outleft.flat<double>().data();
    auto grad_outright_tensor = grad_outright.flat<double>().data();
    auto grad_outup_tensor = grad_outup.flat<double>().data();
    auto grad_outdown_tensor = grad_outdown.flat<double>().data();
    auto outleft_tensor = outleft.flat<double>().data();
    auto outright_tensor = outright.flat<double>().data();
    auto outup_tensor = outup.flat<double>().data();
    auto outdown_tensor = outdown.flat<double>().data();
    auto grad_left_tensor = grad_left->flat<double>().data();
    auto grad_right_tensor = grad_right->flat<double>().data();
    auto grad_up_tensor = grad_up->flat<double>().data();
    auto grad_down_tensor = grad_down->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("DataExchangeGrad").Device(DEVICE_CPU), DataExchangeGradOp);


