/// \file inner_product.cc
/// \author David Stutz
/// \brief Implementation of a inner product (i.e. fully connected layer)
/// operation in Tensorflow.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("InnerProduct")
  .Input("input: float")
  .Input("weights: float")
  .Output("inner_product: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

    shape_inference::ShapeHandle weight_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));
    
    shape_inference::DimensionHandle output_rows = c->Dim(weight_shape, 0);
  
    shape_inference::DimensionHandle input_rows = c->Dim(input_shape, 0);
    shape_inference::DimensionHandle weight_cols = c->Dim(weight_shape, 1);
    shape_inference::DimensionHandle merged;
    TF_RETURN_IF_ERROR(c->Merge(input_rows, weight_cols, &merged));

    c->set_output(0, c->Matrix(output_rows, 1));
    return Status::OK();
  });

/// \brief Implementation of an inner product operation.
/// \param context
/// \author David Stutz
class InnerProductOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit InnerProductOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // some checks to be sure ...
    DCHECK_EQ(2, context->num_inputs());
    
    // get the input tensor
    const Tensor& input = context->input(0);
    
    // get the weight tensor
    const Tensor& weights = context->input(1);
    
    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weights.shape();
    
    // check input is a standing vector
    DCHECK_EQ(input_shape.dims(), 2);
    DCHECK_EQ(input_shape.dim_size(1), 1);
    
    // check weights is matrix of correct size
    DCHECK_EQ(weights_shape.dims(), 2);
    DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));
    
    // create output shape
    TensorShape output_shape;
    output_shape.AddDim(weights_shape.dim_size(0));
    output_shape.AddDim(1);
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    auto input_tensor = input.matrix<float>();
    auto weights_tensor = weights.matrix<float>();
    auto output_tensor = output->matrix<float>();
    
    for (int i = 0; i < output->shape().dim_size(0); i++) {
      output_tensor(i, 0) = 0;
      for (int j = 0; j < weights.shape().dim_size(1); j++) {
        output_tensor(i, 0) += weights_tensor(i, j)*input_tensor(j, 0);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("InnerProduct").Device(DEVICE_CPU), InnerProductOp);
