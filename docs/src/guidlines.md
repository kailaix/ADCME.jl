# Custom Operator Guidelines

* Whenever memory is needed, one should allocate memory by TensorFlow context. 
```cpp
Tensor* tmp_var = nullptr;
TensorShae tmp_shape({10,10});
OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, tmp_shape, &tmp_var));
```

There are three methods to allocate Tensors when an Op kernel executes ([details](https://github.com/tensorflow/tensorflow/blob/584876113e6248639d18d4e16c77b47cb1b251c1/tensorflow/core/framework/op_kernel.h#L753-L801))
- `allocate_persistent`: if the memory is used between Op invocations.
- `allocate_temp`: if the memory is used only within `Compute`.
- `allocate_output`: if the memory will be used as output


## Create Custom Variables

In this section, we analyze the codes in the file `tensorflow/core/kernels/variable_ops.cc`, `tensorflow/core/kernels/variable_ops.h`. The signature for `VariableOp` is
```cpp
REGISTER_OP("VariableV2")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);
```
Since Variables will be updated during training, it is necessary to guarantee the thread safety. Therefore, each VariableOp instance will usually hold a private mutex
```cpp
  mutex init_mu_;
```

In the implementation, 