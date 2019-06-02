using ADCME
using PyCall
if Sys.isapple()
    py"""
    import tensorflow as tf
    from tensorflow.python.framework import ops
    inner_product_module = tf.load_op_library('build/libinner_product.dylib')
    inner_product_grad_module = tf.load_op_library('build/libinner_product_grad.dylib')
    @ops.RegisterGradient("InnerProduct")
    def _inner_product_grad_cc(op, grad):
        return inner_product_grad_module.inner_product_grad(grad, *op.inputs)
    """
else
    py"""
    import tensorflow as tf
    from tensorflow.python.framework import ops
    inner_product_module = tf.load_op_library('build/libinner_product.so')
    inner_product_grad_module = tf.load_op_library('build/libinner_product_grad.so')
    @ops.RegisterGradient("InnerProduct")
    def _inner_product_grad_cc(op, grad):
        return inner_product_grad_module.inner_product_grad(grad, *op.inputs)
    """
end

inner_product = py"inner_product_module.inner_product"

x = constant(rand(100,1), dtype=Float32)
W = constant(rand(100,100), dtype=Float32)
g = inner_product(x, W)
dg = gradients(sum(W*x), x) 
sess = Session()
run(sess, [g, dg])
