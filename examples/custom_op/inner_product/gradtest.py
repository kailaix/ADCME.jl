#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.

.. moduleauthor:: David Stutz
"""

import unittest
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import sparse_ops
inner_product_module = tf.load_op_library('build/libinner_product.dylib')
inner_product_grad_module = tf.load_op_library('build/libinner_product_grad.dylib')
@ops.RegisterGradient("InnerProduct")
def _inner_product_grad_cc(op, grad):
    return inner_product_grad_module.inner_product_grad(grad, op.inputs[0], op.inputs[1])


sess = tf.Session()
res = inner_product_module.inner_product([[1], [2]], [[1, 2], [3, 4]])
print(sess.run(res))

x = tf.constant(np.asarray([[1.], [2.]]).astype(np.float32))
W = tf.constant(np.asarray([[1, 2], [3, 4]]).astype(np.float32))
y = inner_product_module.inner_product(x, W)
grad_y = tf.gradients(y[1], W)[0]
sess.run(tf.global_variables_initializer())
print(sess.run(grad_y))
sess.close()
