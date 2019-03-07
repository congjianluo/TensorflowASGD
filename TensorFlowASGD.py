"""AsynchronousStochasticGradientDescent for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops, resource_variable_ops
from tensorflow.python.training import optimizer, training_ops


class AsynchronousStochasticGradientDescent(optimizer.Optimizer):
    """Optimizer that implements the gradient descent algorithm.
    """

    def __init__(self, learning_rate, t0=0, use_locking=False, name="GradientDescent"):
        """Construct a new gradient descent optimizer.

        Args:
          learning_rate: A Tensor or a floating point value.  The learning
            rate to use.
          use_locking: If True use locks for update operations.
          name: Optional name prefix for the operations created when applying
            gradients. Defaults to "GradientDescent".
        """
        super(AsynchronousStochasticGradientDescent, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._t0 = t0

    def _create_slots(self, var_list):
        for v in var_list:
            self._get_or_make_slot(v, math_ops.cast(0, v.dtype.base_dtype), "step", self._name)
            self._get_or_make_slot(v, math_ops.cast(self._learning_rate, v.dtype.base_dtype), "eta", self._name)
            self._get_or_make_slot(v, math_ops.cast(1, v.dtype.base_dtype), "mu", self._name)
            self._zeros_slot(v, "ax", self._name)

    def _apply_dense(self, grad, var):
        step = self.get_slot(var, "step")
        step_t = step.assign(step + 1)

        mu = self.get_slot(var, "mu")
        ax = self.get_slot(var, "ax")

        var_t = training_ops.apply_gradient_descent(
            var,
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking)

        var_T = var_t.op

        if mu != 1:
            ax_t = ax.assign(ax + (var_t - ax) * mu)
        else:
            ax_t = ax.assign(var_t)
        mu_t = mu.assign(1 / tf.maximum(math_ops.cast(1, step_t.dtype), step_t - self._t0))
        return control_flow_ops.group(*[var_T, step_t, ax_t, mu_t])

    def _resource_apply_dense(self, grad, handle):
        return training_ops.resource_apply_gradient_descent(
            handle, math_ops.cast(self._learning_rate_tensor,
                                  grad.dtype.base_dtype),
            grad, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, handle, indices):
        return resource_variable_ops.resource_scatter_add(
            handle, indices, -grad * self._learning_rate)

    def _apply_sparse_duplicate_indices(self, grad, var):
        delta = ops.IndexedSlices(
            grad.values *
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad.indices, grad.dense_shape)
        return var.scatter_sub(delta, use_locking=self._use_locking)

    def _prepare(self):
        self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                           name="learning_rate")

    @staticmethod
    def _broadcast(tensor, shape):
        return tf.add(tensor, tf.zeros(shape, dtype=tensor.dtype))
