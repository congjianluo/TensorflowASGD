"""AsynchronousStochasticGradientDescent for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import math_ops, state_ops, control_flow_ops
from tensorflow.python.training import optimizer


class AsynchronousStochasticGradientDescent(optimizer.Optimizer):
    """Optimizer that implements the gradient descent algorithm.
    """

    def __init__(self, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0, use_locking=False,
                 name="AsynchronousStochasticGradientDescent"):
        """Construct a new gradient descent optimizer.

        Args:
          learning_rate: A Tensor or a floating point value.  The learning
            rate to use.
          use_locking: If True use locks for update operations.
          name: Optional name prefix for the operations created when applying
            gradients. Defaults to "GradientDescent".
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        super(AsynchronousStochasticGradientDescent, self).__init__(use_locking, name)
        self._lr = lr
        self._lambd = lambd
        self._alpha = alpha
        self._t0 = t0
        self._weight_decay = weight_decay
        self._name = name
        self.step = 0

    def _create_slots(self, var_list):
        for v in var_list:
            self._get_or_make_slot(v, math_ops.cast(0, v.dtype.base_dtype), "step", self._name)
            self._get_or_make_slot(v, math_ops.cast(self._lr, v.dtype.base_dtype), "eta", self._name)
            self._get_or_make_slot(v, math_ops.cast(1, v.dtype.base_dtype), "mu", self._name)
            self._get_or_make_slot(v, self._broadcast(math_ops.cast(0, v.dtype.base_dtype), v.shape), "ax", self._name)

    def _prepare(self):
        # self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        # self._lambd_t = ops.convert_to_tensor(self._t0, name="lambd")
        # self._alpha_t = ops.convert_to_tensor(self._t0, name="alpha")
        return

    def _apply_dense(self, grad, var):
        step = self.get_slot(var, "step")
        step_t = step.assign(step + 1)

        eta = self.get_slot(var, "eta")
        mu = self.get_slot(var, "mu")
        ax = self.get_slot(var, "ax")

        grad_update = grad
        # Not work now.
        if self._weight_decay != 0:
            grad_update = grad + self._weight_decay * var
            grad = grad + self._weight_decay * var

        # decay term
        var_t = (1 - self._lambd * eta) * var
        # var_update = state_ops.assign(var, self._broadcast(math_ops.cast(0, var.dtype.base_dtype), var.shape))
        var_update = state_ops.assign(var, var_t - eta * grad)

        if mu != 1:
            ax_t = ax.assign(ax + (var_update - ax) * mu)
        else:
            ax_t = ax.assign(var_update)

        eta_t = eta.assign(self._lr / tf.pow(1 + self._lambd * self._lr * step_t, self._alpha))

        mu_t = mu.assign(1 / tf.maximum(math_ops.cast(1, step_t.dtype), step_t - self._t0))

        return control_flow_ops.group(*[var_update, step_t, ax_t, eta_t, mu_t, grad_update])

    def _resource_apply_dense(self, grad, handle):
        raise NotImplementedError("Not implement _resource_apply_dense!")

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        raise NotImplementedError("Not implement _resource_apply_sparse_duplicate_indices!")

    def _apply_sparse_duplicate_indices(self, grad, var):
        raise NotImplementedError("Not implement _apply_sparse_duplicate_indices!")

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Not implement _apply_sparse!")


    @staticmethod
    def _broadcast(tensor, shape):
        return tf.add(tensor, tf.zeros(shape, dtype=tensor.dtype))
