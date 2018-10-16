# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables


@tf_export("train.AsynchronousStochasticGradientDescent")
class AsynchronousStochasticGradientDescent(object):

    def apply(self, var_list=None):
        if var_list is None:
            var_list = variables.trainable_variables()

        for var in var_list:
            if var.dtype.base_dtype not in [dtypes.float16, dtypes.float32,
                                            dtypes.float64]:
                raise TypeError("The variables must be half, float, or double: %s" %
                                var.name)

            if var not in self._averages:
                with ops.init_scope():
                    if isinstance(var, variables.Variable):
                        avg = slot_creator.create_slot(var, var.initial_value(), self.name, colocate_with_primary=True)

                self._averages[var] = avg

        with ops.name_scope(self.name) as scope:
            return

    def __init__(self, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0,
                 name="AsynchronousStochasticGradientDescent"):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # 初始化变量
        self._lr = lr
        self._lambd = lambd
        self._alpha = alpha
        self._t0 = t0
        self._weight_decay = weight_decay
        self._name = name
        self._averages = {}

    @property
    def name(self):
        """The name of this AsynchronousStochasticGradientDescent object."""
        return self._name
