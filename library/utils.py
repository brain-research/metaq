# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator
import numpy as np
from collections import OrderedDict

import tensorflow as tf


class GraphCollection(dict):
  """Handy class for graph tensor collection."""

  def __init__(self, *args, **kwargs):
    super(GraphCollection, self).__init__(*args, **kwargs)
    self.__dict__ = self


class TensorAvarageAccumulator(object):
  """Accumulate tensor values from multiple batches to enable larger effective
  batch size.
  """
  def __init__(self, input_tensor, accum_weight=1):
    accum_weight = tf.cast(accum_weight, tf.float32)
    self._accum_var = tf.Variable(
        tf.zeros_like(input_tensor),
        dtype=input_tensor.dtype,
        trainable=False
    )
    self._counter = tf.Variable(0, dtype=tf.float32, trainable=False)
    self._zeros_op = tf.group(
        self._accum_var.assign(tf.zeros_like(input_tensor)),
        self._counter.assign(0)
    )
    self._accum_op = tf.group(
        self._accum_var.assign_add(input_tensor * accum_weight),
        self._counter.assign_add(accum_weight)
    )
    self._value = tf.cond(
        self._counter > 0,
        lambda: self._accum_var / self._counter,
        lambda: 0.0
    )

  @property
  def accum_var(self):
    return self._accum_var

  @property
  def counter(self):
    return self._counter

  @property
  def zeros_op(self):
    return self._zeros_op

  @property
  def accum_op(self):
    return self._accum_op

  @property
  def value(self):
    return self._value


class TensorAverageAccumulatorList(object):
  def __init__(self, input_tensors, accum_weight=1):
    self.accumulators = [
        TensorAvarageAccumulator(t, accum_weight) for t in input_tensors
    ]

    self._accum_vars = [a.accum_var for a in self.accumulators]
    self._zeros_op = tf.group(*[a.zeros_op for a in self.accumulators])
    self._accum_op = tf.group(*[a.accum_op for a in self.accumulators])
    self._values = [a.value for a in self.accumulators]

  @property
  def accum_vars(self):
    return self._accum_vars

  @property
  def counter(self):
    return self.accumulators[0].counter

  @property
  def zeros_op(self):
    return self._zeros_op

  @property
  def accum_op(self):
    return self._accum_op

  @property
  def values(self):
    return self._values


class TensorAverageAccumulatorDict(object):
  def __init__(self, input_tensors, accum_weight=1):
    self.accumulators = AttributeDict({
        k: TensorAvarageAccumulator(t, accum_weight) for k, t in input_tensors.items()
    })

    self._accum_vars = AttributeDict(
        {k: a.accum_var for k, a in self.accumulators.items()}
    )
    self._zeros_op = tf.group(*[a.zeros_op for a in self.accumulators.values()])
    self._accum_op = tf.group(*[a.accum_op for a in self.accumulators.values()])
    self._values = AttributeDict(
        {k: a.value for k, a in self.accumulators.items()}
    )

  @property
  def accum_vars(self):
    return self._accum_vars

  @property
  def counter(self):
    return self.accumulators[0].counter

  @property
  def zeros_op(self):
    return self._zeros_op

  @property
  def accum_op(self):
    return self._accum_op

  @property
  def values(self):
    return self._values
