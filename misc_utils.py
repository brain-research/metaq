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

"""Direct tensorboard logger."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import time
from collections import OrderedDict
from itertools import izip

import numpy as np
import scipy

import tensorflow as tf


def define_flags_with_default(**kwargs):
  ordered_kwargs = OrderedDict()
  for key in sorted(kwargs.keys()):
    ordered_kwargs[key] = kwargs[key]

  for key, val in ordered_kwargs.items():
    if isinstance(val, bool):
      # Note that True and False are instances of int.
      tf.app.flags.DEFINE_bool(key, val, 'automatically defined flag')
    elif isinstance(val, int):
      tf.app.flags.DEFINE_integer(key, val, 'automatically defined flag')
    elif isinstance(val, float):
      tf.app.flags.DEFINE_float(key, val, 'automatically defined flag')
    elif isinstance(val, str):
      tf.app.flags.DEFINE_string(key, val, 'automatically defined flag')
    else:
      raise ValueError('Incorrect value type')
  return ordered_kwargs

class TensorBoardLogger(object):
  """Logging to TensorBoard outside of TensorFlow ops."""

  def __init__(self, output_dir):
    if not tf.gfile.Exists(output_dir):
      tf.gfile.MakeDirs(output_dir)
    self.file_writer = tf.summary.FileWriter(output_dir)

  def log_scaler(self, step, name, value):
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=name, simple_value=value)]
    )
    self.file_writer.add_summary(summary, step)

  def log_dict(self, step, data):
    summary = tf.Summary(
        value=[
            tf.Summary.Value(tag=name, simple_value=value)
            for name, value in data.items()
        ]
    )
    self.file_writer.add_summary(summary, step)
