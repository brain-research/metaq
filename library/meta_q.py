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

from parameterized_model import ParameterizedModel
import utils


def get_loss(input_tensor, loss_type):
  """Compute loss by name of loss function."""
  input_length = len(input_tensor.get_shape().as_list())
  if loss_type == 'l2':
    error = 0.5 * tf.square(input_tensor)
  elif loss_type == 'l1':
    error = tf.abs(input_tensor)
  elif loss_type == 'huber':
    error = tf.where(
        tf.abs(input_tensor) < 1.0,
        tf.square(input_tensor) * 0.5,
        tf.abs(input_tensor) - 0.5
    )
  else:
    raise ValueError('Unsupported loss type')

  return tf.reduce_mean(tf.reduce_sum(error, axis=range(1, input_length)))


def select_q_values_by_action(q_values, actions):
  """Select q values by the corresponding actions."""
  return tf.reduce_sum(
      q_values * tf.one_hot(actions, q_values.get_shape()[1]),
      axis=1
  )


def bellman_error(predicted_q_current, current_actions, rewards,
                  target_predicted_q_next, dones, discount_factor,
                  predicted_q_next=None, online_target=False, double_q=False,
                  loss_type='l2', residue_gradient=False, soft_q=False,
                  soft_q_temperature=1.0, debug_ground_truth=False):

  assert not (double_q and soft_q)   # Currently we do not support double soft q
  if double_q or online_target:
    assert predicted_q_next is not None

  current_action_q_value = select_q_values_by_action(
      predicted_q_current, current_actions
  )

  if debug_ground_truth:
    error = current_action_q_value - rewards
    return get_loss(error, loss_type), None

  if online_target:
    # Use online q network for target
    next_q = predicted_q_next
  else:
    # Use target network
    next_q = target_predicted_q_next

  if soft_q:
    target_value = soft_q_temperature * tf.reduce_logsumexp(
        next_q/soft_q_temperature, 1
    )
  else:
    if double_q:
      next_step_actions = tf.argmax(predicted_q_next, axis=1)
    else:
      next_step_actions = tf.argmax(target_predicted_q_next, axis=1)

    target_value = select_q_values_by_action(next_q, next_step_actions)

  target_value = (1.0 - tf.cast(dones, tf.float32)) * target_value

  # if not residue_gradient:
  #   target_value = tf.stop_gradient(target_value)

  error = current_action_q_value - rewards - discount_factor * target_value
  return get_loss(error, loss_type), target_value


class QFunction(object):
  """Callable q function."""

  def __init__(self, session, observation_tensor, output_tensor,
               parameter_tensor=None, parameter=None):
    self.session = session
    self.observation_tensor = observation_tensor
    self.output_tensor = output_tensor
    self.parameter_tensor = parameter_tensor
    self.parameter = parameter

  def __call__(self, observations):
    feed_dict = {self.observation_tensor: observations}
    if self.parameter is not None:
      feed_dict[self.parameter_tensor] = self.parameter
    return self.session.run(self.output_tensor, feed_dict)

  def act_greedy(self, observations, epsilon=0.0):
    q_values = self(observations)
    n_actions = q_values.shape[1]
    actions = []
    for q in q_values:
      if epsilon == 0.0:
        actions.append(np.argmax(q))
      else:
        prob = epsilon / n_actions * np.ones(n_actions)
        prob[np.argmax(q)] += 1 - epsilon
        actions.append(np.random.choice(n_actions, p=prob))
    return np.array(actions, dtype=np.int64)

  def act_bolzmann(self, observations, temp=1.0):
    q_values = self(observations)
    n_actions = q_values.shape[1]
    actions = []
    for q in q_values:
      ex = np.exp(q - np.max(q))
      prob = ex / np.sum(ex)
      actions.append(np.random.choice(n_actions, p=prob))
    return np.array(actions, dtype=np.int64)


class QPolicy(object):
  """Construct a policy from callable Q function."""

  def __init__(self, q_function, epsilon=0.0, bolzmann=False, bolzmann_temp=1.0):
    self.q_function = q_function
    self.epsilon = epsilon
    self.bolzmann=bolzmann
    self.bolzmann_temp=bolzmann_temp

  def __call__(self, observations):
    if self.bolzmann:
      return self.q_function.act_bolzmann(observations, self.bolzmann_temp)
    else:
      return self.q_function.act_greedy(observations, self.epsilon)


class MetaQ(object):
  """Meta Q learning class that owns the graph."""

  config = None
  q_function_forward = None
  parameter_variable = None
  graph = None
  graph_built = False
  session = None

  @classmethod
  def get_default_config(self):
    return tf.HParams(
        # General
        observation_shape=[None],
        action_dim=None,
        discount_factor=0.9,
        inner_loop_gradient_clipping=0.0,
        outer_loop_gradient_clipping=0.0,
        inner_loop_learning_rate=0.1,
        outer_loop_learning_rate=0.001,

        # Q learning specific
        policy_gradient_outer_loop=False,
        inner_loop_residual_gradient=False,
        outer_loop_residual_gradient=False,
        inner_loop_q_loss_type='l2',
        outer_loop_q_loss_type='l2',
        inner_loop_soft_q=False,
        outer_loop_soft_q=False,
        inner_loop_soft_q_temperature=1.0,
        outer_loop_soft_q_temperature=1.0,
        inner_loop_online_target=True,
        inner_loop_double_q=False,

        # MAML specific
        inner_loop_steps=1,
        inner_loop_stop_gradient=False,
        inner_loop_optimizer='sgd',
        outer_loop_optimizer='momentum',
        outer_loop_optimizer_first_momentum=0.9,
        outer_loop_optimizer_second_momentum=0.999,

        # Debug options
        debug_inner_loop_ground_truth=False,
        debug_outer_loop_ground_truth=False,
    )

  def __init__(self, config, q_function_forward):
    self.config = MetaQ.get_default_config()
    self.config.override_from_dict(config.values())
    self.q_function_forward = q_function_forward
    self.graph = utils.GraphCollection()

  def init_session(self, session):
    self.session = session

  def _build_placeholders(self):
    config = self.config
    graph = self.graph
    # Pre-update data
    graph.pre_update_observations = tf.placeholder(
        dtype=tf.float32, shape=config.observation_shape,
        name='pre_update_observations',
    )
    graph.pre_update_next_observations = tf.placeholder(
        dtype=tf.float32, shape=config.observation_shape,
        name='pre_update_next_observations',
    )
    graph.pre_update_actions = tf.placeholder(
        dtype=tf.int64, shape=[None],
        name='pre_update_actions',
    )
    graph.pre_update_rewards = tf.placeholder(
        dtype=tf.float32, shape=[None],
        name='pre_update_rewards',
    )
    graph.pre_update_dones = tf.placeholder(
        dtype=tf.bool, shape=[None],
        name='pre_update_dones',
    )

    # Post update data
    graph.post_update_observations = tf.placeholder(
        dtype=tf.float32, shape=config.observation_shape,
        name='post_update_observations',
    )
    graph.post_update_next_observations = tf.placeholder(
        dtype=tf.float32, shape=config.observation_shape,
        name='post_update_next_observations',
    )
    graph.post_update_actions = tf.placeholder(
        dtype=tf.int64, shape=[None],
        name='post_update_actions',
    )
    graph.post_update_rewards = tf.placeholder(
        dtype=tf.float32, shape=[None],
        name='post_update_rewards',
    )
    graph.post_update_dones = tf.placeholder(
        dtype=tf.bool, shape=[None],
        name='post_update_dones',
    )
    graph.post_update_advantage = tf.placeholder(
        dtype=tf.float32, shape=[None],
        name='post_update_advantage',
    )

  def forward_q_function(self, observations, weights):
    with self.parameterized_model.build_parameterized(weights):
      q_values = self.q_function_forward(observations)

    return q_values

  def _inner_loop_bellman_error(self, weights, target_weights,
                                return_prediction=False):
    config = self.config
    graph = self.graph
    predicted_q_current = self.forward_q_function(
        graph.pre_update_observations, weights
    )
    target_predicted_q_next = self.forward_q_function(
        graph.pre_update_next_observations, target_weights
    )
    predicted_q_next = self.forward_q_function(
        graph.pre_update_next_observations, weights
    )
    inner_bellman_error, inner_target_value = bellman_error(
        predicted_q_current=predicted_q_current,
        current_actions=graph.pre_update_actions,
        rewards=graph.pre_update_rewards,
        target_predicted_q_next=target_predicted_q_next,
        discount_factor=config.discount_factor,
        dones=graph.pre_update_dones,
        predicted_q_next=predicted_q_next,
        online_target=config.inner_loop_online_target,
        double_q=config.inner_loop_double_q,
        loss_type=config.inner_loop_q_loss_type,
        residue_gradient=config.inner_loop_residual_gradient,
        soft_q=config.inner_loop_soft_q,
        soft_q_temperature=config.inner_loop_soft_q_temperature,
        debug_ground_truth=config.debug_inner_loop_ground_truth
    )
    if not return_prediction:
      return inner_bellman_error, inner_target_value
    else:
      # Reuse the pre-update graph for q function evaluation
      return inner_bellman_error, inner_target_value, predicted_q_current


  def _outer_loop_bellman_error(self, weights):
    config = self.config
    graph = self.graph
    predicted_q_current = self.forward_q_function(
        graph.post_update_observations, weights
    )
    predicted_q_next = self.forward_q_function(
        graph.post_update_next_observations, weights
    )
    target_predicted_q_next = predicted_q_next
    return bellman_error(
        predicted_q_current=predicted_q_current,
        current_actions=graph.post_update_actions,
        rewards=graph.post_update_rewards,
        target_predicted_q_next=target_predicted_q_next,
        discount_factor=config.discount_factor,
        dones=graph.post_update_dones,
        predicted_q_next=predicted_q_next,
        online_target=False,
        double_q=False,
        loss_type=config.outer_loop_q_loss_type,
        residue_gradient=config.outer_loop_residual_gradient,
        soft_q=config.outer_loop_soft_q,
        soft_q_temperature=config.outer_loop_soft_q_temperature,
        debug_ground_truth=config.debug_outer_loop_ground_truth
    )

  def _outer_loop_policy_gradient_loss(self, weights):
    config = self.config
    graph = self.graph
    q_values = self.forward_q_function(
        graph.post_update_observations, weights
    )
    log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=q_values, labels=graph.post_update_actions
    )
    return -graph.post_update_advantage * log_probs

  def _outer_loop_loss(self, weights):
    config = self.config
    if config.policy_gradient_outer_loop:
      return self._outer_loop_policy_gradient_loss(weights), None
    else:
      return self._outer_loop_bellman_error(weights)

  def _outer_loop_optimizer(self):
    config = self.config
    if config.outer_loop_optimizer == 'sgd':
      return tf.train.GradientDescentOptimizer(config.outer_loop_learning_rate)
    elif config.outer_loop_optimizer == 'momentum':
      return tf.train.MomentumOptimizer(
          config.outer_loop_learning_rate,
          config.outer_loop_optimizer_first_momentum
      )
    elif config.outer_loop_optimizer == 'adam':
      return tf.train.AdamOptimizer(
          config.outer_loop_learning_rate,
          config.outer_loop_optimizer_first_momentum,
          config.outer_loop_optimizer_second_momentum
      )
    else:
      raise ValueError('Unsupported optimizer')

  def build_graph(self):
    assert not self.graph_built
    self.graph_built = True

    config = self.config
    graph = self.graph

    self._build_placeholders()

    self.parameterized_model = ParameterizedModel('meta_dqn')

    with self.parameterized_model.build_template():
      self.q_function_forward(graph.pre_update_observations)

    self.weight_variable = self.parameterized_model.parameter
    graph.weight_variable = self.weight_variable


    # Build inner loop
    target_weights = self.weight_variable
    updated_weights = self.weight_variable

    graph.pre_update_loss, _, graph.pre_update_q_values = self._inner_loop_bellman_error(
        updated_weights, target_weights, return_prediction=True
    )

    graph.inner_loop_loses = []
    graph.inner_loop_gradients = []
    graph.inner_loop_weights = []

    for step in range(config.inner_loop_steps):
      loss, target_value = self._inner_loop_bellman_error(
          updated_weights, target_weights
      )

      if config.inner_loop_residual_gradient:
        stop_gradients = None
      else:
        stop_gradients = target_value

      grad = tf.gradients(
          loss, updated_weights, stop_gradients=stop_gradients
      )[0]

      if config.inner_loop_stop_gradient:
        grad = tf.stop_gradient(grad)

      if config.inner_loop_gradient_clipping > 0.0:
        grad = tf.clip_by_norm(grad, config.inner_loop_gradient_clipping)

      updated_weights = updated_weights - config.inner_loop_learning_rate * grad

      graph.inner_loop_loses.append(loss)
      graph.inner_loop_gradients.append(grad)
      graph.inner_loop_weights.append(updated_weights)

    graph.post_update_weights = updated_weights
    graph.post_update_loss, stop_gradients = self._outer_loop_loss(updated_weights)

    if config.outer_loop_residual_gradient:
      stop_gradients = None

    graph.post_update_gradient = tf.gradients(
        graph.post_update_loss, self.weight_variable,
        stop_gradients=stop_gradients
    )[0]

    graph.train_accumulator = utils.TensorAverageAccumulatorList(
        [graph.pre_update_loss, graph.post_update_loss,
         graph.post_update_gradient]
    )

    graph.accumulate_gradient_op = graph.train_accumulator.accum_op
    graph.clear_gradient_op = graph.train_accumulator.zeros_op

    graph.mean_pre_update_loss, graph.mean_post_update_loss, graph.mean_post_update_gradient = (
        graph.train_accumulator.values
    )

    graph.optimizer = self._outer_loop_optimizer()

    graph.train_op = graph.optimizer.apply_gradients(
        [(graph.mean_post_update_gradient, self.weight_variable)]
    )

  def _inner_loop_feed_dict(self, observations, actions, rewards,
                            next_obervations, dones):
    graph = self.graph
    return {
        graph.pre_update_observations: observations,
        graph.pre_update_next_observations: next_obervations,
        graph.pre_update_actions: actions,
        graph.pre_update_rewards: rewards,
        graph.pre_update_dones: dones
    }

  def _outer_loop_feed_dict(self, observations, actions, rewards,
                            next_obervations, dones, post_update_advantage=None):
    graph = self.graph
    feed_dict = {
        graph.post_update_observations: observations,
        graph.post_update_next_observations: next_obervations,
        graph.post_update_actions: actions,
        graph.post_update_rewards: rewards,
        graph.post_update_dones: dones
    }

    if post_update_advantage is not None:
      feed_dict[graph.post_update_advantage] = post_update_advantage
    return feed_dict

  def get_current_weights(self):
    assert self.session is not None
    return self.session.run(self.weight_variable)

  def get_pre_update_q_function(self):
    graph = self.graph
    weights = self.get_current_weights()
    return QFunction(
        self.session, graph.pre_update_observations, graph.pre_update_q_values,
        self.weight_variable, weights
    )

  def get_post_update_q_function(self, observations, actions, rewards,
                                 next_obervations, dones):
    assert self.session is not None
    graph = self.graph

    updated_weights = self.session.run(
        graph.post_update_weights,
        self._inner_loop_feed_dict(
            observations, actions, rewards, next_obervations, dones
        )
    )
    return QFunction(
        self.session, graph.pre_update_observations, graph.pre_update_q_values,
        self.weight_variable, updated_weights
    )

  def clear_gradient(self):
    graph = self.graph
    self.session.run(graph.clear_gradient_op)

  def accumulate_gradient(self, pre_update_observations,
                          pre_update_actions, pre_update_rewards,
                          pre_update_next_obervations, pre_update_dones,
                          post_update_observations, post_update_actions,
                          post_update_rewards, post_update_next_obervations,
                          post_update_dones,
                          post_update_advantage=None):

    inner_loop_feed_dict = self._inner_loop_feed_dict(
        pre_update_observations, pre_update_actions, pre_update_rewards,
        pre_update_next_obervations, pre_update_dones
    )
    outer_loop_feed_dict = self._outer_loop_feed_dict(
        post_update_observations, post_update_actions, post_update_rewards,
        post_update_next_obervations, post_update_dones,
        post_update_advantage
    )
    feed_dict = outer_loop_feed_dict
    feed_dict.update(inner_loop_feed_dict)

    graph = self.graph

    pre_update_loss, post_update_loss, _ = self.session.run(
        [graph.pre_update_loss, graph.post_update_loss, graph.accumulate_gradient_op],
        feed_dict
    )
    return pre_update_loss, post_update_loss

  def get_mean_loss(self):
    graph = self.graph
    return self.session.run([graph.mean_pre_update_loss, graph.mean_post_update_loss])

  def run_train_step(self):
    graph = self.graph
    pre_update_loss, post_update_loss, _ = self.session.run(
        [graph.mean_pre_update_loss, graph.mean_post_update_loss,
         graph.train_op]
    )
    self.clear_gradient()   # Clear gradient after training.
    return pre_update_loss, post_update_loss

