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

"""Supervised meta learning of random sinusoid function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy

import os
import pickle
import numpy as np
import tensorflow as tf

import cartpole
import cartpole_utils
from library.meta_q import MetaQ
from library.meta_q import QPolicy
from misc_utils import define_flags_with_default
from misc_utils import TensorBoardLogger

flags_def = define_flags_with_default(

    goal_x=0.0,
    min_goal_x=-4,
    max_goal_x=4,
    x_threshold=6,
    max_reward_for_dist=1.0,
    reward_per_time_step=0.0,
    fixed_initial_state=False,

    use_vizier=True,  # Turn this flag off to run locally
    report_steps=100,
    vizier_objective='reward',

    # General
    output_dir='/tmp/meta_q_cartpole',
    random_seed=42,

    # Network specific
    nn_arch='512-512-512',
    activation='relu',

    outer_loop_steps=3000,

    # Env specific
    n_meta_tasks=8,
    inner_loop_n_trajs=8,
    outer_loop_n_trajs=8,
    outer_loop_greedy_eval_n_trajs=100,

    inner_loop_data_collection='epsilon_greedy',
    inner_loop_greedy_epsilon=0.1,
    inner_loop_bolzmann_temp=1.0,

    outer_loop_data_collection='epsilon_greedy',
    outer_loop_greedy_epsilon=0.1,
    outer_loop_bolzmann_temp=1.0,

    fixed_env=False,

    # MetaQ specific
    discount_factor=0.9,
    inner_loop_gradient_clipping=0.0,
    outer_loop_gradient_clipping=0.0,
    inner_loop_learning_rate=0.1,
    outer_loop_learning_rate=0.001,

    # policy_gradient_outer_loop=False,  # Currently not supported!
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

    inner_loop_steps=1,
    inner_loop_stop_gradient=False,
    inner_loop_optimizer='sgd',
    outer_loop_optimizer='momentum',
    outer_loop_optimizer_first_momentum=0.9,
    outer_loop_optimizer_second_momentum=0.999,

    debug_inner_loop_ground_truth=False,
    debug_outer_loop_ground_truth=False,
    create_video_locally=False,
    use_hparam_search=False,
)


def fully_connected_net(nn_arch='512-512-512', activation='relu'):
  activation_function = {
      'relu': tf.nn.relu,
      'leaky_relu': tf.nn.leaky_relu,
      'selu': tf.nn.selu,
      'elu': tf.nn.elu,
  }[activation]

  def forward(observation):
    N_ACTIONS = 2
    if nn_arch == '':
      hidden_dims = []
    else:
      hidden_dims = [int(x) for x in nn_arch.split('-')]
    x = observation
    for hd in hidden_dims:
      x = tf.layers.dense(x, hd)
      x = activation_function(x)

    x = tf.layers.dense(x, N_ACTIONS)
    return x
  return forward


def run_experiment(study_hparams=None, trial_handle=None):

  FLAGS = deepcopy(tf.app.flags.FLAGS)

  if FLAGS.use_hparam_search:
    for key, val in study_hparams.values().items():
      setattr(FLAGS, key, val)

  tf.reset_default_graph()
  np.random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)

  # Initialize env

  kwargs = {
      'goal_x': FLAGS.goal_x,
      'min_goal_x': FLAGS.min_goal_x,
      'max_goal_x': FLAGS.max_goal_x,
      'x_threshold': FLAGS.x_threshold,
      'max_reward_for_dist': FLAGS.max_reward_for_dist,
      'reward_per_time_step': FLAGS.reward_per_time_step,
      'fixed_initial_state': FLAGS.fixed_initial_state,
  }
  env = cartpole.make_env(kwargs)

  if not FLAGS.fixed_env:
    env.env.randomize()

  if trial_handle:
    tensorboard_path = os.path.join(FLAGS.output_dir, trial_handle)
  else:
    tensorboard_path = FLAGS.output_dir
  tf.gfile.MakeDirs(tensorboard_path)

  with tf.gfile.Open(os.path.join(tensorboard_path, 'env_args'), mode='wb') as f:
    f.write(pickle.dumps(kwargs))

  kwargs = dict(
      observation_shape=[None] + list(env.observation_space.shape),
      action_dim=1
  )
  default_hps = MetaQ.get_default_config().values()

  for key in flags_def:
    if key in default_hps:
      kwargs[key] = getattr(FLAGS, key)

  hps = tf.HParams(**kwargs)

  meta_q = MetaQ(hps, fully_connected_net(FLAGS.nn_arch, FLAGS.activation))
  meta_q.build_graph()

  init_op = tf.global_variables_initializer()

  logger = TensorBoardLogger(tensorboard_path)

  with tf.Session() as sess:
    sess.run(init_op)
    meta_q.init_session(sess)

    for outer_step in range(FLAGS.outer_loop_steps):

      pre_update_rewards = []
      post_update_rewards = []
      post_update_greedy_rewards = []

      finetuned_policy = None
      for task_id in range(FLAGS.n_meta_tasks):
        print('Task: {}'.format(task_id))

        if not FLAGS.fixed_env:
          env.env.randomize()

        # Inner loop
        if FLAGS.inner_loop_data_collection == 'random':
          policy = None
        else:
          q_func = meta_q.get_pre_update_q_function()
          if FLAGS.inner_loop_data_collection == 'epsilon_greedy':
            policy = QPolicy(q_func, epsilon=FLAGS.inner_loop_greedy_epsilon)
          elif FLAGS.inner_loop_data_collection == 'bolzmann':
            policy = QPolicy(
                q_func, bolzmann=True,
                bolzmann_temp=FLAGS.inner_loop_bolzmann_temp
            )
        (inner_observations, inner_actions, inner_rewards,
         inner_next_observations, inner_dones) = cartpole_utils.collect_data(
             env, n_trajs=FLAGS.inner_loop_n_trajs, policy=policy)

        pre_update_rewards.append(np.sum(
            inner_rewards) / FLAGS.inner_loop_n_trajs)

        # Evaluating true rewards
        post_update_q_func = meta_q.get_post_update_q_function(
            inner_observations, inner_actions, inner_rewards,
            inner_next_observations, inner_dones
        )

        policy = QPolicy(post_update_q_func, epsilon=0.0)
        _, _, greedy_rewards, _, _ = cartpole_utils.collect_data(
            env, n_trajs=FLAGS.outer_loop_greedy_eval_n_trajs, policy=policy
        )
        post_update_greedy_rewards.append(
            np.sum(greedy_rewards) / FLAGS.outer_loop_greedy_eval_n_trajs
        )

        finetuned_policy = policy
        # Outer loop
        if FLAGS.outer_loop_data_collection == 'random':
          policy = None
        else:
          if FLAGS.outer_loop_data_collection == 'epsilon_greedy':
            policy = QPolicy(
                post_update_q_func, epsilon=FLAGS.outer_loop_greedy_epsilon
            )
          elif FLAGS.outer_loop_data_collection == 'bolzmann':
            policy = QPolicy(
                post_update_q_func, bolzmann=True,
                bolzmann_temp=FLAGS.outer_loop_bolzmann_temp
            )
        (outer_observations, outer_actions, outer_rewards,
         outer_next_observations, outer_dones) = cartpole_utils.collect_data(
             env, n_trajs=FLAGS.outer_loop_n_trajs, policy=policy)

        post_update_rewards.append(np.sum(
            outer_rewards) / FLAGS.outer_loop_n_trajs)

        meta_q.accumulate_gradient(
            inner_observations, inner_actions, inner_rewards,
            inner_next_observations, inner_dones,
            outer_observations, outer_actions, outer_rewards,
            outer_next_observations, outer_dones,
        )

      pre_update_loss, post_update_loss = meta_q.run_train_step()
      pre_update_reward = np.mean(pre_update_rewards)
      post_update_reward = np.mean(post_update_rewards)
      post_update_greedy_reward = np.mean(post_update_greedy_rewards)
      post_update_greedy_reward_variance = np.var(post_update_greedy_rewards)

      log_data = dict(
          pre_update_loss=pre_update_loss,
          post_update_loss=post_update_loss,
          pre_update_reward=pre_update_reward,
          post_update_reward=post_update_reward,
          post_update_greedy_reward=post_update_greedy_reward,
          post_update_greedy_reward_variance=post_update_greedy_reward_variance,
          goal_x=env.env.goal_x,
      )
      print('Outer step: {}, '.format(outer_step), log_data)
      logger.log_dict(outer_step, log_data)

    if FLAGS.create_video_locally:
      video_path = os.path.join(tensorboard_path, 'video.mp4')
    else:
      video_path = None

    observations, actions, rewards, next_observations, dones = cartpole_utils.collect_data(
        env, time_steps=1000, n_trajs=1, policy=finetuned_policy, render_path=video_path,
        state_path=os.path.join(tensorboard_path, 'state'))

    print('Number of actions: ', len(actions))
    print('Total reward: ', np.sum(rewards))
    print('Dones: ', dones)
    with tf.gfile.Open(os.path.join(tensorboard_path, 'actions'), mode='wb') as f:
      f.write(pickle.dumps(actions))

    with tf.gfile.Open(os.path.join(tensorboard_path, 'dones'), mode='wb') as f:
      f.write(pickle.dumps(dones))

    with tf.gfile.Open(os.path.join(tensorboard_path, 'goal'), mode='wb') as f:
      f.write(pickle.dumps(env.env.goal_x))

