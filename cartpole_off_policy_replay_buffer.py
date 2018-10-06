"""Supervised meta learning of random sinusoid function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy

import math
import os
import pickle
import numpy as np
import tensorflow.google as tf
import time

from google3.experimental.brain.meta_value.envs.cartpole import cartpole
from google3.experimental.brain.meta_value.envs.cartpole import cartpole_utils
from google3.experimental.brain.meta_value.meta_q.meta_q import MetaQ
from google3.experimental.brain.meta_value.meta_q.meta_q import QPolicy
from google3.experimental.brain.meta_value.meta_q.multitask_replay_buffer import MultiTaskReplayBuffer
from google3.experimental.brain.meta_value.utils.misc_utils import define_flags_with_default
from google3.experimental.brain.meta_value.utils.tensorboard_logger import TensorBoardLogger

from google3.pyglib import gfile

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
    video_report_steps=1000,
    vizier_objective='greedy_reward',

    # General
    output_dir='/tmp/meta_q_cartpole',
    random_seed=42,

    # Network specific
    nn_arch='512-512-512',
    activation='leaky_relu',

    outer_loop_steps=3000,

    # Env specific
    n_meta_tasks=8,
    inner_loop_n_states=1000,
    outer_loop_n_states=1000,
    inner_loop_n_trajs=50,
    outer_loop_n_trajs=50,
    outer_loop_greedy_eval_n_trajs=100,

    inner_loop_data_collection='epsilon_greedy',
    inner_loop_greedy_epsilon=0.2,
    inner_loop_bolzmann_temp=1.0,

    outer_loop_data_collection='epsilon_greedy',
    outer_loop_greedy_epsilon=0.5,
    outer_loop_bolzmann_temp=1.0,

    fixed_env=False,

    # MetaQ specific
    discount_factor=0.9,
    inner_loop_gradient_clipping=1.0,
    outer_loop_gradient_clipping=0.0,
    inner_loop_learning_rate=0.00249242,
    outer_loop_learning_rate=0.00127288,

    # policy_gradient_outer_loop=False,  # Currently not supported!
    inner_loop_residual_gradient=True,
    outer_loop_residual_gradient=True,
    inner_loop_q_loss_type='l2',
    outer_loop_q_loss_type='l2',
    inner_loop_soft_q=False,
    outer_loop_soft_q=True,
    inner_loop_soft_q_temperature=1.0,
    outer_loop_soft_q_temperature=114.479,
    inner_loop_online_target=False,
    inner_loop_double_q=False,

    inner_loop_steps=3,
    inner_loop_stop_gradient=False,
    inner_loop_optimizer='sgd',
    outer_loop_optimizer='adam',
    outer_loop_optimizer_first_momentum=0.9,
    outer_loop_optimizer_second_momentum=0.999,

    debug_inner_loop_ground_truth=False,
    debug_outer_loop_ground_truth=False,
    create_video_locally=False,
    on_policy_steps=200,
    weight_rewards=True,
    reweight_rewards=0.0,
    target_update_freq=200,
    outer_loop_online_target=False,
    outer_loop_double_q=False,
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


def evaluate(policy, env, meta_q, inner_loop_n_trajs=8, outer_loop_n_trajs=8,
             n=21, weight_rewards=True, video_data=None):
  # Create 21 evaluation tasks between min_goal_x and max_goal_x.
  spacing = (env.env.max_goal_x - env.env.min_goal_x) / (n - 1)
  goal_positions = np.arange(start=env.env.min_goal_x,
                             stop=env.env.max_goal_x+0.01, step=spacing)
  post_update_greedy_rewards = []
  for i in range(0, len(goal_positions)):
    env.env.goal_x = goal_positions[i]
    (inner_observations, inner_actions, inner_rewards,
     inner_next_observations, inner_dones) = cartpole_utils.collect_data(
         env, n_trajs=inner_loop_n_trajs, policy=policy)
    post_update_q_func = meta_q.get_post_update_q_function(
        inner_observations, inner_actions, inner_rewards,
        inner_next_observations, inner_dones
    )
    policy = QPolicy(post_update_q_func, epsilon=0.0)

    if video_data:
      video_data['filename'] = 'video_data' + str(i)
      _ = cartpole_utils.collect_data_old(env, n_trajs=1,
                                          policy=policy, video_data=video_data)
    _, _, greedy_rewards, _, _ = cartpole_utils.collect_data(
        env, n_trajs=outer_loop_n_trajs, policy=policy)

    average_reward = np.sum(greedy_rewards) / outer_loop_n_trajs
    if weight_rewards:  # Weights the reward by the difficulty of the task
      average_reward = average_reward * 0.25 * math.exp(abs(goal_positions[i]))
    post_update_greedy_rewards.append(average_reward)
  return post_update_greedy_rewards


def collect_off_policy_data(env, goal_positions, meta_q, post_update_q_func, buffers, num_trajs,
                            data_collection, greedy_epsilon, bolzmann_temp,
                            collect_from_fine_tune=True):
  # set n_trajs to 8
  if post_update_q_func is None:
    policy = None
  elif data_collection == 'random':
    policy = None
  else:
    if data_collection == 'epsilon_greedy':
      policy = QPolicy(post_update_q_func, epsilon=greedy_epsilon)
    elif data_collection == 'bolzmann':
      policy = QPolicy(post_update_q_func, bolzmann=True,
                       bolzmann_temp=bolzmann_temp)

  for task_id in range(0, len(goal_positions)):
    env.env.goal_x = goal_positions[task_id]
    env.env.task_id = task_id
    data = cartpole_utils.collect_data(env, n_trajs=num_trajs, policy=policy)
    # print(data[0])
    buffers.add(task_id, *data)

    if collect_from_fine_tune:
      fine_tuned_q_func = meta_q.get_post_update_q_function(*data)
      fine_tuned_policy = QPolicy(fine_tuned_q_func, epsilon=0.0)
      fine_tuned_data = cartpole_utils.collect_data(env, n_trajs=num_trajs,
                                                    policy=fine_tuned_policy)
      buffers.add(task_id, *fine_tuned_data)
  return buffers


def run_experiment(study_hparams=None, trial_handle=None, tuner=None):

  FLAGS = deepcopy(tf.app.flags.FLAGS)

  if FLAGS.use_vizier:
    for key, val in study_hparams.values().items():
      setattr(FLAGS, key, val)

  tf.reset_default_graph()
  np.random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)

  # Initialize env

  env_kwargs = {
      'goal_x': FLAGS.goal_x,
      'min_goal_x': FLAGS.min_goal_x,
      'max_goal_x': FLAGS.max_goal_x,
      'x_threshold': FLAGS.x_threshold,
      'max_reward_for_dist': FLAGS.max_reward_for_dist,
      'reward_per_time_step': FLAGS.reward_per_time_step,
      'fixed_initial_state': FLAGS.fixed_initial_state,
      'reweight_rewards': FLAGS.reweight_rewards
  }
  env = cartpole.make_env(env_kwargs)
  eval_env = cartpole.make_env(env_kwargs)

  if not FLAGS.fixed_env:
    env.env.randomize()

  if trial_handle:
    tensorboard_path = os.path.join(FLAGS.output_dir, trial_handle)
  else:
    tensorboard_path = FLAGS.output_dir
  tf.gfile.MakeDirs(tensorboard_path)

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

    inner_loop_buffer = MultiTaskReplayBuffer(len(env.env.goal_positions), 200000, FLAGS.random_seed)
    outer_loop_buffer = MultiTaskReplayBuffer(len(env.env.goal_positions), 200000, FLAGS.random_seed)

    pre_update_rewards = []
    post_update_rewards = []
    post_update_greedy_rewards = []
    post_update_q_func = None
    for outer_step in range(FLAGS.outer_loop_steps):
      print('State is ', env.env.state)
      if outer_step % FLAGS.on_policy_steps == 0:
        if FLAGS.fixed_env:
          goal_positions = [env.env.goal_x]
        else:
          goal_positions = env.env.goal_positions
        # NOTE: Approximately ~30 to 60 states per trajectory
        inner_loop_buffer = collect_off_policy_data(
            env, goal_positions, meta_q, post_update_q_func, inner_loop_buffer,
            FLAGS.inner_loop_n_trajs, FLAGS.inner_loop_data_collection,
            FLAGS.inner_loop_greedy_epsilon, FLAGS.inner_loop_bolzmann_temp)
        outer_loop_buffer = collect_off_policy_data(
            env, goal_positions, meta_q, post_update_q_func, outer_loop_buffer,
            FLAGS.outer_loop_n_trajs, FLAGS.outer_loop_data_collection,
            FLAGS.outer_loop_greedy_epsilon, FLAGS.outer_loop_bolzmann_temp)

      post_update_greedy_rewards = []

      finetuned_policy = None
      for task_id in range(FLAGS.n_meta_tasks):
        # print('Task: {}'.format(task_id))

        if not FLAGS.fixed_env:
          env.env.randomize()

        (inner_observations, inner_actions, inner_rewards,
         inner_next_observations, inner_dones) = inner_loop_buffer.sample(
             env.env.task_id, FLAGS.inner_loop_n_states)
        # Evaluating true rewards
        post_update_q_func = meta_q.get_post_update_q_function(
            inner_observations, inner_actions, inner_rewards,
            inner_next_observations, inner_dones
        )

        policy = QPolicy(post_update_q_func, epsilon=0.0)

        if outer_step % FLAGS.report_steps == 0 or outer_step >= (FLAGS.outer_loop_steps - 1):
          _, _, greedy_rewards, _, _ = cartpole_utils.collect_data(
              env, n_trajs=FLAGS.outer_loop_greedy_eval_n_trajs, policy=policy
          )
          post_update_greedy_rewards.append(
              np.sum(greedy_rewards) / FLAGS.outer_loop_greedy_eval_n_trajs
          )

        finetuned_policy = policy

        (outer_observations, outer_actions, outer_rewards,
         outer_next_observations, outer_dones) = outer_loop_buffer.sample(
                env.env.task_id, FLAGS.outer_loop_n_states)
        meta_q.accumulate_gradient(
            inner_observations, inner_actions, inner_rewards,
            inner_next_observations, inner_dones,
            outer_observations, outer_actions, outer_rewards,
            outer_next_observations, outer_dones,
        )

      pre_update_loss, post_update_loss = meta_q.run_train_step()

      if not FLAGS.outer_loop_online_target and outer_step % FLAGS.target_update_freq == 0:
        print("updating target network")
        meta_q.update_target_network()

      log_data = dict(
          pre_update_loss=pre_update_loss,
          post_update_loss=post_update_loss,
          goal_x=env.env.goal_x,
      )

      #TODO(hkannan): uncomment this later!!!
      if outer_step % FLAGS.report_steps == 0 or outer_step >= (FLAGS.outer_loop_steps - 1):
        # reward_across_20_tasks = evaluate(
        #     policy, eval_env, meta_q,
        #     inner_loop_n_trajs=FLAGS.inner_loop_n_trajs,
        #     outer_loop_n_trajs=FLAGS.outer_loop_n_trajs, n=21,
        #     weight_rewards=FLAGS.weight_rewards)
        # log_data['reward_mean'] = np.mean(reward_across_20_tasks)
        # log_data['reward_variance'] = np.var(reward_across_20_tasks)
        log_data['post_update_greedy_reward'] = np.mean(post_update_greedy_rewards)
        log_data['post_update_greedy_reward_variance'] = np.var(post_update_greedy_rewards)

      print('Outer step: {}, '.format(outer_step), log_data)
      logger.log_dict(outer_step, log_data)
      # if outer_step % FLAGS.video_report_steps == 0 or outer_step >= (FLAGS.outer_loop_steps - 1):
      #   video_data = {
      #       'env_kwargs': env_kwargs,
      #       'inner_loop_data_collection': FLAGS.inner_loop_data_collection,
      #       'inner_loop_greedy_epsilon': FLAGS.inner_loop_greedy_epsilon,
      #       'inner_loop_bolzmann_temp': FLAGS.inner_loop_bolzmann_temp,
      #       'inner_loop_n_trajs': FLAGS.inner_loop_n_trajs,
      #       'meta_q_kwargs': kwargs,
      #       'weights': meta_q.get_current_weights(),
      #       'tensorboard_path': tensorboard_path,
      #       'filename': 'random_task'
      #   }
      #   reward_across_20_tasks = evaluate(
      #       policy, eval_env, meta_q,
      #       inner_loop_n_trajs=FLAGS.inner_loop_n_trajs,
      #       outer_loop_n_trajs=FLAGS.outer_loop_n_trajs, n=21,
      #       weight_rewards=FLAGS.weight_rewards, video_data=video_data)
      #   log_data['reward_mean'] = np.mean(reward_across_20_tasks)
      #   log_data['reward_variance'] = np.var(reward_across_20_tasks)
      #   logger.log_dict(outer_step, log_data)

      if outer_step >= (FLAGS.outer_loop_steps - 1):
        greedy_reward_path = os.path.join(tensorboard_path, 'reward')
        with gfile.Open(greedy_reward_path, mode='wb') as f:
          f.write(pickle.dumps(log_data['post_update_greedy_reward']))
      if FLAGS.use_vizier:
        for v in log_data.values():
          if not np.isfinite(v):
            tuner.report_done(infeasible=True,
                              infeasible_reason='Nan or inf encountered')
            return

        if outer_step % FLAGS.report_steps == 0 or outer_step >= (FLAGS.outer_loop_steps - 1):
          if FLAGS.vizier_objective == 'greedy_reward':
            objective_value = log_data['post_update_greedy_reward']
          elif FLAGS.vizier_objective == 'loss':
            objective_value = post_update_loss
          elif FLAGS.vizier_objective == 'reward':
            objective_value = log_data['reward_mean']
          else:
            raise ValueError('Unsupported vizier objective!')
          tuner.report_measure(
              objective_value=objective_value,
              global_step=outer_step,
              metrics=log_data
          )

  if FLAGS.use_vizier:
    tuner.report_done()
