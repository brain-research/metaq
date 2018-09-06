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

"""Util functions for cartpole."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def make_video_from_images(rendered_images, render_path, goal_pos, reward):
  print('Number of images ', len(rendered_images))
  animation_writer = animation.writers['ffmpeg']
  writer = animation_writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
  fig = plt.figure()
  fig, ax = plt.subplots(figsize=(5, 8))
  im = ax.imshow(rendered_images[0])
  number_of_steps = len(rendered_images)
  def update(i):
    # print(i)
    im = ax.imshow(rendered_images[i])

    ax.set_title(
        'Goal position: {0:.2f}\nNumber of steps: {1}\nReward: {2:.2f}'.format(
            goal_pos, number_of_steps, reward), fontsize=20)
    ax.set_axis_off()
    return im

  if len(rendered_images) <= 30:
    step_size = 1
  else:
    step_size = 5
  im_ani = animation.FuncAnimation(fig, update,
                                   frames=np.arange(1, len(rendered_images),
                                                    step_size),
                                   interval=50, repeat_delay=3000)
  im_ani.save(render_path, writer=writer)


def collect_data(env, time_steps=None, n_trajs=None, policy=None,
                 render_path=None, state_path=None):
  assert not (time_steps is None and n_trajs is None)

  observations = []
  next_observations = []
  rewards = []
  actions = []
  dones = []

  n_actions = env.action_space.n

  trajs = 0
  step = 0

  rendered_images = []
  obs = env.reset()
  if state_path:
    print('env.env.state is ')
    print(env.env.state)
    with tf.gfile.Open(state_path, mode='wb') as f:
      f.write(pickle.dumps(env.env.state))

  while True:
    if policy is None:
      # Random data collection
      action = np.random.choice(n_actions)
    else:
      action = policy(obs.reshape(1, -1))[0]

    next_obs, reward, done, _ = env.step(action)

    observations.append(obs)
    next_observations.append(next_obs)
    actions.append(action)
    rewards.append(reward)
    dones.append(done)
    if render_path:
      rendered_images.append(env.env.render(mode='rgb_array').copy())
    if done:
      obs = env.reset()
      # raw_input()
      trajs += 1
    else:
      obs = next_obs

    step += 1

    if n_trajs is not None:
      if trajs >= n_trajs:
        break
    else:
      if step >= time_steps:
        break

  actions = np.array(actions, dtype=np.int64)
  rewards = np.array(rewards, dtype=np.float32)
  dones = np.array(dones, dtype=np.bool)

  if render_path is not None:
    print('goal is ')
    print(env.env.goal_x)
    make_video_from_images(rendered_images, render_path, env.env.goal_x,
                           np.sum(rewards))

  return observations, actions, rewards, next_observations, dones
