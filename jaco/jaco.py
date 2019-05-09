# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Jaco arm domain."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import os.path
# Internal dependencies.

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.utils import randomizers

from lxml import etree
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import ipdb

SUITE = containers.TaggedTasks()

_DEFAULT_TIME_LIMIT = 10
_ACTION_COST_D = 0.0025

# default 0.002
_CONTROL_TIMESTEP = 0.01

# default 1, 500/s
_N_SUB_STEPS = 50

_HOME_POSE = [275.35, 167.43, 57.49, 240.86, 82.70, 75.72, 0, 0, 0]
_JOINT_H = [10000, 313, 341, 10000, 10000, 10000, 0, 0, 0]
_JOINT_L = [-10000, 47, 19, -10000, -10000, -10000, 0, 0, 0]

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return _make_model(), common.ASSETS

def real_to_sim(angles):
  zero_offset = np.array([-180, 270, 90, 180, 180, -90, 0, 0, 0])

  # correct for the different physical directions of a +theta
  # movement between mujoco
  directions = np.array([-1, 1, -1, -1, -1, -1, 1, 1, 1])

  # correct for the degrees -> radians shift going from arm
  # to mujoco
  scales = np.array([np.pi / 180] * 6 + [0.78 / 6800] * 3)

  return (angles - zero_offset) * directions * scales

def damping_with_noise(random):
  DAMPING = 40
  RANGE = 10
  return random.uniform(DAMPING-RANGE, DAMPING+RANGE, 6)

def position_penalty(physics, joints_z):
  _FLOOR_H = 0.1
  _HEIGHT_COST_D = 0.001
  pz = np.array(joints_z)
  pz = np.fmin(pz - _FLOOR_H, np.zeros(9))
  return _HEIGHT_COST_D * np.linalg.norm(pz)

def ef_pose_penalty(ef_angle):
  COST_D = 0.1
  cost = COST_D * np.arccos(np.dot(ef_angle, [0,0,-1]) / np.linalg.norm(ef_angle))
  return cost

@SUITE.add('benchmarking', 'easy')
def basic(time_limit=_DEFAULT_TIME_LIMIT, random=None):
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = JacoReacher(random=random)
  # return control.Environment(physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit)
  return control.Environment(physics, task, n_sub_steps=_N_SUB_STEPS, time_limit=time_limit)
  # return control.Environment(physics, task, time_limit=time_limit)

def _make_model():
  model_path = os.path.join(os.path.dirname( __file__ ), 'jaco_pos.xml')
  xml_string = common.read_model(model_path)
  mjcf = etree.fromstring(xml_string)
  return etree.tostring(mjcf, pretty_print=True)

class Physics(mujoco.Physics):

  def finger_to_target_distance(self):
    """Returns the distance from the tip to the target."""
    tip_to_target = (self.named.data.geom_xpos['target'] -
                     self.named.data.site_xpos['palm'])
    return np.linalg.norm(tip_to_target)

  def target_pos(self):
    return self.named.data.geom_xpos['target']

  def target_height(self):
    return self.named.data.geom_xpos['target', 'z']

  def finger_to_target(self):
    """Returns the distance from the tip to the target."""
    tip_to_target = (self.named.data.geom_xpos['target'] -
                     self.named.data.site_xpos['palm'])
    return tip_to_target

  def get_target(self):
    return self.named.data.geom_xpos['target']

  def move_hand(self, position):
    self.named.data.mocap_pos[0] = position

  # todo: extremely slow, should parallel, or manually compute
  def ground_penalty(self):
    joints_z = [
      self.named.data.geom_xpos['jaco_joint_1', 'z'],
      self.named.data.geom_xpos['jaco_joint_2', 'z'],
      self.named.data.geom_xpos['jaco_joint_3', 'z'],
      self.named.data.geom_xpos['jaco_joint_4', 'z'],
      self.named.data.geom_xpos['jaco_joint_5', 'z'],
      self.named.data.geom_xpos['jaco_joint_6', 'z'],
      self.named.data.geom_xpos['jaco_link_fingertip_1', 'z'],
      self.named.data.geom_xpos['jaco_link_fingertip_2', 'z'],
      self.named.data.geom_xpos['jaco_link_fingertip_3', 'z'],
    ]
    return position_penalty(self, joints_z)

  def pose_penalty(self):
    ef_angle = self.named.data.site_xmat['palm'].reshape(3,3).dot([0,0,-1])
    return ef_pose_penalty(ef_angle)

class JacoReacher(base.Task):

  def __init__(self, random=None):
    super(JacoReacher, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.
    Initializes the cart and pole according to `swing_up`, and in both cases
    adds a small random initial velocity to break symmetry.
    Args:
      physics: An instance of `Physics`.
    """
    # physics.named.model.geom_size['target', 0] = self._target_size
    # randomizers.randomize_limited_and_rotational_joints(physics, self.random)

    # for _ in range(100):
      # physics.step()

    physics.data.time = 0
    self._timeout_progress = 0

    # randomize target position
    angle = self.random.uniform(np.pi, 2 * np.pi)
    anglez = self.random.uniform(0, np.pi)
    radius = self.random.uniform(.35, .60)

    physics.named.data.qpos[['target_x', 'target_y']] = radius * np.cos(angle), radius * np.sin(angle)

    home_noise = np.concatenate((self.random.uniform(-1.0, 1.0, 6), [0,0,0]))
    physics.named.data.qpos[[
      'jaco_joint_1',
      'jaco_joint_2',
      'jaco_joint_3',
      'jaco_joint_4',
      'jaco_joint_5',
      'jaco_joint_6',
      'jaco_joint_finger_1',
      'jaco_joint_finger_2',
      'jaco_joint_finger_3']] = real_to_sim(_HOME_POSE)
    # print(real_to_sim(_JOINT_L))
    # print(real_to_sim(_JOINT_H))
    # print(real_to_sim(_HOME_POSE))
    physics.named.model.dof_damping[['jaco_joint_1',
      'jaco_joint_2',
      'jaco_joint_3',
      'jaco_joint_4',
      'jaco_joint_5',
      'jaco_joint_6']] = damping_with_noise(self.random)

  def get_observation(self, physics):
    """Returns an observation of the (bounded) physics state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()[0:9]
    obs['to_target'] = physics.finger_to_target()
    obs['target'] = physics.target_pos()
    obs['ef_rot'] = physics.named.data.site_xmat['palm'].reshape(3,3).dot([0,0,-1])
    # obs['velocity'] = physics.velocity()[0:9]
    return obs

  def before_step(self, action, physics):
    super(JacoReacher, self).before_step(action, physics)
    physics.action_cost = _ACTION_COST_D * np.square(action).sum()

  def get_reward(self, physics):
    """Returns a sparse or a smooth reward, as specified in the constructor."""
    reward = -physics.finger_to_target_distance()
    reward -= physics.action_cost
    reward -= physics.pose_penalty()

    return reward
