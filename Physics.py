from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from dm_control.utils import inverse_kinematics as ik

class Controller:
    def __init__(self, *args, **kwargs):
        """Creates class instance.
        """
        pass

    def train(self, obs_trajs, acs_trajs, rews_trajs):
        """Trains this controller using lists of trajectories.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        """Resets this controller.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def act(self, obs, t, get_pred_cost=False):
        """Performs an action.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def dump_logs(self, primary_logdir, iter_logdir):
        """Dumps logs into primary log directory and per-train iteration log directory.
        """
        raise NotImplementedError("Must be implemented in subclass.")

class Physics(Controller):

    def __init__(self, params):
      super().__init__(params)

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        pass

    def act(self, obs, env):
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        """
        _TOL = 1e-14
        _MAX_STEPS = 100
        site_name = 'palm'
        target_pos = env.physics.named.data.geom_xpos['target']
        target_quat = env.physics.named.model.geom_quat['target']
        # print('target pos: ', target_pos)
        joint_names = ['jaco_joint_1','jaco_joint_2','jaco_joint_3','jaco_joint_4','jaco_joint_5','jaco_joint_6']
        result = ik.qpos_from_site_pose(
            physics=env.physics,
            site_name=site_name,
            target_pos=target_pos,
            joint_names=joint_names,
            tol=_TOL,
            max_steps=_MAX_STEPS,
            inplace=False)
        return result.qpos[0:9]
        # print('sol: ', result)
        # print('obs: ', obs)
        # print(env.physics.named.model.geom_pos)
        # print(env.physics.named.model.jnt_axis)
        # print(env.physics.named.model.geom_quat)
