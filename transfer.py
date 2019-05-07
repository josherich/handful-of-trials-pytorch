from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint
import time
import ipdb
import collections
import numpy as np

from dotmap import DotMap
from config import create_config
from DotmapUtils import get_required_argument

from jaco.jacoEnv import env
from MPC import MPC
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import sys
import math
sys.path.append('../kinova-raw')
sys.path.append('../jaco-simulation')

import kinova
import cv2
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800, 600)

# where the robot has to be (in kinova coordinates)
# to be at zero in mujoco
zero_offset = np.array([-180, 270, 90, 180, 180, -90, 0, 0, 0])


# correct for the different physical directions of a +theta
# movement between mujoco
directions = np.array([-1, 1, -1, -1, -1, -1, 1, 1, 1])


# correct for the degrees -> radians shift going from arm
# to mujoco
scales = np.array([math.pi / 180] * 6 + [0.78 / 6800] * 3)

def get_jaco_angles():
    pos = kinova.get_angular_position()
    angles = [a for a in pos.Actuators] + [a for a in pos.Fingers]
    return angles

def move_mujoco_to_real(env):
    angles = get_jaco_angles()
    env.dmcenv.physics.named.data.qpos[:9] = real_to_sim(angles)

def real_to_sim(angles):

    return (angles - zero_offset) * directions * scales

def sim_to_real(angles):
      return (angles / (directions * scales)) + zero_offset

kinova.start()

def agent_sample(env, horizon, policy, record_fname):
    solution = None
    video_record = record_fname is not None
    recorder = None if not video_record else VideoRecorder(env, record_fname)

    times, rewards = [], []
    O, A, reward_sum, done = [env.reset()], [], 0, False
    J_O, J_A, J_reward_sum, J_done = [env.reset()], [], 0, False
    
    real_c = kinova.get_cartesian_position()
    sim_c = env.dmcenv.physics.named.data.site_xpos['palm']
    print("Real pos:", real_c)
    print("sim pos:", sim_c)
    # move_mujoco_to_real(env)
    policy.reset()

    for t in range(horizon):
        if video_record:
            recorder.capture_frame()

        start = time.time()
        # print(O)
        # print(t)
        solution = policy.act(O[t], t)
        A.append(solution)

        times.append(time.time() - start)

        # === Do action on real jaco ===
        kinova.move_angular_delta(A[t][0:6]/directions[0:6])
        # move_jaco_real(A[t] + O[t][0:9])

        # print("ac: ", A[t] + O[t][0:9])
        obs, reward, done, info = env.step(A[t] + O[t][0:9])
        # print("obs: ", obs)
        # print("reward: ", reward)

        # === Get obs from real jaco ===
        real_c = kinova.get_cartesian_position()
        sim_c = env.dmcenv.physics.named.data.site_xpos['palm']
        print("Real pos:", real_c)
        print("sim pos:", sim_c)
        angles = get_jaco_angles()
        obs_angles = real_to_sim(angles)
        # print("real_obs", obs_angles)
        # O[t][0:9] = obs_angles
        obs[0:9] = obs_angles
        obs[9:12] = obs[12:15] - real_c.Coordinates[0:3]
        # === sync ===
        # move_mujoco_to_real(env)

        screen = env.render(mode='rgb_array')
        cv2.imshow('image', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

        O.append(obs)
        # reward_sum += reward
        # rewards.append(reward)
        # if done:
        #     break

        # === stop ===

    if video_record:
        recorder.capture_frame()
        recorder.close()

# jaco-lubb, with velocity
# jaco-lub-novel, without velocity
def main(env, ctrl_type, ctrl_args, overrides, model_dir, logdir):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})

    overrides.append(["ctrl_cfg.prop_cfg.model_init_cfg.model_dir", model_dir])
    overrides.append(["ctrl_cfg.prop_cfg.model_init_cfg.load_model", "True"])
    overrides.append(["ctrl_cfg.prop_cfg.model_pretrained", "True"])
    overrides.append(["exp_cfg.exp_cfg.ninit_rollouts", "0"])
    overrides.append(["exp_cfg.exp_cfg.ntrain_iters", "1"])
    overrides.append(["exp_cfg.log_cfg.nrecord", "1"])
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    cfg.pprint()

    env = get_required_argument(cfg.exp_cfg.sim_cfg, "env", "Must provide environment.")
    # 150 for Jaco
    task_hor = get_required_argument(cfg.exp_cfg.sim_cfg, "task_hor", "Must provide task horizon.")
    policy = MPC(cfg.ctrl_cfg)

    agent_sample(env, task_hor, policy, "transfer.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True)
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[])
    parser.add_argument('-o', '--override', action='append', nargs=2, default=[])
    parser.add_argument('-model-dir', type=str, required=True)
    parser.add_argument('-logdir', type=str, required=True)
    args = parser.parse_args()

    main(args.env, "MPC", args.ctrl_arg, args.override, args.model_dir, args.logdir)
