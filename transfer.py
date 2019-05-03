from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint
import time
import ipdb

from dotmap import DotMap
from config import create_config
from DotmapUtils import get_required_argument

from jaco.jaco_gym import env
from MPC import MPC
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# import kinova
# sys.path.append('../kinova-raw')
# sys.path.append('../jaco-simulation')

import cv2
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800, 600)


def get_jaco_angles():
    pos = kinova.get_angular_position()
    angles = [a for a in pos.Actuators] + [a for a in pos.Fingers]
    return angles

def move_mujoco_to_real(env):
    angles = get_jaco_angles()
    env.dmcenv.physics.named.data.qpos[:9] = real_to_sim(angles)

def real_to_sim(angles):
    # where the robot has to be (in kinova coordinates)
    # to be at zero in mujoco
    zero_offset = np.array([-180, 270, 90, 180, 180, -90, 0, 0, 0])

    # correct for the different physical directions of a +theta
    # movement between mujoco
    directions = np.array([-1, 1, -1, -1, -1, -1, 1, 1, 1])

    # correct for the degrees -> radians shift going from arm
    # to mujoco
    scales = np.array([math.pi / 180] * 6 + [0.78 / 6800] * 3)

    return (angles - zero_offset) * directions * scales


def agent_sample(env, horizon, policy, record_fname):
    solution = None
    video_record = record_fname is not None
    recorder = None if not video_record else VideoRecorder(env, record_fname)

    times, rewards = [], []
    O, A, reward_sum, done = [env.reset()], [], 0, False

    policy.reset()

    for t in range(horizon):
        if video_record:
            recorder.capture_frame()

        start = time.time()

        solution = policy.act(O[t], t)
        A.append(solution)

        times.append(time.time() - start)

        # === Do action on real jaco ===
        # move_jaco_real(A[t] + O[t][0:9])

        print("ac: ", A[t] + O[t][0:9])
        obs, reward, done, info = env.step(A[t] + O[t][0:9])
        print("obs: ", obs)
        print("reward: ", reward)

        # === Get obs from real jaco ===
        # angles = get_jaco_angles()
        # obs_angles = real_to_sim(angles)
        # O[t][0:9] = obs_angles

        # === sync ===
        # move_mojuco_to_real(env)

        screen = env.render(mode='rgb_array')
        cv2.imshow('image', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

        O.append(obs)
        reward_sum += reward
        rewards.append(reward)
        if done:
            break

        # === stop ===
        ipdb.set_trace()

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