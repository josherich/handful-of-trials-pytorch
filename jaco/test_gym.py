import numpy as np

from dm_control import suite
import jaco
import pyglet

import inspect
import ipdb
import dm_control2gym
import cv2

LOCAL_DOMAINS = {name: module for name, module in locals().items()
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}

suite._DOMAINS = {**suite._DOMAINS, **LOCAL_DOMAINS}
# env = suite.load(domain_name="jaco", task_name="basic")

env = dm_control2gym.make(domain_name="jaco", task_name="basic")
# ipdb.set_trace()

# env = jaco.basic()

# Iterate over a task set:
# for domain_name, task_name in suite.BENCHMARKING:
#   env = suite.load(domain_name, task_name)

width = 320
height = 240
# window = pyglet.window.Window(width=width, height=height, display=None)


# Step through an episode and print out reward, discount and observation.
# action_spec = env.action_spec()
obs = env.reset()

done = False
while not done:
  for i in range(50):
    # action = env.action_space.sample()
    action = np.random.uniform(-20, 20, size=[9])
    new_obs, reward, done, _ = env.step(action)

    screen = env.render(mode='rgb_array')
    cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    if(cv2.waitKey(25) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        break

    # time_step = env.step(action)
    # import ipdb; ipdb.set_trace()
    # print(time_step.reward, time_step.discount, time_step.observation, env.physics.finger_to_target_distance())
    # print(reward, env.physics.finger_to_target_distance())
    # pixel1 = env.physics.render(height, width, camera_id=1)
    # pixel2 = env.physics.render(height, width, camera_id=2)
    # pixel = np.concatenate([pixel1, pixel2], 1)
    # cv2.imshow('arm', pixel)
    # cv2.waitKey(10)
