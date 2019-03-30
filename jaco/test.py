import numpy as np

from dm_control import suite
import jaco
import pyglet

import inspect
# Load one task:
import cv2


# env = suite.load('cartpole', 'swingup')
LOCAL_DOMAINS = {name: module for name, module in locals().items()
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}
suite._DOMAINS = {**suite._DOMAINS, **LOCAL_DOMAINS}
env = suite.load(domain_name="jaco", task_name="basic")
# env = jaco.basic()
#
# Iterate over a task set:
# for domain_name, task_name in suite.BENCHMARKING:
#   env = suite.load(domain_name, task_name)

width = 640
height = 480
fullwidth = width * 2

# window = pyglet.window.Window(width=fullwidth, height=height)

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()

print(action_spec)

action = np.zeros([9])
time_step = env.step(action)

# def move_target_to_hand():
#   env.physics.named.model.geom_pos['target'] = env.physics.named.data.xpos['jaco_link_hand']
#
# def move_mocap_to_hand():
#   env.physics.named.data.mocap_pos['endpoint'] = env.physics.named.data.xpos['jaco_link_hand']
#
# def zero_mocap_offset():
#   env.physics.named.model.eq_data['weld'].fill(0)
#
# zero_mocap_offset()

cv2.namedWindow('arm', cv2.WINDOW_NORMAL)
cv2.resizeWindow('arm', fullwidth, height)

# env.physics.named.data.mocap_pos[0] = env.physics.named.data.xpos['jaco_link_hand']
while not time_step.last():
  action = np.random.uniform(-10, 10, size=[9])
  # env.physics.named.data.mocap_pos[0] = env.physics.named.data.geom_xpos['jaco_hand']


  for i in range(50):
    action = np.random.uniform(-10, 10, size=[9])
    time_step = env.step(action)
    # print(time_step.reward, time_step.discount, time_step.observation, env.physics.finger_to_target_distance())
    print(time_step.reward, env.physics.finger_to_target_distance(), time_step)
    pixel1 = env.physics.render(height, width, camera_id=1)
    pixel2 = env.physics.render(height, width, camera_id=2)
    pixel = np.concatenate([pixel1, pixel2], 1)
    cv2.imshow('arm', pixel)
    cv2.waitKey(10)
    # if i > 0:
    # window.switch_to()
    # window.clear()
    # if i == 0:
    # window.dispatch_events()
    # pyglet.image.ImageData(fullwidth, height, 'RGB', pixel.tobytes(), pitch=fullwidth * -3).blit(0, 0)
    # window.flip()
  # import ipdb; ipdb.set_trace()
  # env.physics.move_hand([1,0.5,0.5])
cv2.destroyAllWindows()


# TimeStep(step_type=<StepType.MID: 1>, 
#   reward=-0.5475899111478988, 
#   discount=1.0, 
#   observation=OrderedDict([
#     ('position', array([-0.00196379,  0.00387444, -0.02669908, -0.02630046,  0.00845395,
#         0.00378278, -0.00176875,  0.03072577, -0.00391754])), 
#     ('to_target', array([-0.47246855, -0.25489203, -0.10797329])), 
#     ('velocity', array([ 0.41572338,  0.11839176, -5.86221252, -0.74216021, -6.64653205,
#        -9.30152644, -3.27251714,  3.95746543,  9.43523292]))]))
