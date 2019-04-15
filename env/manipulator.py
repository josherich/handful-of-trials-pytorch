from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from dm_control import suite

import inspect
import jaco.dm_control2gym as dm_control2gym

LOCAL_DOMAINS = {name: module for name, module in locals().items()
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}

suite._DOMAINS = {**suite._DOMAINS, **LOCAL_DOMAINS}

env = dm_control2gym.make(domain_name="manipulator", task_name="bring_ball")

# OrderedDict([(
#   'arm_pos', array([[ 0.50070737, -0.86561662],
#        [ 0.9883704 ,  0.15206559],
#        [ 0.79097832,  0.61184417],
#        [ 0.41500005,  0.90982139],
#        [ 0.31807409,  0.94806586],
#        [-0.24375409,  0.96983707],
#        [ 0.33368303,  0.94268533],
#        [ 0.08459313,  0.99641558]])), 
# ('arm_vel', array([-2.84969089,  0.04893121, -3.3700001 ,  0.55957788,  1.48381607,
#         0.24666598,  1.95009571, -0.08496249])), 
# ('touch', array([0., 0., 0., 0., 0.])), 
# ('hand_pos', array([ 0.14347651,  0.17819177, -0.8988051 , -0.43834847])), 
# ('object_pos', array([-0.37220607,  0.50184697, -0.96964297,  0.24452509])), 
# ('object_vel', array([-1.41140956e+00, -1.76580000e+00,  4.21376412e-14])), 
# ('target_pos', array([0.03452618, 0.39604538, 0.57322216, 0.8194    ]))])