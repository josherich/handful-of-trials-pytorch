from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import gym
import os

import torch
from torch import nn as nn
from torch.nn import functional as F

from config.utils import swish, get_affine_params
from DotmapUtils import get_required_argument

from jaco.jacoEnv import env
from jaco.jaco import position_penalty
from dm_control.mujoco.wrapper import mjbindings
mjlib = mjbindings.mjlib

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
_ACTION_COST_D = 0.0025


class PtModel(nn.Module):

    def __init__(self, ensemble_size, in_features, out_features):
        super().__init__()

        self.num_nets = ensemble_size

        self.in_features = in_features
        self.out_features = out_features

        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size, in_features, 200)

        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size, 200, 200)

        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size, 200, 200)

        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size, 200, out_features)

        self.inputs_mu = nn.Parameter(torch.zeros((1, in_features)), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros((1, in_features)), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)

    def compute_decays(self):

        lin0_decays = 0.00025 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.0005 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.0005 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.00075 * (self.lin3_w ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays

    def fit_input_stats(self, data):

        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(TORCH_DEVICE).float()

    def forward(self, inputs, ret_logvar=False):

        # Transform inputs
        inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        inputs = inputs.matmul(self.lin0_w) + self.lin0_b

        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b

        mean = inputs[:, :, :self.out_features // 2]

        logvar = inputs[:, :, self.out_features // 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)

    def save(self, directory):
        torch.save(self.state_dict(), '%s/model.pth' % (directory))

    def load(self, directory):
        state_dict = torch.load('%s/model.pth' % (directory),  map_location=lambda storage, loc: storage)
        # print(state_dict)
        self.load_state_dict(state_dict)

class JacoConfigModule:
    ENV_NAME = "MBRLJaco"
    TASK_HORIZON = 150
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 33, 24
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        self.ENV = env
        print(self.ENV.observation_space)
        print(self.ENV.action_space)
        self.ENV.reset()
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }
        self.UPDATE_FNS = [self.update_goal]

        self.goal = None

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    def update_goal(self):
        self.goal = None

    def obs_cost_fn(self, obs):

        assert isinstance(obs, torch.Tensor)

        obs = obs.detach().cpu().numpy()
        # position[0:8] target[9:11] velocity[12:20]
        # print(obs[:,9:12].mean())
        cost = np.sum(np.square(obs[:,9:12]), axis=1)
        cost -= obs[:,23]

        return torch.from_numpy(cost).float().to(TORCH_DEVICE)

    def pose_cost_fn(self, obs):
        pz_sum = 0
        physics = self.ENV.dmcenv.physics.copy(share_model=True)
        dtype = physics.data.qpos.dtype
        for o in obs:
            update_nv = np.array(o.cpu()[0:9], dtype=dtype)
            mjlib.mj_integratePos(physics.model.ptr, physics.data.qpos, update_nv, 1)
            mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

            joints_z = [
             physics.named.data.geom_xpos['jaco_joint_1', 'z'],
             physics.named.data.geom_xpos['jaco_joint_2', 'z'],
             physics.named.data.geom_xpos['jaco_joint_3', 'z'],
             physics.named.data.geom_xpos['jaco_joint_4', 'z'],
             physics.named.data.geom_xpos['jaco_joint_5', 'z'],
             physics.named.data.geom_xpos['jaco_joint_6', 'z'],
             physics.named.data.geom_xpos['jaco_link_fingertip_1', 'z'],
             physics.named.data.geom_xpos['jaco_link_fingertip_2', 'z'],
             physics.named.data.geom_xpos['jaco_link_fingertip_3', 'z'],
            ]
            pz_sum += position_penalty(physics, joints_z)
        return pz_sum

    @staticmethod
    def ac_cost_fn(acs):
        return _ACTION_COST_D * (acs ** 2).sum(dim=1)

    def nn_constructor(self, model_init_cfg):
        ensemble_size = get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size")

        load_model = model_init_cfg.get("load_model", False)
        model = PtModel(ensemble_size,
                        self.MODEL_IN, self.MODEL_OUT * 2).to(TORCH_DEVICE)
        if load_model:
            model_dir = model_init_cfg.get("model_dir", None)
            model.load(model_dir)
        # * 2 because we output both the mean and the variance
        model.optim = torch.optim.Adam(model.parameters(), lr=0.001)

        return model


CONFIG_MODULE = JacoConfigModule