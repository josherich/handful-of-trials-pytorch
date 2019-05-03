import numpy as np
import tensorflow as tf
import torch
from torch import nn as nn


def swish(x):
    return x * torch.sigmoid(x)


def truncated_normal(size, std):
    # We use TF to implement initialization function for neural network weight because:
    # 1. Pytorch doesn't support truncated normal
    # 2. This specific type of initialization is important for rapid progress early in training in cartpole

    # Do not allow tf to use gpu memory unnecessarily
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True

    sess = tf.Session(config=cfg)
    val = sess.run(tf.truncated_normal(shape=size, stddev=std))

    # Close the session and free resources
    sess.close()

    return torch.tensor(val, dtype=torch.float32)


def get_affine_params(ensemble_size, in_features, out_features):

    w = truncated_normal(size=(ensemble_size, in_features, out_features),
                         std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)

    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

    return w, b

# from scipy.stats import truncnorm
# import matplotlib.pyplot as plt

# TODO: compare with tf implementation
def truncated_normal_(size, std=1):
    mean=0
    tensor = torch.zeros(size)
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

# fig, ax = plt.subplots(1, 1)

# def test_truncnorm():
#     a, b = -2, 2
#     size = 1000000
#     r = truncnorm.rvs(a, b, size=size)
#     ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=50)

#     tensor = torch.zeros(size)
#     truncated_normal_(tensor)
#     r = tensor.numpy()

#     ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=50)
#     ax.legend(loc='best', frameon=False)
#     plt.show()


# if __name__ == '__main__':
#     test_truncnorm()
