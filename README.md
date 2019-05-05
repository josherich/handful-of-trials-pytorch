# Transfer learning on Kinova Jaco using PETS, TD3

This repo includes some modifications to do transfer learning on Jaco robotic arms

1. Mujoco model for
    - [delta position control](https://github.com/josherich/handful-of-trials-pytorch/blob/master/jaco/jaco_pos.xml)
    - [motor control](https://github.com/josherich/handful-of-trials-pytorch/blob/master/jaco/jaco_motor.xml)

2. [Jaco environment for gym, dm_control](https://github.com/josherich/handful-of-trials-pytorch/blob/master/jaco/jaco.py)

3. [Jaco module for PETS](https://github.com/josherich/handful-of-trials-pytorch/blob/master/config/jaco.py)

4. [IK solver](https://github.com/josherich/handful-of-trials-pytorch/blob/master/Physics.py)

5. [Script](https://github.com/josherich/handful-of-trials-pytorch/blob/master/transfer.py) for running transfer experiments

### Train
```bash
python mbexp.py -env jaco
```

### Run and Render
```bash
python render.py -env jaco -model-dir path/to/model -logdir path/to/log
```

### Run transfer experiment
```bash
python transfer.py -env jaco -model-dir path/to/model -logdir path/to/log
```

### Run IK solver
```bash
python mbexp.py -env jaco -physics
```

---

This repo contains a pytorch implementation of the wonderful model-based Reinforcement Learning algorithms proposed in [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114).

As of now, the repo only supports the most high-performing variant: probabilistic ensemble for the learned dynamics model, TSinf trajectory sampling and Cross Entropy method for action optimization.

The code is structured with the same levels of abstraction as the original TF implementation, with the exception that the TF dynamics model is replaced by a Pytorch dynamics model.

I'm happy to take pull request if you see ways to improve the repo :).

## Performance

![](graphs/cartpole.png) ![](graphs/pusher.png)

![](graphs/reacher.png)

The y-axis indicates the maximum reward seen so far, as is done in the paper.

## Requirements

1. The requirements in the original [TF implementation](https://github.com/kchua/handful-of-trials)
2. Pytorch 1.0.0

For specific requirements, please take a look at the pip dependency file `requirements.txt` and conda dependency file `environments.yml`.

## Install

1. install mujoco 2.0

```
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
mv mujoco200_linux.zip ~/.mujoco
```

2. install dm_control

```
pip install git+git://github.com/deepmind/dm_control.git
```

3. install dm_control2gym

```
git clone https://github.com/martinseilair/dm_control2gym
pip install .
```

4. install dependencies

```
pip install -r requirements.txt
```

## Running Experiments

Experiments for a particular environment can be run using:

```
python mbexp.py
    -env    ENV       (required) The name of the environment. Select from
                                 [cartpole, reacher, pusher, halfcheetah, jaco, manipulator].
```

Results will be saved in `<logdir>/<date+time of experiment start>/`.
Trial data will be contained in `logs.mat`, with the following contents:

```
{
    "observations": NumPy array of shape
        [num_train_iters * nrollouts_per_iter + ninit_rollouts, trial_lengths, obs_dim]
    "actions": NumPy array of shape
        [num_train_iters * nrollouts_per_iter + ninit_rollouts, trial_lengths, ac_dim]
    "rewards": NumPy array of shape
        [num_train_iters * nrollouts_per_iter + ninit_rollouts, trial_lengths, 1]
    "returns": Numpy array of shape [1, num_train_iters * neval]
}
```

To visualize the result, please take a look at `plotter.ipynb`

## Render Results

```
python render.py -env ENV -model-dir path/to/model/ -logdir path/to/log
```

## Acknowledgement

Huge thank to the authors of the paper for open-sourcing their [code](https://github.com/kchua/handful-of-trials/). Most of this repo is taken from the official TF implementation.
