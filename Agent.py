from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
from dotmap import DotMap
from gym.wrappers.monitoring.video_recorder import VideoRecorder

render_video = True

if render_video:
    import cv2
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 800, 600)

class Agent:
    """An general class for RL agents.
    """

    def __init__(self, params):
        """Initializes an agent.

        Arguments:
            params: (DotMap) A DotMap of agent parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the
                    action noise if params.noisy_actions is True.
        """
        assert params.get("noisy_actions", False) is False
        self.env = params.env

        if isinstance(self.env, DotMap):
            raise ValueError("Environment must be provided to the agent at initialization.")

    def sample(self, horizon, policy, record_fname=None):
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.
            record_fname: (str/None) The name of the file to which a recording of the rollout
                will be saved. If None, the rollout will not be recorded.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
        solution = None
        video_record = record_fname is not None
        recorder = None if not video_record else VideoRecorder(self.env, record_fname)

        times, rewards = [], []
        O, A, reward_sum, done = [self.env.reset()], [], 0, False

        policy.reset()
        if policy.__class__.__name__ == "Physics":
            solution = policy.act(O[0], self.env.dmcenv)

        for t in range(horizon):
            if video_record:
                recorder.capture_frame()
            start = time.time()
            if policy.__class__.__name__ == "Physics":
                force = solution - O[t][0:9]
                A.append(force * 5)
            else:
                solution = policy.act(O[t], t)
                A.append(solution)

            times.append(time.time() - start)
            if hasattr(self.env, 'action_mode') and self.env.action_mode is "delta":
                A[t] = self.env.after_action(A[t], O[t])
            obs, reward, done, info = self.env.step(A[t])

            if render_video:
                screen = self.env.render(mode='rgb_array')
                cv2.imshow('image', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
                if(cv2.waitKey(25) & 0xFF == ord('q')):
                    cv2.destroyAllWindows()
                    break

            O.append(obs)
            reward_sum += reward
            rewards.append(reward)
            if done:
                break

        if video_record:
            recorder.capture_frame()
            recorder.close()

        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }
