from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from dotmap import DotMap

import time

try:
    from gym.monitoring import VideoRecorder
except ImportError:
    from dmbrl.monitor import VideoRecorder


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
        self.env = params.env
        self.noise_stddev = params.noise_stddev if params.get("noisy_actions", False) else None

        if isinstance(self.env, DotMap):
            raise ValueError("Environment must be provided to the agent at initialization.")
        if (not isinstance(self.noise_stddev, float)) and params.get("noisy_actions", False):
            raise ValueError("Must provide standard deviation for noise for noisy actions.")

        if self.noise_stddev is not None:
            self.dU = self.env.action_space.shape[0]

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
        video_record = record_fname is not None
        recorder = None if not video_record else VideoRecorder(self.env, record_fname)

        times, rewards = [], []
        O, A, reward_sum, done = [self.env.reset()], [], 0, False
        env_infos = []
        agent_infos = []

        policy.reset()
        # print("init obs {}".format(O[0]))
        for t in range(horizon):
            if video_record:
                recorder.capture_frame()
            start = time.time()
            action, predicted_cost = policy.act(O[t], t, get_pred_cost=True)
            A.append(action)
            agent_infos.append({
                'best_predicted_cost': predicted_cost
            })
            times.append(time.time() - start)

            if self.noise_stddev is None:
                action = A[t]
            else:
                action = A[t] + np.random.normal(loc=0, scale=self.noise_stddev, size=[self.dU])
                action = np.minimum(np.maximum(action, self.env.action_space.low), self.env.action_space.high)
            obs, reward, done, info = self.env.step(action)
            # print("obs {} action {} at time {}".format(obs, action, t))
            O.append(obs)
            env_infos.append(info)
            reward_sum += reward
            rewards.append(reward)
            if done:
                break

        if video_record:
            recorder.capture_frame()
            recorder.close()

        # print("Average action selection time: ", np.mean(times))
        # print("Rollout length: ", len(A))

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
            "env_infos": env_infos,
            "agent_infos": agent_infos,
        }
