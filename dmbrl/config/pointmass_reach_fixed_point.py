from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym

from dmbrl.config.pointmass_base import PointmassBaseConfigModule
from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.envs.pygame import register_custom_envs
import numpy as np
import tensorflow as tf

register_custom_envs()


class PointmassReachFixedPointConfigModule(PointmassBaseConfigModule):
    TASK_HORIZON = 100
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 10
    MODEL_IN, MODEL_OUT = 4, 2
    GP_NINDUCING_POINTS = 200

    GOAL_NP = np.array([1, 1])

    def __init__(self):
        super().__init__()
        env = gym.make('Point2DFixedGoalEnv-v0')
        env = FlatGoalEnv(env, append_goal_to_obs=False)
        self.ENV = env

    @staticmethod
    def obs_cost_fn(obs):
        if isinstance(obs, np.ndarray):
            return np.sum(
                np.square(obs - PointmassReachFixedPointConfigModule.GOAL_NP),
                axis=1,
            )
        else:
            return tf.reduce_sum(
                tf.square(obs - PointmassReachFixedPointConfigModule.GOAL_NP),
                axis=1,
            )
        # if isinstance(obs, np.ndarray):
        #     return -np.exp(-np.sum(
        #         np.square(obs - PointmassReachFixedPointConfigModule.GOAL_NP),
        #         axis=1,
        #     ))
        # else:
        #     return -tf.exp(-tf.reduce_sum(
        #         tf.square(obs - PointmassReachFixedPointConfigModule.GOAL_NP),
        #         axis=1,
        #     ))


CONFIG_MODULE = PointmassReachFixedPointConfigModule
