from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym

from dmbrl.config.pointmass_base import PointmassBaseConfigModule
from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.envs.pygame import register_custom_envs

register_custom_envs()


class PointmassNoWallConfigModule(PointmassBaseConfigModule):
    TASK_HORIZON = 100
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 10
    MODEL_IN, MODEL_OUT = 6, 4
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        super().__init__()
        env = gym.make('Point2DLargeEnv-offscreen-v0')
        env = FlatGoalEnv(env, append_goal_to_obs=True)
        self.ENV = env


CONFIG_MODULE = PointmassNoWallConfigModule
