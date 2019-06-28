from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym

from dmbrl.config.pointmass import PointmassConfigModule
from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC

from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.envs.pygame import register_custom_envs
register_custom_envs()


class PointmassNoWallConfigModule(PointmassConfigModule):
    ENV_NAME           = 'Point2DLargeEnv-offscreen-v0'
    TASK_HORIZON       = 20
    NTRAIN_ITERS       = 20
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR           = 10
    MODEL_IN, MODEL_OUT = 6, 4

    def __init__(self):
        super().__init__()
        env = gym.make(self.ENV_NAME)
        env = FlatGoalEnv(env, append_goal_to_obs=True)
        self.ENV = env

CONFIG_MODULE = PointmassNoWallConfigModule

