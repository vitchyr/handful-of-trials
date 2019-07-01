from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf

from dmbrl.config.pointmass_base import PointmassBaseConfigModule
from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.envs.pygame import register_custom_envs

register_custom_envs()


class PointmassUWallConfigModule(PointmassBaseConfigModule):
    TASK_HORIZON       = 5
    NTRAIN_ITERS       = 10
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR           = 5
    MODEL_IN, MODEL_OUT = 6, 4

    def __init__(self):
        env = gym.make("PointmassUWallTestEnvBig-v1")
        env = FlatGoalEnv(env, append_goal_to_obs=True)
        self.ENV = env
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 2}
        self.OPT_CFG = {
            "Random": {
                "popsize": 10
            },
            "CEM": {
                "popsize":    5,
                "num_elites": 2,
                "max_iters":  2,
                "alpha":      0.1,
            }
        }
        self.UPDATE_FNS = []

CONFIG_MODULE = PointmassUWallConfigModule

