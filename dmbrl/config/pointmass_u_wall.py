from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import math
import tensorflow as tf

from dmbrl.config.pointmass_base import PointmassBaseConfigModule
from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.envs.pygame import register_custom_envs

register_custom_envs()


class PointmassUWallConfigModule(PointmassBaseConfigModule):
    # TASK_HORIZON = 100
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    # NUM_STEPS_TOTAL = int(2**16)
    NUM_STEPS_TOTAL = int(2**8)
    PLAN_HOR = 10
    MODEL_IN, MODEL_OUT = 6, 4
    GP_NINDUCING_POINTS = 200
    PATH_LENGTH_TO_SOLVE = 16.

    def __init__(self, steps_needed_to_solve, planning_horizon,
                 task_horizon_factor=2):
        env = gym.make("PointmassUWallTrainEnvBig-v1")
        env.action_scale = self.PATH_LENGTH_TO_SOLVE / steps_needed_to_solve
        env = FlatGoalEnv(env, append_goal_to_obs=True)
        PointmassUWallConfigModule.TASK_HORIZON = int(
            task_horizon_factor * steps_needed_to_solve
        )
        PointmassUWallConfigModule.PLAN_HOR = planning_horizon
        PointmassUWallConfigModule.NROLLOUTS_PER_ITER = math.ceil(
            PointmassUWallConfigModule.NUM_STEPS_TOTAL / (
                PointmassUWallConfigModule.TASK_HORIZON *
                PointmassUWallConfigModule.NTRAIN_ITERS
            )
        )
        print('-------------')
        print("task horizon", PointmassUWallConfigModule.TASK_HORIZON)
        print("plan horizon", PointmassUWallConfigModule.PLAN_HOR)
        print("nrolls per iter", PointmassUWallConfigModule.NROLLOUTS_PER_ITER)
        print("action_scale", env.wrapped_env.action_scale)
        print('-------------')
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

