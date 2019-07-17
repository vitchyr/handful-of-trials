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


class PointmassFlatWallConfigModule(PointmassBaseConfigModule):
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    NUM_STEPS_TOTAL = int(2**16)
    PLAN_HOR = 10
    MODEL_IN, MODEL_OUT = 6, 4
    GP_NINDUCING_POINTS = 200
    PATH_LENGTH_TO_SOLVE = 20.

    def __init__(self, steps_needed_to_solve, planning_horizon,
                 task_horizon_factor=2):
        env = gym.make("PointmassFlatWallTrainEnvBig-v0")
        env.action_scale = self.PATH_LENGTH_TO_SOLVE / steps_needed_to_solve
        env = FlatGoalEnv(env, append_goal_to_obs=True)
        PointmassFlatWallConfigModule.TASK_HORIZON = int(
            task_horizon_factor * steps_needed_to_solve
        )
        PointmassFlatWallConfigModule.PLAN_HOR = planning_horizon
        PointmassFlatWallConfigModule.NROLLOUTS_PER_ITER = math.ceil(
            PointmassFlatWallConfigModule.NUM_STEPS_TOTAL / (
                PointmassFlatWallConfigModule.TASK_HORIZON *
                PointmassFlatWallConfigModule.NTRAIN_ITERS
            )
        )
        print('-------------')
        print("task horizon", PointmassFlatWallConfigModule.TASK_HORIZON)
        print("plan horizon", PointmassFlatWallConfigModule.PLAN_HOR)
        print("nrolls per iter", PointmassFlatWallConfigModule.NROLLOUTS_PER_ITER)
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


CONFIG_MODULE = PointmassFlatWallConfigModule

