from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC

from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.envs.pygame import register_custom_envs
register_custom_envs()


class PointmassBaseConfigModule(object):
    # Set these in concrete classes:
    # TASK_HORIZON       = ?
    # NTRAIN_ITERS       = ?
    # NROLLOUTS_PER_ITER = ?
    # PLAN_HOR           = ?
    # MODEL_IN, MODEL_OUT = ?, ?

    def __init__(self):
        cfg = tf.ConfigProto(log_device_placement=True)
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 10
            },
            "CEM": {
                "popsize":    400,
                "num_elites": 40,
                "max_iters":  5,
                "alpha":      0.1,
            }
        }
        self.UPDATE_FNS = []

        # Fill in other things to be done here.

    @staticmethod
    def obs_preproc(obs):
        return obs

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    @staticmethod
    def obs_cost_fn(obs):
        if isinstance(obs, np.ndarray):
            return np.sum(
                np.square(obs[:, :2] - obs[:, 2:]),
                axis=1,
            )
        else:
            return tf.reduce_sum(
                tf.square(obs[:, :2] - obs[:, 2:]),
                axis=1,
            )

    @staticmethod
    def ac_cost_fn(acs):
        if isinstance(acs, np.ndarray):
            return 0
        else:
            return 0

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(500, input_dim=self.MODEL_IN, activation='swish', weight_decay=0.0001))
            model.add(FC(500, activation='swish', weight_decay=0.00025))
            model.add(FC(500, activation='swish', weight_decay=0.00025))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0005))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model
