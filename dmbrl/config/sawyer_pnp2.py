from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
import dmbrl.env
from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.envs.mujoco import register_custom_envs
register_custom_envs()


class SawyerPnP2ConfigModule:
    ENV_NAME = 'SawyerPickAndPlaceOneObjTallWallTrainEnv-v2'
    TASK_HORIZON = 100
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 17, 14
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        self.ENV = FlatGoalEnv(gym.make(self.ENV_NAME), append_goal_to_obs=True)
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {
            "epochs": 1,
            "max_num_batches_total": 200,
        }
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    def obs_cost_fn(self, obs):
        if isinstance(obs, np.ndarray):
            return np.linalg.norm(
                np.square(obs[:, :7] - obs[:, 7:]),
                axis=1,
            )
        else:
            return tf.norm(
                obs[:, :7] - obs[:, 7:],
                axis=1,
                )

    @staticmethod
    def ac_cost_fn(acs):
        return 0

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(200, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.00025))
            model.add(FC(200, activation="swish", weight_decay=0.0005))
            model.add(FC(200, activation="swish", weight_decay=0.0005))
            model.add(FC(self.MODEL_OUT, weight_decay=0.00075))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model

    def gp_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model",
            kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),
            kernel_args=model_init_cfg.get("kernel_args", {}),
            num_inducing_points=get_required_argument(
                model_init_cfg, "num_inducing_points", "Must provide number of inducing points."
            ),
            sess=self.SESS
        ))
        return model


CONFIG_MODULE = SawyerPnP2ConfigModule
