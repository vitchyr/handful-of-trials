from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config
import tensorflow as tf


def exp(steps_needed_to_solve, planning_horizon, logdir):
    tf.reset_default_graph()
    env = "pointmass_u_wall"
    ctrl_type = "MPC"
    ctrl_args = []
    overrides = [
        ["exp_cfg.log_cfg.nrecord", 1],
    ]
    config_module_kwargs = {
        'steps_needed_to_solve': steps_needed_to_solve,
        'planning_horizon': planning_horizon
    }
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir,
                        config_module_kwargs)
    cfg.pprint()

    if ctrl_type == "MPC":
        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
    exp = MBExperiment(cfg.exp_cfg)

    os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

    # exp.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', type=int, default=8)
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    args = parser.parse_args()
    # exp(8, 10, args.logdir)
    #
    # for planning_H in [8, 16, 32]:
    #     for H in [8, 16, 32, 64, 128]:
    #         exp(H, args.logdir)
