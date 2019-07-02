from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import json
import os
import argparse
import pprint

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config
import tensorflow as tf

from dmbrl.util import save_git_info, MyEncoder


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
    config_dict = cfg.toDict()
    config_dict['config_module_kwargs'] = config_module_kwargs
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(config_dict))
    with open(os.path.join(exp.logdir, "variant.json"), "w") as f:
        json.dump(config_dict, f, indent=2, sort_keys=True, cls=MyEncoder)
    save_git_info(exp.logdir)

    print("log dir:", exp.logdir)

    exp.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', type=int, default=8)
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    args = parser.parse_args()
    for planning_H in [4, 8, 16]:
        exp(8, planning_H, args.logdir)
    #
    # for planning_H in [8, 'half', 'same']:
    #     for required_H in [8, 16, 32, 64]:
    #         if planning_H == 'half':
    #             planning_H = required_H // 2
    #         elif planning_H == 'same':
    #             planning_H = required_H
    #         exp(required_H, planning_H, args.logdir)
