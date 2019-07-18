from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import json
import os
import argparse
import pprint

from dotmap import DotMap

from easy_logger import logger
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
        ["exp_cfg.log_cfg.neval", 5],
        ["exp_cfg.log_cfg.nrecord_eval_mode", 1],
        ["exp_cfg.log_cfg.neval_eval_mode", 3],
        # ["exp_cfg.log_cfg.nrecord", 2],
        # ["exp_cfg.log_cfg.neval", 5],
        # ["exp_cfg.log_cfg.neval_eval_mode", 5],
        ["exp_cfg.exp_cfg.ntrain_iters", 200],
        # ["ctrl_cfg.opt_cfg.plan_hor", 1],
        # ["ctrl_cfg.opt_cfg.cfg.popsize", 5],
        # ["ctrl_cfg.opt_cfg.cfg.num_elites", 2],
        # ["ctrl_cfg.opt_cfg.cfg.max_iters", 1],
        ["ctrl_cfg.opt_cfg.cfg.popsize", "200"],
        ["ctrl_cfg.opt_cfg.cfg.num_elites", "20"],
        ["ctrl_cfg.opt_cfg.cfg.max_iters", "5"],
        # ["ctrl_cfg.opt_cfg.init_var_scale", "4."],
    ]
    config_module_kwargs = {
        'steps_needed_to_solve': steps_needed_to_solve,
        'planning_horizon': planning_horizon,
        'task_horizon_factor': 4,
    }
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir,
                        config_module_kwargs)
    # HACK
    cfg.ctrl_cfg.opt_cfg.task_hor = cfg.exp_cfg.sim_cfg.task_hor
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

    logger.reset()
    logger.set_snapshot_dir(exp.logdir)
    logger.add_tabular_output(os.path.join(exp.logdir, 'progress.csv'))
    logger.log_variant(os.path.join(exp.logdir, 'variant.json'), config_dict)

    print("log dir:", exp.logdir)

    exp.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', type=int, default=8)
    parser.add_argument('-logdir', type=str, default='log/test',
                        help='Directory to which results will be logged (default: ./log)')
    args = parser.parse_args()
    exp(10, 10, args.logdir)
    exp(15, 15, args.logdir)
    exp(20, 20, args.logdir)
    # exp(5, 5, args.logdir)
    # for planning_H in [4, 8, 16]:
    #     exp(5, planning_H, args.logdir)
    # for planning_H in [4, 8, 16, 20]:
    #     exp(8, planning_H, args.logdir)

    # for planning_H in [8, 'half', 'same']:
    #     for required_H in [8, 16, 32, 64]:
    #         if planning_H == 'half':
    #             planning_H = required_H // 2
    #         elif planning_H == 'same':
    #             planning_H = required_H
    #         exp(required_H, planning_H, args.logdir)
