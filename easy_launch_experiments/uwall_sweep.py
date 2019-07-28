from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import argparse
import pythonplusplus.machine_learning.hyperparameter as hyp
import tensorflow as tf
from dotmap import DotMap
from easy_launcher.launcher_util import run_experiment

from dmbrl.config import create_config
from dmbrl.controllers.MPC import MPC
from dmbrl.misc.MBExp import MBExperiment


def exp(variant):
    from easy_logger import logger

    tf.reset_default_graph()
    config_module_kwargs = {
        'steps_needed_to_solve': variant['steps_needed_to_solve'],
        'planning_horizon': variant['steps_needed_to_solve'],
        'task_horizon_factor': 4,
    }
    cfg = create_config(
        env_name="pointmass_u_wall",
        ctrl_type="MPC",
        ctrl_args=DotMap(),
        overrides=[
            ["exp_cfg.log_cfg.nrecord", 1],
            ["exp_cfg.log_cfg.neval", 5],
            ["exp_cfg.log_cfg.nrecord_eval_mode", 1],
            ["exp_cfg.log_cfg.neval_eval_mode", 3],
            ["exp_cfg.exp_cfg.ntrain_iters", 500],
            ["ctrl_cfg.opt_cfg.cfg.popsize", "200"],
            ["ctrl_cfg.opt_cfg.cfg.num_elites", "20"],
            ["ctrl_cfg.opt_cfg.cfg.max_iters", "5"],
            # ["exp_cfg.exp_cfg.ntrain_iters", 2],
            # ["ctrl_cfg.opt_cfg.cfg.popsize", "2"],
            # ["ctrl_cfg.opt_cfg.cfg.num_elites", "2"],
            # ["ctrl_cfg.opt_cfg.cfg.max_iters", "1"],
        ],
        logdir=logger.get_snapshot_dir(),
        config_module_kwargs=config_module_kwargs,
    )
    # HACK
    cfg.ctrl_cfg.opt_cfg.task_hor = cfg.exp_cfg.sim_cfg.task_hor
    cfg.exp_cfg.log_cfg.logdir = logger.get_snapshot_dir()

    cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
    exp = MBExperiment(cfg.exp_cfg)

    config_dict = cfg.toDict()
    config_dict['config_module_kwargs'] = config_module_kwargs
    logger.log_variant(
        os.path.join(exp.logdir, 'complete_variant.json'),
        config_dict,
    )

    print("log dir:", exp.logdir)

    exp.run_experiment()


def main():
    n_seeds = 1
    mode = 'local_docker'
    # mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )
    mode = 'here_no_doodad'
    exp_prefix = 'dev-time'

    n_seeds = 3
    mode = 'sss'
    exp_prefix = 'neurips-rebut-pets-new-uwall-5-steps-sss'

    search_space = {
        # 'steps_needed_to_solve': [5, 10, 20],
        'steps_needed_to_solve': [5],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=dict(
            steps_needed_to_solve=5,
        ),
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                exp,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                snapshot_mode='gap_and_last',
                snapshot_gap=25,
                use_gpu=True,
            )

if __name__ == "__main__":
    main()
